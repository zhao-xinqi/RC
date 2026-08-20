"""
test.py - RDK S100 摄像头采集/帧率测试
仅采集 + imshow, 画面叠加实时 FPS。用 v4l2-ctl 改参数冲高帧率:
  MJPG 压缩编码, 显式设帧率, 手动短曝光, 关闭自动曝光/白平衡/对焦。
无桌面(SSH)环境下 imshow 不可用, 自动降级为控制台打印 FPS。

运行: python test.py [--device /dev/video0] [--width 640] [--height 480]
                     [--fps 400] [--exposure-ms 5] [--auto-exposure] [--no-v4l2ctl] [--headless]
手动对照: v4l2-ctl --device /dev/video0 --list-formats-ext   # 看各分辨率支持的最高帧率

排查"锁60fps": 代码本身不锁帧。若 --headless 也是 60, 说明采集链路只出 60,
用下面命令绕过 OpenCV 直接量摄像头真实出帧率 (输出会打印实际 fps):
  v4l2-ctl --device /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG --set-parm=400
  v4l2-ctl --device /dev/video0 --stream-mmap --stream-count=200
"""

import argparse
import subprocess
import time
import cv2


def run(cmd):
    """执行 shell 命令, 返回 (退出码, 输出). 命令不存在返回 (-1, '')."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr
    except OSError:
        return -1, ''


def ctl(device, arg):
    """执行单条 v4l2-ctl 参数命令并打印结果, 返回是否成功."""
    code, _ = run('v4l2-ctl --device %s %s' % (device, arg))
    print('  [v4l2-ctl] %s : %s' % (arg, 'OK' if code == 0 else '失败(驱动可能不支持)'))
    return code == 0


def tune_camera(device, w, h, exp_ms, auto_exp):
    """v4l2-ctl 调参: 格式(MJPG)/曝光/关闭自动功能 (各项独立容错).

    帧率不在打开前设: OpenCV 打开时会重新协商格式并重置帧间隔,
    帧率统一放到 OpenCV 打开后再补 (见 main 中 --set-parm).
    """
    ctl(device, '--set-fmt-video=width=%d,height=%d,pixelformat=MJPG' % (w, h))
    if auto_exp:
        ctl(device, '--set-ctrl=exposure_auto=0')                      # 0=自动
    else:
        ctl(device, '--set-ctrl=exposure_auto=1')                      # 1=手动
        if not ctl(device, '--set-ctrl=exposure_absolute=%d' % int(exp_ms * 10)):
            ctl(device, '--set-ctrl=exposure=%d' % int(exp_ms * 1000))  # 部分驱动用微秒
    ctl(device, '--set-ctrl=white_balance_temperature_auto=0')
    ctl(device, '--set-ctrl=focus_auto=0')


def main():
    ap = argparse.ArgumentParser(description='摄像头采集/帧率测试')
    ap.add_argument('--device', default='/dev/video0', help='摄像头设备')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=400, help='目标帧率(取摄像头支持的上限)')
    ap.add_argument('--exposure-ms', type=float, default=5.0,
                    help='手动曝光ms, 越小帧率越高但画面越暗')
    ap.add_argument('--auto-exposure', action='store_true', help='保持自动曝光')
    ap.add_argument('--no-v4l2ctl', action='store_true', help='跳过v4l2-ctl, 纯OpenCV')
    ap.add_argument('--headless', action='store_true',
                    help='不显示窗口, 只测纯采集帧率(排查锁60fps)')
    a = ap.parse_args()

    dev = a.device if not a.device.isdigit() else '/dev/video%d' % int(a.device)
    print('设备: %s' % dev)

    if not a.no_v4l2ctl:
        _, out = run('v4l2-ctl --device %s --list-formats-ext' % dev)
        print(out or '[v4l2-ctl] 未安装, 跳过模式查询')
        tune_camera(dev, a.width, a.height, a.exposure_ms, a.auto_exposure)

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError('无法打开摄像头: %s' % dev)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    fps_ok = cap.set(cv2.CAP_PROP_FPS, a.fps)   # OpenCV 侧请求帧率

    # 兜底: OpenCV 打开后再补一次 --set-parm (部分驱动下帧率状态跨fd生效,
    # OpenCV 自己的 fps 请求被忽略时, 这一刀能让流真正跑目标帧率)
    if not a.no_v4l2ctl:
        ctl(dev, '--set-parm=%d' % a.fps)

    print('请求 %dfps: OpenCV接受=%s, 驱动协商=%g'
          % (a.fps, fps_ok, cap.get(cv2.CAP_PROP_FPS)))

    # --headless 或 无桌面环境时降级为控制台打印 FPS (纯采集, 不受显示刷新率门控)
    show = not a.headless
    if show:
        try:
            cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
        except cv2.error:
            show = False
            print('[WARN] 无显示窗口, 改控制台打印 FPS')

    ema_fps, last, idx, t0 = 0.0, time.perf_counter(), 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.perf_counter()
            ema_fps = 0.9 * ema_fps + 0.1 / max(now - last, 1e-6)   # 帧率滑动平均
            last, idx = now, idx + 1

            if show:
                cv2.putText(frame, '%.1f fps' % ema_fps, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.imshow('cam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif idx % 30 == 0:
                print('frame=%d fps=%.1f' % (idx, ema_fps))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        print('平均fps=%.1f' % (idx / max(time.perf_counter() - t0, 1e-6)))


if __name__ == '__main__':
    main()
