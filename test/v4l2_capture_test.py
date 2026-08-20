"""
v4l2_capture_test.py - 纯 v4l2 摄像头测试 (RDK S100 板端运行)

功能:
  1. 查看支持格式 (--list-formats-ext) 和 可用控制项 (--list-ctrls);
  2. 用 v4l2-ctl 直接采图 (绕过 OpenCV), MJPG 帧拆成 jpg, 打印实测帧率;
  3. --benchmark 逐分辨率实测持续帧率。

已知情况 (640x480 MJPG):
  - 帧率随曝光时间波动: 自动曝光暗环境 -> ~60fps, 亮一些 -> 70~80fps;
  - 本相机可能没有 exposure_auto 等标准曝光控制 (会打印 'unknown control' 警告),
    曝光控制改为独立命令执行, 失败不影响采流测量;
  - 标称 400fps 需极短曝光且未必能持续。

注意: 帧率只信 v4l2-ctl 实测值; 短采样或命令失败统一报"未知/不支持",
不会把失败耗时除出荒谬的几万fps。测持续帧率建议 --count 大一点 + --skip。

运行示例:
  python3 v4l2_capture_test.py                          # 默认 MJPG 640x480 采100帧
  python3 v4l2_capture_test.py --count 400
  python3 v4l2_capture_test.py --exposure-ms 2.5        # 手动短曝光(仅当驱动支持)
  python3 v4l2_capture_test.py --benchmark
"""

import argparse
import os
import re
import subprocess
import time


# ================================================================
# 基础工具
# ================================================================

def run(cmd):
    """执行 shell 命令, 返回 (退出码, 输出文本)."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr
    except OSError:
        return -1, ''


def v4l2_ctl(dev, *args):
    """拼接执行单条 v4l2-ctl 命令: v4l2-ctl --device <dev> <args...>."""
    return run('v4l2-ctl --device %s %s' % (dev, ' '.join(args)))


def parse_fps(text):
    """取 v4l2-ctl 输出里的实测帧率.

    必须排除 'Frame rate set to 400.000 fps' 这类 --set-parm 的请求值回显,
    否则短采样 (v4l2-ctl 未打印实测线) 时会误报成请求帧率.
    """
    text = re.sub(r'Frame rate set to\s+[\d.]+ fps', '', text)
    m = re.findall(r'([\d.]+) fps', text)
    return float(m[-1]) if m else None


def stream_failed(out):
    """粗判 v4l2-ctl 输出里流是否没有按请求设置成功."""
    t = out.lower()
    return any(k in t for k in ('failed', 'cannot ', 'unable to', 'no space left'))


def stream_fps(dev, fmt_opts, count, skip, raw=None):
    """单条命令 设格式+流式采帧, 返回 (退出码, 输出, 实测帧率).

    --stream-mmap 是真正开流的关键: 没有它 v4l2-ctl 只设置不采帧, 测不出帧率。
    帧率只信 v4l2-ctl 实测值; 命令失败/流未跑起来时返回 None (绝不瞎估),
    避免把"失败瞬间返回的极短耗时"除出一个荒谬的几万fps。
    raw 不为 None 时同时把原始流落盘到该文件 (供后续拆帧).
    """
    opts = fmt_opts + ['--stream-mmap --stream-skip=%d --stream-count=%d' % (skip, count)]
    if raw:
        opts.append('--stream-to=%s' % raw)
    t0 = time.perf_counter()
    code, out = v4l2_ctl(dev, *opts)
    elapsed = time.perf_counter() - t0

    if code != 0 or stream_failed(out):
        return code, out, None          # 命令失败: 不报告帧率

    fps = parse_fps(out)
    # 兜底: 仅当命令成功、耗时可信 (排除启动开销占比大) 才按墙钟估算
    if fps is None and elapsed > 0.2 and count >= 50:
        fps = count / elapsed
    if fps is not None and fps > 2000:
        fps = None                      # 明显异常, 不可能是真实持续帧率
    return code, out, fps


# ================================================================
# 曝光控制 (独立命令, 失败不影响采流)
# ================================================================

def exposure_opts(exp_ms, auto_exp):
    """生成曝光控制参数列表 (v4l2-ctl --set-ctrl=...).

    exp_ms 不为 None -> 手动短曝光 (帧率上限≈1/曝光时间);
    auto_exp True   -> 强制自动曝光;
    都不给           -> 不干预 (用相机默认).
    """
    opts = []
    if exp_ms is not None:
        opts.append('--set-ctrl=exposure_auto=1')      # 1=手动
        opts.append('--set-ctrl=exposure_absolute=%d' % int(exp_ms * 10))  # 单位100us
    elif auto_exp:
        opts.append('--set-ctrl=exposure_auto=0')      # 0=自动
    return opts


def apply_exposure(dev, exp_opts):
    """单独执行曝光控制 (control 全局生效, 与采流分离避免污染帧率测量).

    控制项不存在时 (如本相机无 exposure_auto) 只警告, 不影响后续采流.
    """
    for opt in exp_opts:
        code, out = v4l2_ctl(dev, opt)
        if code != 0 or 'unknown' in out.lower():
            first = out.strip().splitlines()[0] if out.strip() else '未知错误'
            print('[WARN] 曝光控制失败: %s -> %s' % (opt, first))


# ================================================================
# 采图与拆帧
# ================================================================

def split_jpegs(data, out_dir):
    """把连续 MJPG 帧流按 JPEG 起止标记 (FFD8...FFD9) 拆成独立 jpg 文件.

    JPEG 熵编码对 0xFF 有转义 (FF 00), 起止标记不会在帧数据内误判.
    返回拆出的文件路径列表.
    """
    files, start, idx = [], -1, 0
    i, n = 0, len(data)
    while i < n - 1:
        if start < 0 and data[i] == 0xFF and data[i + 1] == 0xD8:
            start = i                              # 帧起始 (SOI)
        elif start >= 0 and data[i] == 0xFF and data[i + 1] == 0xD9:
            path = os.path.join(out_dir, 'frame_%02d.jpg' % idx)   # 帧结束 (EOI)
            with open(path, 'wb') as f:
                f.write(data[start:i + 2])
            files.append(path)
            idx += 1
            start = -1
        i += 1
    return files


def capture(dev, w, h, fmt, count, out_dir, parm, skip):
    """单次 fd 内 设格式+采图: 保存原始流, MJPG 拆成 jpg, 打印实测帧率."""
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, 'stream_%dx%d_%s.bin' % (w, h, fmt))
    if os.path.exists(raw):
        os.remove(raw)                      # 清掉上一次残留, 避免采流失败时读到旧数据

    fmt_opts = ['--set-fmt-video=width=%d,height=%d,pixelformat=%s' % (w, h, fmt)]
    if parm > 0:
        fmt_opts.append('--set-parm=%d' % parm)

    code, out, fps = stream_fps(dev, fmt_opts, count, skip, raw)
    print(out.strip())

    if not os.path.exists(raw) or os.path.getsize(raw) == 0:
        print('[WARN] 未采到数据 (请检查 格式/分辨率/设备权限)')
        return None

    if fmt == 'MJPG':
        with open(raw, 'rb') as f:
            data = f.read()
        files = split_jpegs(data, out_dir)
        if not files:
            print('[WARN] 未从流中拆出 JPEG 帧 (确认像素格式确实是 MJPG)')
        else:
            print('  已拆出 %d 帧 -> %s/' % (len(files), out_dir))
            print('  示例: %s (%d bytes)'
                  % (os.path.basename(files[0]), os.path.getsize(files[0])))
    else:
        print('  已保存原始流 %s (%d bytes, %s 非JPEG需自行转换)'
              % (raw, os.path.getsize(raw), fmt))

    print('实测帧率: %s' % ('%.1f fps' % fps if fps else '未知'))
    return fps


# ================================================================
# 逐分辨率实测持续帧率
# ================================================================

def benchmark(dev, fmt, count, parm, skip):
    """从 1080p 到 160x120 逐个分辨率实测持续帧率 (不落盘, 只测流)."""
    sizes = [(1920, 1080), (1280, 720), (960, 540), (640, 480), (320, 240), (160, 120)]
    print('===== 各分辨率实测持续帧率 (%s, 采%d帧/skip%d) =====' % (fmt, count, skip))
    print('  请求分辨率      实测帧率      协商分辨率')

    for w, h in sizes:
        fmt_opts = ['--set-fmt-video=width=%d,height=%d,pixelformat=%s' % (w, h, fmt)]
        if parm > 0:
            fmt_opts.append('--set-parm=%d' % parm)

        code, out, fps = stream_fps(dev, fmt_opts, count, skip)

        _, info = v4l2_ctl(dev, '--get-fmt-video')
        m = re.search(r'Width/Height\s*:\s*(\d+)/(\d+)', info)
        act = '%dx%d' % (int(m.group(1)), int(m.group(2))) if m else '-'

        if code != 0 or fps is None:
            print('  %-12s  不支持或出错    %s' % ('%dx%d' % (w, h), act))
        else:
            print('  %-12s  %6.1f fps       %s' % ('%dx%d' % (w, h), fps, act))


# ================================================================
# 主程序
# ================================================================

def main():
    ap = argparse.ArgumentParser(description='纯v4l2摄像头测试 (看格式/控制项/采图/测帧率)')
    ap.add_argument('--device', default='/dev/video0', help='摄像头设备')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--format', default='MJPG', help='像素格式 (默认MJPG)')
    ap.add_argument('--count', type=int, default=100, help='采图帧数(建议>=100测持续帧率)')
    ap.add_argument('--skip', type=int, default=10, help='跳过起始帧再测, 排除初始突发')
    ap.add_argument('--parm', type=int, default=400, help='请求帧率, 0=不设置')
    ap.add_argument('--exposure-ms', type=float, default=None,
                    help='手动曝光ms (仅当驱动有 exposure_auto 控制项时生效)')
    ap.add_argument('--auto-exposure', action='store_true', help='强制自动曝光')
    ap.add_argument('--out', default='captures', help='采图保存目录')
    ap.add_argument('--benchmark', action='store_true', help='逐分辨率实测持续帧率')
    a = ap.parse_args()

    dev = a.device if not a.device.isdigit() else '/dev/video%d' % int(a.device)
    print('================ v4l2 摄像头测试 (%s) ================' % dev)

    print('[1/2] 支持的格式 (--list-formats-ext)')
    _, out = v4l2_ctl(dev, '--list-formats-ext')
    print(out.strip() or '[无输出]')
    print('[1/2] 可用控制项 (--list-ctrls)  <- 找曝光/增益等真实控制名')
    _, out = v4l2_ctl(dev, '--list-ctrls')
    print(out.strip() or '[无输出]')

    # 曝光控制在独立命令里设置, 失败只警告, 不污染后面的采流测量
    exp_opts = exposure_opts(a.exposure_ms, a.auto_exposure)
    apply_exposure(dev, exp_opts)

    if a.benchmark:
        print('[2/2] 逐分辨率实测持续帧率')
        benchmark(dev, a.format, a.count, a.parm, a.skip)
    else:
        print('[2/2] 采图: %s %dx%d x%d 帧' % (a.format, a.width, a.height, a.count))
        capture(dev, a.width, a.height, a.format, a.count, a.out, a.parm, a.skip)


if __name__ == '__main__':
    main()
