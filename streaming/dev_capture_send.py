"""
dev_capture_send.py - 开发机采集与发送

功能:
  在开发机上以 ~200fps 高速采集 USB 摄像头 (MJPG), 将每帧编码为 JPEG,
  通过 TCP/UDP 实时传输到 RDK S100 板端 (s100_receiver.py) 或本机测试接收端
  (dev_test_receiver.py).

线程模型 (高帧率方案, 依据 test/README.md):
  采集线程 : 独占 cap.read(), 只保留最新一帧 (旧帧自然丢弃, 不积压)
  发送线程 : 取最新帧 -> JPEG 编码 -> 按帧协议发送 (网络慢时在发送端丢帧)
  主线程   : 打印统计 (采集/发送 fps / 带宽), 处理 Ctrl+C / --time 退出

关键点:
  - 相机参数: MSMF + MJPG + 640x480 + CAP_PROP_FPS=1000 (链路实测 ~204fps)
  - JPEG 质量可调, 640x480@q70 单帧约 20~60KB; 可 --scale 缩放再编码降带宽
  - 网络跟不上时主动丢最旧帧, 不无限积压 (延迟恒定)
  - 主循环必须 time.sleep(0.001) 让出 GIL, 否则死循环饿死采集线程 (掉到 ~35fps)

运行示例:
  python dev_capture_send.py                                # 相机0 -> S100(默认TCP)
  python dev_capture_send.py --proto udp                    # 换 UDP
  python dev_capture_send.py --host 127.0.0.1               # 本机回环测试
  python dev_capture_send.py --source test\\yolo_test.mp4   # 用视频文件代替相机 (测试用)
  python dev_capture_send.py --quality 60 --scale 0.5       # 压缩带宽 (无线链路)
  python dev_capture_send.py --time 10 --show               # 只跑10秒并预览发送画面
"""

# ================================================================
# 库导入
# ================================================================
import argparse
import sys
import threading
import time

import cv2

from frame_protocol import FrameSender

# ================================================================
# 默认配置
# ================================================================

# ---- 摄像头 (依据 test/README.md 实测结论) ----
SOURCE = 0                    # 0/1 为相机索引; 字符串为视频文件路径
BACKEND = 'MSMF'              # 'MSMF'(Windows默认) / 'ANY'(自动)
FORMAT = 'MJPG'               # 必须 MJPG 才能上 200fps (YUYV 顶死 USB2.0 只有 60fps)
WIDTH = 640                   # 实测"最高分辨率+最高帧率"的最优组合
HEIGHT = 480
FPS_TARGET = 1000             # 请求超高帧率, 逼相机协商出上限 (实测链路 ~204)

SCALE = 1.0                   # 编码前缩放因子 (0.5 -> 320x240, 带宽降 ~4x)
JPEG_QUALITY = 70             # JPEG 质量 0~100, 越低帧越小带宽越低
MAX_FPS = 0                   # 发送限速 (0=不限速, 按链路实测上限)

# ---- 网络 ----
PROTO = 'tcp'                 # 'tcp' 可靠 / 'udp' 低延迟可丢帧
HOST = '192.168.43.38'        # S100 的 IP (按实际网络修改)
PORT = 8900

# ---- 统计 / 退出 ----
STATS_INTERVAL = 1.0          # 控制台统计打印间隔 (秒)
TIME_LIMIT = 0                # 0=持续运行; >0 秒后自动退出 (测试用)
SHOW = False                  # 是否预览发送画面 (主线程节流显示)
LOOP_VIDEO = False            # 视频文件模式: 播完循环 (测试持续速率用)


# ================================================================
# 工具类
# ================================================================

class _LatestFrameStore:
    """线程安全"最新帧"容器 (条件变量版), 带代数计数.

    采用"最新帧覆盖"策略: 采集线程 put() 后通知, 发送线程阻塞等待新帧,
    不再用 time.sleep 轮询 (Windows 上 time.sleep 粒度约 15.6ms,
    轮询会把发送吞吐压到 ~60fps). 来不及发送的旧帧被覆盖丢弃, 不积压.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._gen = 0           # 代数: 每次写入 +1

    def put(self, frame):
        with self._cond:
            self._frame = frame
            self._gen += 1
            self._cond.notify_all()

    def wait_new(self, last_gen, timeout=0.2):
        """阻塞等待新帧; 返回 (最新代数, 最新帧); 超时返回 (last_gen, None).

        timeout 用于周期性检查 stop_evt (等待期间不占 CPU).
        """
        with self._cond:
            while self._gen == last_gen:
                if not self._cond.wait(timeout):
                    return last_gen, None
            return self._gen, self._frame

    def peek_new(self, last_gen):
        """非阻塞查看是否有新帧 (预览用); 返回 (最新代数, 最新帧或 None)."""
        with self._cond:
            if self._gen == last_gen:
                return last_gen, None
            return self._gen, self._frame


class _Stats:
    """跨线程统计 (采集/发送计数, 发送字节数, 丢帧数)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cap_count = 0      # 采集帧数
        self.send_count = 0     # 实际发送帧数
        self.send_bytes = 0     # 实际发送字节数
        self.drop_count = 0     # 发送端丢帧数 (网络跟不上/编码跟不上)


# ================================================================
# 采集线程
# ================================================================

def _capture_worker(cap, store, stats, stop_evt, loop=False):
    """采集线程: 独占 cap.read(), 只保留最新一帧.

    loop: 视频文件模式读到末尾后从头循环 (用于持续速率测试/演示回放);
          相机模式该参数无效.
    """
    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 循环: 回到视频开头
                continue
            time.sleep(0.01)    # 读帧失败 (可能暂时无帧), 稍等重试
            continue
        store.put(frame)
        with stats.lock:
            stats.cap_count += 1


# ================================================================
# 发送线程
# ================================================================

def _encode(frame, scale, quality):
    """缩放 + JPEG 编码, 返回 (payload 字节, 编码后尺寸 (w, h))."""
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', frame,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None, None
    h, w = frame.shape[:2]
    return buf.tobytes(), (w, h)


def _sleep_precise(seconds):
    """高精度短睡眠 (Windows time.sleep 粒度约 15.6ms, 短延时用忙等补齐).

    仅用于限速/节流等非关键路径; 主要线程的等待一律用条件变量而非 sleep.
    """
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    if seconds >= 0.01:
        time.sleep(seconds * 0.9)   # 先正常睡大头
    while time.perf_counter() < end:   # 忙等补齐剩余 (微秒级)
        pass


def _send_worker(sender, store, stats, stop_evt, scale, quality, max_fps, proto):
    """发送线程: 取最新帧 -> 编码 -> 发送; 网络/编码跟不上时丢帧.

    帧序号沿用采集代数 (gen), 接收端据此做丢帧统计.
    """
    last_gen = 0
    while not stop_evt.is_set():
        # 阻塞等待新帧 (条件变量, 不占 CPU, 不受 time.sleep 粒度影响)
        gen, frame = store.wait_new(last_gen, timeout=0.2)
        if frame is None:
            continue

        # 统计被覆盖丢弃的帧数 (gen 跳过的帧)
        with stats.lock:
            stats.drop_count += max(gen - last_gen - 1, 0)
        last_gen = gen

        # 可选发送限速 (用高精度睡眠, 避免 Windows sleep 粒度误差)
        if max_fps > 0:
            _sleep_precise(1.0 / max_fps)

        payload, (w, h) = _encode(frame, scale, quality)
        if payload is None:
            continue

        ok = sender.send(gen, w, h, payload)
        if ok:
            with stats.lock:
                stats.send_count += 1
                stats.send_bytes += len(payload)
        else:
            # TCP 断连: 尝试重连后继续 (重连期间丢帧由 send 返回 False 体现)
            if proto == 'tcp' and not stop_evt.is_set():
                if sender.connect(timeout=3.0):
                    print('[发送] 已重新连接到接收端')


# ================================================================
# 摄像头 / 视频源
# ================================================================

def open_source(source, backend, width, height, fps_target, fmt):
    """打开摄像头 (MJPG 高帧率) 或视频文件 (测试用)."""
    if isinstance(source, str):
        # 视频文件: 直接按文件播放 (用于无相机时测试协议/回放素材)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {source}")
        return cap

    api = cv2.CAP_MSMF if backend == 'MSMF' else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, api)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头: {source} (请确认相机已接入, 或改用 --backend ANY)")
    return cap


# ================================================================
# 主程序
# ================================================================

def main():
    # 行缓冲: 后台/重定向运行(如 S100 服务)时输出不积压
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description='开发机摄像头采集 + TCP/UDP 发送')
    parser.add_argument('--source', default=SOURCE, type=_parse_source,
                        help='摄像头索引(0/1) 或 视频文件路径; 默认 0')
    parser.add_argument('--backend', default=BACKEND, choices=['MSMF', 'ANY'],
                        help='Windows 采集后端; 默认 MSMF')
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--scale', type=float, default=SCALE,
                        help='编码前缩放因子 (0.5=减半, 带宽降约4x)')
    parser.add_argument('--quality', type=int, default=JPEG_QUALITY,
                        help='JPEG 质量 0~100; 默认 70')
    parser.add_argument('--max-fps', type=int, default=MAX_FPS,
                        help='发送限速 (0=不限)')
    parser.add_argument('--proto', default=PROTO, choices=['tcp', 'udp'])
    parser.add_argument('--host', default=HOST, help='接收端 IP (S100)')
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--time', type=int, default=TIME_LIMIT,
                        help='运行秒数, 0=持续')
    parser.add_argument('--loop', action='store_true', default=LOOP_VIDEO,
                        help='视频文件模式: 播完自动循环 (测试持续速率用)')
    parser.add_argument('--show', action='store_true', default=SHOW,
                        help='预览发送画面')
    parser.add_argument('--stats', type=float, default=STATS_INTERVAL,
                        help='统计打印间隔秒')
    args = parser.parse_args()

    # ---- 1. 打开采集源 ----
    print(f'[采集] 打开 {"摄像头 " + str(args.source) if isinstance(args.source, int) else "视频 " + args.source} ...')
    cap = open_source(args.source, args.backend,
                      args.width, args.height, FPS_TARGET, FORMAT)
    print(f'[采集] 请求 {args.width}x{args.height} {FORMAT} (FPS_TARGET={FPS_TARGET}); '
          f'实际帧率以统计为准')

    # ---- 2. 建立网络连接 ----
    sender = FrameSender(proto=args.proto, host=args.host, port=args.port)
    if args.proto == 'tcp':
        t0 = time.monotonic()
        connected = sender.connect(timeout=5.0)
        while not connected and not (args.time and time.monotonic() - t0 > args.time):
            print(f'[发送] 无法连接 {args.host}:{args.port}, 2 秒后重试 ...')
            time.sleep(2)
            connected = sender.connect(timeout=5.0)
        if not connected:
            print('[发送] 未连接到接收端, 退出')
            cap.release()
            return
        print(f'[发送] 已连接 {args.host}:{args.port} (TCP)')
    else:
        # UDP: 无需握手, 但必须调用 connect 创建发送 socket (否则 send 直接失败)
        sender.connect()
        print(f'[发送] 目标 {args.host}:{args.port} (UDP, 无需建立连接)')

    # ---- 3. 启动采集/发送线程 ----
    store = _LatestFrameStore()
    stats = _Stats()
    stop_evt = threading.Event()

    cap_thread = threading.Thread(target=_capture_worker,
                                  args=(cap, store, stats, stop_evt, args.loop),
                                  name='capture', daemon=True)
    send_thread = threading.Thread(target=_send_worker,
                                   args=(sender, store, stats, stop_evt,
                                         args.scale, args.quality, args.max_fps,
                                         args.proto),
                                   name='sender', daemon=True)
    cap_thread.start()
    send_thread.start()

    # ---- 4. 主循环: 统计 + 可选预览 + 退出控制 ----
    win = 'DevCam | Sending'
    if args.show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            print('[警告] 无法创建预览窗口 (无桌面环境), 关闭预览')
            args.show = False

    t_start = time.monotonic()
    last_stats = t_start
    last_cap, last_send, last_bytes, last_drop = 0, 0, 0, 0
    show_gen = 0
    try:
        while not stop_evt.is_set():
            # 主循环只做统计/预览/退出检查, 帧搬运由采集/发送线程用条件变量完成,
            # 这里 sleep 粒度不影响吞吐
            time.sleep(0.02)

            # ---- 预览 (节流: 只显示最新一帧) ----
            if args.show:
                gen, frame = store.peek_new(show_gen)
                if frame is not None:
                    show_gen = gen
                    cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ---- 周期统计 ----
            now = time.monotonic()
            if now - last_stats >= args.stats:
                with stats.lock:
                    cap_c, send_c, send_b, drop_c = (stats.cap_count, stats.send_count,
                                                     stats.send_bytes, stats.drop_count)
                dt = now - last_stats
                cap_fps = (cap_c - last_cap) / dt
                send_fps = (send_c - last_send) / dt
                mbps = (send_b - last_bytes) / dt / 1_000_000 * 8
                print(f'[统计] 采集 {cap_fps:5.1f}fps | 发送 {send_fps:5.1f}fps | '
                      f'带宽 {mbps:6.1f} Mbps | 发送端丢帧 {drop_c - last_drop:4d}'
                      f' (累计 {drop_c}) | 累计发送 {send_c} 帧')
                last_cap, last_send, last_bytes, last_drop = cap_c, send_c, send_b, drop_c
                last_stats = now

            # ---- 定时退出 ----
            if args.time and now - t_start > args.time:
                print(f'[发送] 到达 --time {args.time}s, 退出')
                break
    except KeyboardInterrupt:
        print('[发送] 收到 Ctrl+C, 退出')
    finally:
        # ---- 5. 收尾 ----
        stop_evt.set()
        cap_thread.join(timeout=2)
        send_thread.join(timeout=2)
        cap.release()
        sender.close()
        if args.show:
            cv2.destroyAllWindows()
        with stats.lock:
            print(f'[发送] 结束: 累计采集 {stats.cap_count} 帧, '
                  f'发送 {stats.send_count} 帧, 发送端丢帧 {stats.drop_count} 帧')


def _parse_source(value: str):
    """把命令行 --source 解析为 int (相机索引) 或 str (文件路径)."""
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == '__main__':
    main()
