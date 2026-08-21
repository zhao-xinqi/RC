"""
dev_test_receiver.py - 开发机测试接收与处理

用途:
  在本机回环验证"采集发送 -> 接收处理"全链路, 无需 S100 真机:
    终端 1: python dev_test_receiver.py --proto tcp
    终端 2: python dev_capture_send.py --proto tcp --host 127.0.0.1

功能:
  - 作为接收端 (TCP 监听 / UDP 绑定), 持续接收 JPEG 帧
  - 解码 + 处理最新帧 (默认灰度统计; --run-line 时加载真实 P1 巡线检测器)
  - 统计: 接收 fps / 处理 fps / 单帧大小 / 带宽 / 丢帧数
  - --show 显示画面; --save 保存处理结果视频

运行示例:
  python dev_test_receiver.py --proto tcp
  python dev_test_receiver.py --proto udp
  python dev_test_receiver.py --run-line            # 加载真实巡线检测器做处理
  python dev_test_receiver.py --show                # 显示接收画面
  python dev_test_receiver.py --save result          # 保存处理视频到 result/
"""

# ================================================================
# 库导入
# ================================================================
import argparse
import os
import sys
import time

import cv2
import numpy as np

from frame_protocol import FrameReceiver

# ================================================================
# 默认配置
# ================================================================
PROTO = 'tcp'                 # 'tcp' / 'udp'
HOST = '0.0.0.0'              # 监听地址 (回环测试可换 127.0.0.1)
PORT = 8900
RUN_LINE = False              # 是否加载真实 P1 巡线检测器处理
SHOW = False                  # 是否显示接收画面
SAVE_DIR = ''                 # 空=不保存; 否则保存处理视频到该目录
STATS_INTERVAL = 1.0          # 统计打印间隔 (秒)


# ================================================================
# 处理函数
# ================================================================

def process_frame(frame, frame_id):
    """默认处理: 灰度统计 (验证解码与处理链路正常).

    可替换为任意视觉任务; 返回值 dict 会被绘制到画面左下角.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {'info': f'frame={frame_id} gray_mean={gray.mean():.1f}'}


def load_line_detector():
    """加载真实 P1 巡线检测器 (复用 s100_tasks.py 的任务封装)."""
    from s100_tasks import create_line_detector
    detector = create_line_detector({})
    print('[接收] 已加载真实 P1 巡线检测器 (验证真实处理链路)')
    return detector


def process_line(frame, detector, frame_id):
    """用 P1 巡线检测器处理一帧, 返回可绘制信息."""
    result = detector.detect(frame)
    if result.get('detected'):
        info = (f'frame={frame_id} line off={result["offset"]:+.1f}px '
                f'ang={result["angle"]:+.1f}deg')
        return {'result': result, 'info': info, 'detected': True}
    return {'result': result, 'info': f'frame={frame_id} line=--', 'detected': False}


# ================================================================
# 主程序
# ================================================================

def main():
    # 行缓冲: 后台/重定向运行时输出不积压
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description='开发机测试接收与处理 (回环验证协议)')
    parser.add_argument('--proto', default=PROTO, choices=['tcp', 'udp'])
    parser.add_argument('--host', default=HOST, help='监听地址')
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--run-line', action='store_true', default=RUN_LINE,
                        help='加载真实 P1 巡线检测器处理')
    parser.add_argument('--show', action='store_true', default=SHOW,
                        help='显示接收画面')
    parser.add_argument('--save', default=SAVE_DIR,
                        help='保存处理视频到目录 (如 result)')
    parser.add_argument('--stats', type=float, default=STATS_INTERVAL,
                        help='统计打印间隔秒')
    args = parser.parse_args()

    # ---- 1. 启动接收端 ----
    rx = FrameReceiver(proto=args.proto, host=args.host, port=args.port).start()
    if args.proto == 'tcp':
        print(f'[接收] 监听 {args.host}:{args.port} (TCP), 等待开发机接入 ...')
    else:
        print(f'[接收] 绑定 {args.host}:{args.port} (UDP), 等待开发机数据 ...')

    # ---- 2. 可选真实处理 ----
    detector = load_line_detector() if args.run_line else None

    # ---- 3. 可选保存视频 ----
    writer = None
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        path = os.path.join(args.save, 'recv_test.mp4')
        print(f'[接收] 处理视频将保存到: {path}')

    # ---- 4. 主循环: 取最新帧 -> 解码 -> 处理 ----
    win = 'DevTestReceiver'
    if args.show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            print('[警告] 无法创建窗口, 关闭显示')
            args.show = False

    proc_count = 0
    last_stats = None                # 首个统计窗口从收到第一帧起计时
    last_recv, last_drop = 0, 0
    recv_fps = proc_fps = 0.0
    try:
        while True:
            # 阻塞等新帧 (条件变量): 只在有新帧时返回, 不重复处理同一帧,
            # 不受 Windows time.sleep 粒度影响; 无帧时每 50ms 醒来一次检查退出
            frame = rx.wait(timeout=0.05)
            if frame is None:
                if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 解码 JPEG -> BGR
            img = cv2.imdecode(np.frombuffer(frame.payload, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 处理 (统计单帧处理耗时)
            t0 = time.perf_counter()
            if detector is not None:
                out = process_line(img, detector, frame.frame_id)
                proc_ms = (time.perf_counter() - t0) * 1000
                info = out['info']
            else:
                out = process_frame(img, frame.frame_id)
                proc_ms = (time.perf_counter() - t0) * 1000
                info = out['info']
            proc_count += 1

            # ---- 周期统计 ----
            # 统计窗口从第一帧/断流后重新对齐, 避免把等待连接的空闲时间算进帧率
            now = time.monotonic()
            if last_stats is None or now - last_stats > args.stats * 3:
                st = rx.stats()
                last_stats, last_recv, last_drop = now, st['recv'], st['drop']
            elif now - last_stats >= args.stats:
                st = rx.stats()
                recv_fps = (st['recv'] - last_recv) / (now - last_stats)
                proc_fps = proc_count / (now - last_stats)
                kb = len(frame.payload) / 1024.0
                mbps = len(frame.payload) * 8 * recv_fps / 1_000_000
                print(f'[统计] 接收 {recv_fps:5.1f}fps | 处理 {proc_fps:5.1f}fps | '
                      f'帧 {kb:6.1f}KB | 带宽 {mbps:6.1f} Mbps | '
                      f'接收端丢弃 {st["drop"] - last_drop:4d} (累计 {st["drop"]}) | '
                      f'处理耗时 {proc_ms:.2f}ms | {info}')
                last_recv, last_drop = st['recv'], st['drop']
                proc_count = 0
                last_stats = now

            # ---- 显示 ----
            if args.show:
                cv2.putText(img, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f'recv={recv_fps:.1f} proc={proc_fps:.1f}',
                            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(win, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ---- 保存 ----
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*'mp4v'),
                        200.0, (img.shape[1], img.shape[0]))
                writer.write(img)
    except KeyboardInterrupt:
        print('[接收] 收到 Ctrl+C, 退出')
    finally:
        rx.stop()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        st = rx.stats()
        print(f'[接收] 结束: 累计接收 {st["recv"]} 帧, 丢弃 {st["drop"]} 帧')


if __name__ == '__main__':
    main()
