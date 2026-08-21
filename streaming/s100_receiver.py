"""
s100_receiver.py - RDK S100 接收处理

功能:
  在 RDK S100 上监听 TCP/UDP, 接收开发机发来的 JPEG 帧,
  解码后按配置运行各视觉任务 (P1 巡线 / P3 穿门 / P5 放球 / P2P4 球检测),
  周期性输出检测结果 (控制台), 供运动控制状态机使用; 有显示环境时可叠加显示.

线程模型:
  网络接收线程 : 持续 drain socket, 只保留最新完整帧 (防积压/降延迟)
  处理主循环  : 取最新帧 -> 解码 -> (可选)0.预处理 -> 各任务检测 -> 输出

运行示例 (在 S100 上, 与开发机 dev_capture_send.py 配合):
  python s100_receiver.py                              # TCP 监听 0.0.0.0:8900
  python s100_receiver.py --proto udp                  # 换 UDP
  python s100_receiver.py --tasks p1 p3                # 只启用巡线+穿门
  python s100_receiver.py --tasks p1 p4 p5             # 下视: 巡线+抓球+放球
  python s100_receiver.py --show                       # 有桌面时叠加显示
  python s100_receiver.py --save result                # 保存检测视频

对应开发机命令:
  python dev_capture_send.py --host <S100_IP> --proto tcp
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
from s100_tasks import TaskPipeline, draw_results

# ================================================================
# 默认配置
# ================================================================
PROTO = 'tcp'                 # 'tcp' / 'udp'
HOST = '0.0.0.0'              # 监听地址 (S100 全部网口)
PORT = 8900
TASKS = ['p1', 'p2', 'p3', 'p4', 'p5']   # 默认全部启用; 球检测缺环境自动跳过
SHOW = False                  # 是否叠加显示检测结果 (RDK 无桌面时自动降级)
SAVE_DIR = ''                 # 空=不保存; 否则保存检测视频到目录
STATS_INTERVAL = 1.0          # 控制台统计打印间隔 (秒)
PRINT_EVERY = 5               # 每 N 帧打印一次检测结果行


# ================================================================
# 主程序
# ================================================================

def main():
    # 行缓冲: 后台/重定向运行(如 S100 服务)时输出不积压
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description='RDK S100 接收开发机视频并执行视觉任务')
    parser.add_argument('--proto', default=PROTO, choices=['tcp', 'udp'])
    parser.add_argument('--host', default=HOST, help='监听地址')
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--tasks', nargs='+', default=TASKS,
                        choices=['p1', 'p2', 'p3', 'p4', 'p5'],
                        help='启用的任务 (默认全部; 无 hbm 环境自动跳过球检测)')
    parser.add_argument('--show', action='store_true', default=SHOW,
                        help='叠加显示检测结果')
    parser.add_argument('--save', default=SAVE_DIR,
                        help='保存检测视频到目录 (如 result)')
    parser.add_argument('--stats', type=float, default=STATS_INTERVAL,
                        help='统计打印间隔秒')
    parser.add_argument('--print-every', type=int, default=PRINT_EVERY,
                        help='每 N 帧打印一次检测结果')
    args = parser.parse_args()

    # ---- 1. 构建任务管线 ----
    cfg = {key: True for key in args.tasks}      # 任务开关
    pipe = TaskPipeline(cfg)
    if not pipe.task_keys:
        print('[S100] 没有任何任务可运行, 退出')
        return
    print(f'[S100] 启用任务: {", ".join(key.upper() for key in pipe.task_keys)}')

    # ---- 2. 启动接收端 ----
    rx = FrameReceiver(proto=args.proto, host=args.host, port=args.port).start()
    if args.proto == 'tcp':
        print(f'[S100] 监听 {args.host}:{args.port} (TCP), 等待开发机接入 ...')
    else:
        print(f'[S100] 绑定 {args.host}:{args.port} (UDP), 等待开发机数据 ...')

    # ---- 3. 显示 / 保存 ----
    win = 'S100 | Stream Vision'
    if args.show:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            print('[S100] 无法创建窗口 (RDK 无桌面时属正常), 关闭显示')
            args.show = False
    writer = None
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        path = os.path.join(args.save, 's100_detect.mp4')
        print(f'[S100] 检测视频将保存到: {path}')

    # ---- 4. 主循环: 取最新帧 -> 解码 -> 任务检测 -> 输出 ----
    proc_count = 0
    frame_count = 0
    last_stats = None                # 首个统计窗口从收到第一帧起计时
    last_recv, last_drop = 0, 0
    recv_fps = proc_fps = 0.0
    try:
        while True:
            # 阻塞等新帧 (条件变量): 只在有新帧时返回, 不重复处理同一帧,
            # 不受 Windows/Linux time.sleep 粒度影响; 无帧时每 50ms 醒来一次
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
            frame_count += 1

            # 运行各任务检测 (统计单帧耗时)
            t0 = time.perf_counter()
            results = pipe.run(img)
            proc_ms = (time.perf_counter() - t0) * 1000
            proc_count += 1

            # ---- 周期统计 (接收/处理 fps / 带宽 / 丢帧) ----
            # 统计窗口从第一帧/断流后重新对齐, 避免把等待接入的空闲时间算进帧率
            now = time.monotonic()
            if last_stats is None or now - last_stats > args.stats * 3:
                st = rx.stats()
                last_stats, last_recv, last_drop = now, st['recv'], st['drop']
            elif now - last_stats >= args.stats:
                st = rx.stats()
                recv_fps = (st['recv'] - last_recv) / (now - last_stats)
                proc_fps = proc_count / (now - last_stats)
                mbps = len(frame.payload) * 8 * recv_fps / 1_000_000
                print(f'[S100] 接收 {recv_fps:5.1f}fps | 处理 {proc_fps:5.1f}fps | '
                      f'带宽 {mbps:6.1f}Mbps | 丢帧 {st["drop"] - last_drop:4d}'
                      f' (累计 {st["drop"]}) | 单帧耗时 {proc_ms:.2f}ms')
                last_recv, last_drop = st['recv'], st['drop']
                proc_count = 0
                last_stats = now

            # ---- 周期性检测结果 (供运动控制状态机/控制台) ----
            if frame_count % args.print_every == 0:
                print(f'[S100] frame={frame.frame_id} {pipe.format_summary(results)}')

            # ---- 显示 / 保存 ----
            if args.show or args.save:
                vis = draw_results(img, results)
                if args.show:
                    cv2.putText(vis, f'recv={recv_fps:.1f} proc={proc_fps:.1f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                    cv2.imshow(win, vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if args.save:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            path, cv2.VideoWriter_fourcc(*'mp4v'),
                            200.0, (img.shape[1], img.shape[0]))
                    writer.write(vis)
    except KeyboardInterrupt:
        print('[S100] 收到 Ctrl+C, 退出')
    finally:
        rx.stop()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        st = rx.stats()
        print(f'[S100] 结束: 累计接收 {st["recv"]} 帧, 丢弃 {st["drop"]} 帧')


if __name__ == '__main__':
    main()
