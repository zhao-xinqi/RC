#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P1 巡线 - 摄像头只读帧率测试 (read_only.py)

作用:
  打开摄像头, 只循环读帧, 在画面上叠加实时帧率显示.
  不做任何检测/处理, 用于确认摄像头本身的读取上限 (硬上限).

用法:
  python read_only.py                # 默认摄像头 index=0, 640x480
  python read_only.py 1              # 指定摄像头索引
  python read_only.py 1 1280 720     # 指定索引和分辨率

退出:
  Ctrl+C 强制结束
"""

import sys
import time

import cv2

WIN_NAME = "Read Only FPS"   # 窗口名用 ASCII, 避免 Windows 黑屏


def main():
    # 可选参数: [摄像头索引] [宽] [高]
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 480

    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        print(f"[!] 摄像头 index={idx} 打开失败")
        return

    # 预热几帧, 等曝光/白平衡稳定
    for _ in range(5):
        cap.read()
    print(f"[*] 摄像头 index={idx} 就绪, Ctrl+C 退出")

    # 帧率 (指数滑动平均, 平滑显示)
    fps = 0.0
    prev_t = time.perf_counter()

    try:
        while True:
            # 帧率统计
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            ret, frame = cap.read()
            if not ret:
                print("\r[!] 读取帧失败...", end='', flush=True)
                continue

            # 叠加帧率显示
            cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WIN_NAME, frame)
            cv2.waitKey(1)   # 仅刷新窗口, 不做按键退出

            # 控制台单行刷新
            print(f"\r  read FPS: {fps:6.1f}   (单帧 {dt * 1000.0:6.2f} ms)   ",
                  end='', flush=True)

    except KeyboardInterrupt:
        print("\n[*] 已退出")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
