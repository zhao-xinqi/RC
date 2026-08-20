#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_show_fps.py - 采集/显示分离, 终端实时打印 fps (追求最高采集帧率)

目标: 采集尽可能高的帧率 (~200fps), 显示节流, 不拖累采集链路.

思路:
  1. 采集线程独占 cap.read(), 只保留最新一帧 + 累计帧计数 (旧帧自然丢弃);
  2. 主循环取最新帧, 每 DISPLAY_DIV 个"新帧"才显示一次
     (显示频率 = 采集频率 / DISPLAY_DIV, 避免显示器刷新率成为瓶颈);
  3. 终端每 LOG_INTERVAL 秒打印一次 采集 fps / 显示 fps (实时输出).

实测结论 (camera 0, MJPG):
  - 各分辨率 MJPG 采集均被管线卡在 ~204fps (瓶颈在 read() 链路, 不在分辨率),
    因此 640x480 是"最高分辨率 + 最高帧率"的最优组合;
  - YUYV 640x480 只有 60fps (USB2.0 带宽上限), 别用 YUYV;
  - 坑: 主循环绝对不能死循环忙等 (不 sleep), 会抢光 GIL 把采集线程饿到 ~35fps.
    必须 time.sleep(0.001) 让出 GIL, 采集才能稳定 ~204fps.

用法:
  python capture_show_fps.py                   # camera 0, MJPG 640x480, 采集最高帧率
  python capture_show_fps.py --w 1280 --h 720  # 换分辨率
  python capture_show_fps.py --div 2           # 显示节流改成每2帧显示一帧
  python capture_show_fps.py --headless 5      # 不开窗口, 只测5秒采集帧率
  # 窗口按 Esc 退出
"""

import argparse
import threading
import time
import cv2

CAMERA_INDEX = 0    # 当前外接摄像头是 camera 0
FORMAT = "MJPG"     # 必须 MJPG 才能上 200fps
DEFAULT_W = 640     # 实测中 最高分辨率+最高帧率 的档位
DEFAULT_H = 480
FPS_TARGET = 1000   # 请求超高帧率, 逼相机协商出上限 (协商出 400)
DISPLAY_DIV = 4     # 显示节流: 每 4 个新帧显示一帧 (~50fps 显示)
LOG_INTERVAL = 1.0  # 终端 fps 打印间隔(秒)


def fourcc_str(v):
    """把 fourcc 整数解码成可读字符串, 非法值返回 '-'."""
    try:
        if not v:
            return "-"
        s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
        return s if s.isprintable() else "-"
    except Exception:
        return "-"


def open_camera(idx, w, h):
    """按 MSMF + MJPG 打开相机并设好参数, 返回 cap 或 None."""
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FORMAT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    return cap


class Grabber:
    """采集线程: 独占 read(), 只保留最新一帧, 并累计成功读到的帧数.

    采集在独立线程里跑满, 不因显示/处理变慢而掉帧;
    count 单调递增, 主循环用它算真实采集帧率.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None   # 最新帧 (可能仍是旧的, 直到下一次 read 更新)
        self.count = 0      # 累计成功读取帧数

    def run(self):
        while True:
            ok, f = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.frame = f
                self.count += 1


def headless_run(cap, seconds):
    """不开窗口: 跑 seconds 秒, 只打印采集 fps, 用于验证纯采集速度."""
    g = Grabber(cap)
    threading.Thread(target=g.run, daemon=True).start()

    last_cap = 0
    t0 = time.perf_counter()
    t_last = t0
    while time.perf_counter() - t0 < seconds:
        time.sleep(LOG_INTERVAL)
        with g.lock:
            cur = g.count
        now = time.perf_counter()
        fps = (cur - last_cap) / (now - t_last)
        last_cap = cur
        t_last = now
        print("采集 %.1f fps" % fps, flush=True)


def gui_run(cap, div, autoclose=0):
    """窗口模式: 显示节流 + 终端实时打印 采集/显示 fps, Esc 退出.

    autoclose>0 时到点自动退出 (供命令行/脚本自动化测试).
    """
    g = Grabber(cap)
    threading.Thread(target=g.run, daemon=True).start()

    last_cap = 0      # 上次采样时的采集计数
    last_show = 0     # 上次采样时的显示计数
    last_shown_cap = 0  # 上次已显示帧对应的采集计数
    show_count = 0
    t_last = time.perf_counter()
    t_start = t_last

    while True:
        time.sleep(0.001)   # 关键: 让出 GIL, 否则死循环会饿死采集线程 (实测会掉到~35fps)
        with g.lock:
            f = g.frame
            cur_cap = g.count
        if f is None:
            continue
        # 节流: 每 div 个新帧才显示一次 (显示帧率 = 采集帧率/div)
        if cur_cap - last_shown_cap >= div:
            cv2.imshow("cam", f)
            last_shown_cap = cur_cap
            show_count += 1
            if cv2.waitKey(1) & 0xFF == 27:   # Esc 退出
                break

        # 到点自动退出 (autoclose>0 时)
        if autoclose > 0 and time.perf_counter() - t_start >= autoclose:
            print("== 自动退出 (%.0f 秒到点) ==" % autoclose, flush=True)
            break

        # 定时打印实时帧率
        now = time.perf_counter()
        if now - t_last >= LOG_INTERVAL:
            cap_fps = (cur_cap - last_cap) / (now - t_last)
            show_fps = (show_count - last_show) / (now - t_last)
            print("采集 %6.1f fps | 显示 %6.1f fps | 缓冲帧数 %d"
                  % (cap_fps, show_fps, cur_cap - last_shown_cap), flush=True)
            last_cap = cur_cap
            last_show = show_count
            t_last = now


def main():
    ap = argparse.ArgumentParser(description="采集/显示分离 + 实时fps (追求最高采集帧率)")
    ap.add_argument("--device", type=int, default=CAMERA_INDEX, help="相机索引, 默认0")
    ap.add_argument("--w", type=int, default=DEFAULT_W, help="分辨率宽, 默认640")
    ap.add_argument("--h", type=int, default=DEFAULT_H, help="分辨率高, 默认480")
    ap.add_argument("--div", type=int, default=DISPLAY_DIV, help="显示节流: 每N个新帧显示一帧")
    ap.add_argument("--headless", type=float, default=0,
                    help=">0 时不开窗口, 只测该秒数的纯采集帧率")
    ap.add_argument("--autoclose", type=float, default=0,
                    help=">0 时窗口模式到点自动退出 (供命令行测试)")
    a = ap.parse_args()

    cap = open_camera(a.device, a.w, a.h)
    if cap is None:
        print("[!] 无法打开 camera %d (%s %dx%d)" % (a.device, FORMAT, a.w, a.h))
        return

    # 读回真实协商值 (fourcc 读回在 MSMF 下常不可靠, 仅参考)
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfps = cap.get(cv2.CAP_PROP_FPS)
    nfourcc = fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    print("== camera %d 请求 %s %dx%d -> 协商 %dx%d @%.0ffps (fourcc=%s) =="
          % (a.device, FORMAT, a.w, a.h, nw, nh, nfps, nfourcc))

    if a.headless > 0:
        headless_run(cap, a.headless)
    else:
        print("== Esc 退出; 每 %.0f 秒打印一次实时帧率 ==" % LOG_INTERVAL)
        gui_run(cap, a.div, a.autoclose)
    cap.release()


if __name__ == "__main__":
    main()
