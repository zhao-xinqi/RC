#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_show_fps.py - 采集/显示分离, 终端实时打印 fps (追求最高采集帧率)

目标: 采集尽可能高的帧率 (~200fps), 显示节流, 不拖累采集链路.

思路:
  1. 采集线程独占 cap.read(), 只保留最新一帧 + 累计帧计数 (旧帧自然丢弃);
  2. 主循环取最新帧, 每 DISPLAY_DIV 个"新帧"才显示一次
     (显示频率 = 采集频率 / DISPLAY_DIV, 避免显示器刷新率成为瓶颈);
  3. 终端每 LOG_INTERVAL 秒打印一次 采集 fps / 显示 fps (实时输出);
  4. (可选) 传 --rtsp 时另起独立线程, 把最新帧按 --rtsp_fps 节流编码推流,
     与采集/显示互不干扰 (编码/网络阻塞也不会拖累采集).

实测结论 (camera 0, MJPG):
  - 各分辨率 MJPG 采集均被管线卡在 ~204fps (瓶颈在 read() 链路, 不在分辨率),
    因此 640x480 是"最高分辨率 + 最高帧率"的最优组合;
  - YUYV 640x480 只有 60fps (USB2.0 带宽上限), 别用 YUYV;
  - 坑: 主循环绝对不能死循环忙等 (不 sleep), 会抢光 GIL 把采集线程饿到 ~35fps.
    必须 time.sleep(0.001) 让出 GIL, 采集才能稳定 ~204fps.

RTSP 推流说明:
  - OpenCV 只是推流"客户端", 需要一个 RTSP 服务端接收 (本地可用
    MediaMTX / rtsp-simple-server 起一个, 上位机用 VLC / ffplay 拉流查看);
  - 编码默认 H264 (省带宽), 若当前 opencv 编译版没有 libx264 会打不开,
    此时自动回退 MJPG 推流 (VLC 也能播);
  - 推流按 --rtsp_fps 节流, 默认 30fps, 无线链路不稳可降到 15 省带宽.

用法:
  python capture_show_fps.py                   # camera 0, MJPG 640x480, 采集最高帧率
  python capture_show_fps.py --w 1280 --h 720  # 换分辨率
  python capture_show_fps.py --div 2           # 显示节流改成每2帧显示一帧
  python capture_show_fps.py --headless 5      # 不开窗口, 只测5秒采集帧率
  # RTSP 推流 (先本地起 MediaMTX, 再跑下面的命令):
  python capture_show_fps.py --rtsp rtsp://127.0.0.1:8554/cam
  python capture_show_fps.py --rtsp rtsp://192.168.1.100:8554/cam --rtsp_fps 15 --rtsp_codec MJPG
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
RTSP_FPS = 30       # RTSP 推流帧率 (编码帧率, 不是采集帧率; 200fps 编码器吃不下)


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


class RtspPusher:
    """RTSP 推流线程: 从采集线程取最新帧, 按 rtsp_fps 节流编码推送.

    采集 200fps 编码器吃不下也浪费带宽, 所以推流按目标帧率节流;
    推流与采集/显示各占独立线程, VideoWriter.write() 阻塞也不拖累采集.
    """

    def __init__(self, url, w, h, fps, grabber, codec="H264"):
        self.url = url
        self.w, self.h = int(w), int(h)
        self.fps = fps
        self.grabber = grabber
        self.codec = codec
        self.stop_event = threading.Event()
        self.thread = None
        self.writer = self._open()

    def _open(self):
        """按 codec 打开 RTSP VideoWriter; H264 打不开时自动回退 MJPG."""
        writer = cv2.VideoWriter(self.url, cv2.CAP_FFMPEG,
                                 cv2.VideoWriter_fourcc(*self.codec),
                                 self.fps, (self.w, self.h))
        if writer.isOpened():
            print("== RTSP 推流 %s (%s %dx%d@%.0ffps) =="
                  % (self.url, self.codec, self.w, self.h, self.fps), flush=True)
            return writer
        if self.codec != "MJPG":
            # 部分 opencv-python 轮子没带 libx264, 自动回退 MJPG 保可用
            writer = cv2.VideoWriter(self.url, cv2.CAP_FFMPEG,
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     self.fps, (self.w, self.h))
            if writer.isOpened():
                print("[!] %s 不可用, 自动回退 MJPG 推流" % self.codec, flush=True)
                return writer
        print("[!] RTSP 推流打开失败 (请确认 RTSP 服务端已在监听): %s"
              % self.url, flush=True)
        return None

    def start(self):
        """启动推流线程 (daemon, 主程序退出自动结束)."""
        if self.writer is None:
            return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        """请求停止并等待推流线程收尾 (写帧缓冲已 flush)."""
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def run(self):
        """固定节律推"最新帧"; 编码/网络阻塞时不堆积, 自动跳过过期节拍."""
        interval = 1.0 / self.fps
        next_push = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.001)   # 让出 GIL, 别抢采集线程
                now = time.perf_counter()
                if now < next_push:
                    continue
                # 阻塞恢复后只补当前这一拍, 中间落下的节拍直接跳过 (丢旧帧)
                if now - next_push > interval:
                    next_push += int((now - next_push) / interval) * interval
                next_push += interval
                with self.grabber.lock:
                    f = self.grabber.frame
                if f is not None:
                    self.writer.write(f)
        finally:
            self.writer.release()
            print("== RTSP 推流已停止 ==", flush=True)


def start_rtsp(url, w, h, fps, grabber, codec="H264"):
    """url 为空返回 None; 否则创建推流线程并启动, 打不开返回 None."""
    if not url:
        return None
    pusher = RtspPusher(url, w, h, fps, grabber, codec)
    if pusher.writer is None:
        return None
    pusher.start()
    return pusher


def headless_run(cap, seconds, w, h, rtsp_url="", rtsp_fps=RTSP_FPS, rtsp_codec="H264"):
    """不开窗口: 跑 seconds 秒, 只打印采集 fps, 用于验证纯采集速度.

    若传 rtsp_url 则同步启动 RTSP 推流.
    """
    g = Grabber(cap)
    threading.Thread(target=g.run, daemon=True).start()
    pusher = start_rtsp(rtsp_url, w, h, rtsp_fps, g, rtsp_codec)

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

    if pusher:
        pusher.stop()


def gui_run(cap, div, w, h, autoclose=0, rtsp_url="", rtsp_fps=RTSP_FPS, rtsp_codec="H264"):
    """窗口模式: 显示节流 + 终端实时打印 采集/显示 fps, Esc 退出.

    若传 rtsp_url 则同步启动 RTSP 推流.
    autoclose>0 时到点自动退出 (供命令行/脚本自动化测试).
    """
    g = Grabber(cap)
    threading.Thread(target=g.run, daemon=True).start()
    pusher = start_rtsp(rtsp_url, w, h, rtsp_fps, g, rtsp_codec)

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

    if pusher:
        pusher.stop()


def main():
    ap = argparse.ArgumentParser(description="采集/显示分离 + 实时fps + 可选RTSP推流")
    ap.add_argument("--device", type=int, default=CAMERA_INDEX, help="相机索引, 默认0")
    ap.add_argument("--w", type=int, default=DEFAULT_W, help="分辨率宽, 默认640")
    ap.add_argument("--h", type=int, default=DEFAULT_H, help="分辨率高, 默认480")
    ap.add_argument("--div", type=int, default=DISPLAY_DIV, help="显示节流: 每N个新帧显示一帧")
    ap.add_argument("--headless", type=float, default=0,
                    help=">0 时不开窗口, 只测该秒数的纯采集帧率")
    ap.add_argument("--autoclose", type=float, default=0,
                    help=">0 时窗口模式到点自动退出 (供命令行测试)")
    ap.add_argument("--rtsp", default="",
                    help="RTSP推流地址, 如 rtsp://<上位机IP>:8554/cam; 留空则不推流")
    ap.add_argument("--rtsp_fps", type=float, default=RTSP_FPS,
                    help="RTSP推流帧率(编码帧率), 默认%d" % RTSP_FPS)
    ap.add_argument("--rtsp_codec", default="H264",
                    help="RTSP编码, 默认H264, 打不开自动回退MJPG")
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
        headless_run(cap, a.headless, nw, nh, a.rtsp, a.rtsp_fps, a.rtsp_codec)
    else:
        print("== Esc 退出; 每 %.0f 秒打印一次实时帧率 ==" % LOG_INTERVAL)
        gui_run(cap, a.div, nw, nh, a.autoclose, a.rtsp, a.rtsp_fps, a.rtsp_codec)
    cap.release()


if __name__ == "__main__":
    main()
