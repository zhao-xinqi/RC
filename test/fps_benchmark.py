#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fps_benchmark.py - 本地(Windows) 实测 LRCP S400 (camera 1) 极限帧率

思路:
  1. 先用 MSMF 后端逐个设置 格式x分辨率, 读回相机真正协商出的分辨率/帧率,
     得到该相机实际支持的输出模式列表 (不盲猜);
  2. 对每个有效模式做持续采帧实测 (perf_counter 计时, 不信 cap.get(FPS) 的请求值),
     得到"端到端真实可达到帧率" (包含驱动+解码+应用读取整个链路);
  3. 输出按实测帧率排序的表格, 并对前几名做 3 秒长时复测验证可持续性。

用法:
  python fps_benchmark.py               # 发现 + 全模式实测
  python fps_benchmark.py --quick       # 只发现模式, 不做实测
  python fps_benchmark.py --only 640x400 MJPG   # 只测指定模式
"""

import argparse
import cv2
import sys
import time

CAMERA_INDEX = 0
WARMUP = 20          # 预热/排空缓冲帧数
FRAMES = 150         # 实测计时段内读取帧数
LONG_SECONDS = 3.0   # 前几名长时复测秒数

# 候选分辨率 (宽范围, 发现阶段会读回相机实际协商出的值)
CANDIDATE_RES = [
    (1920, 1080), (1280, 960), (1280, 720), (960, 540), (800, 600),
    (640, 480), (640, 400), (640, 360), (576, 432), (480, 480),
    (480, 360), (480, 270), (432, 432), (400, 400), (400, 300),
    (352, 288), (320, 240), (320, 180), (240, 240), (240, 160),
    (176, 144), (160, 120),
]
FORMATS = ["MJPG", "YUYV"]
FPS_TARGET = 1000   # 请求帧率: 设一个超高值让相机给足, 实测时按真实值算


def fourcc_str(v):
    """把 fourcc 整数解码成可读字符串, 非法值返回 '-'."""
    try:
        if not v:
            return "-"
        s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
        return s if s.isprintable() else "-"
    except Exception:
        return "-"


def open_mode(idx, fmt, w, h, fps_target=FPS_TARGET):
    """按 格式+分辨率 打开相机, 读回协商结果. 返回 cap 或 None."""
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    if fmt:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
    if w and h:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps_target > 0:
        cap.set(cv2.CAP_PROP_FPS, fps_target)
    return cap


def negotiated(cap):
    """读取相机实际协商出的 分辨率/帧率/格式."""
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    return (int(w), int(h), fps, fourcc)


def measure(cap, frames=FRAMES, warmup=WARMUP):
    """实测持续帧率: 预热排空缓冲后, 用 perf_counter 对 frames 帧计时."""
    for _ in range(warmup):
        cap.read()
    t0 = time.perf_counter()
    ok = 0
    for _ in range(frames):
        ret, _ = cap.read()
        if ret:
            ok += 1
    dt = time.perf_counter() - t0
    if dt <= 0 or ok == 0:
        return 0.0
    return ok / dt


def discover(idx):
    """发现阶段: 返回 {(fmt, w, h): 协商结果}, 只开流不采帧."""
    modes = {}
    print("== 发现支持的格式/分辨率 (MSMF, camera %d) ==" % idx)
    for fmt in FORMATS:
        for w, h in CANDIDATE_RES:
            cap = open_mode(idx, fmt, w, h)
            if cap is None:
                continue
            nw, nh, nfps, nfourcc = negotiated(cap)
            cap.release()
            # 相机协商出的格式四舍五入成最近的有效模式
            key = (fmt, nw, nh)
            if key not in modes:
                modes[key] = (nfps, nfourcc)
            print("  %-4s %5dx%-4d -> 协商 %dx%d @ %.0ffps (fourcc=%s)"
                  % (fmt, w, h, nw, nh, nfps, nfourcc))
    print("  => 共发现 %d 个有效输出模式" % len(modes))
    return modes


def sweep(idx, modes):
    """实测阶段: 对每个有效模式测真实持续帧率, 按降序输出."""
    print("\n== 实测持续帧率 (WARMUP=%d, FRAMES=%d) ==" % (WARMUP, FRAMES))
    results = []
    for (fmt, w, h), (req_fps, _fourcc) in sorted(modes.items()):
        cap = open_mode(idx, fmt, w, h)
        if cap is None:
            continue
        actual = measure(cap)
        nw, nh, nfps, nfourcc = negotiated(cap)
        cap.release()
        results.append((actual, fmt, nw, nh, nfps, nfourcc))
        print("  %-4s %5dx%-5d 请求帧率=%4.0f 实测=%6.1f fps  (fourcc=%s)"
              % (fmt, nw, nh, nfps, actual, nfourcc))

    results.sort(key=lambda r: r[0], reverse=True)
    print("\n== 按实测帧率排序 (Top 10) ==")
    print("  %-4s %9s %9s %9s" % ("格式", "分辨率", "请求fps", "实测fps"))
    for actual, fmt, nw, nh, nfps, _ in results[:10]:
        print("  %-4s %5dx%-4d %8.0f %9.1f" % (fmt, nw, nh, nfps, actual))
    return results


def long_verify(idx, results, topn=3):
    """对实测帧率最高的几个模式做 3 秒长时复测, 验证能否持续."""
    print("\n== 长时复测 (%.0f 秒, 验证可持续性) ==" % LONG_SECONDS)
    for actual, fmt, nw, nh, nfps, _ in results[:topn]:
        cap = open_mode(idx, fmt, nw, nh)
        if cap is None:
            continue
        for _ in range(WARMUP):
            cap.read()
        t0 = time.perf_counter()
        ok = 0
        while time.perf_counter() - t0 < LONG_SECONDS:
            ret, _ = cap.read()
            if ret:
                ok += 1
        dt = time.perf_counter() - t0
        cap.release()
        sustained = ok / dt if dt > 0 else 0.0
        print("  %-4s %5dx%-4d: 短测 %.1f fps -> 长时 %.1f fps (%.0fs内%d帧)"
              % (fmt, nw, nh, actual, sustained, dt, ok))


def main():
    ap = argparse.ArgumentParser(description="LRCP S400 极限帧率实测 (Windows/MSMF)")
    ap.add_argument("--device", type=int, default=CAMERA_INDEX, help="相机索引, 默认1")
    ap.add_argument("--quick", action="store_true", help="只发现模式, 不做实测")
    ap.add_argument("--only", nargs=2, metavar=("WxH", "FMT"),
                    help="只测指定模式, 如: --only 640x400 MJPG")
    a = ap.parse_args()

    idx = a.device
    if a.only:
        w, h = map(int, a.only[0].lower().split("x"))
        fmt = a.only[1].upper()
        cap = open_mode(idx, fmt, w, h)
        if cap is None:
            print("[!] 无法打开 %s %dx%d" % (fmt, w, h))
            sys.exit(1)
        nw, nh, nfps, nfourcc = negotiated(cap)
        actual = measure(cap)
        cap.release()
        print("请求 %s %dx%d -> 协商 %dx%d @%.0ffps, 实测 %.1f fps (fourcc=%s)"
              % (fmt, w, h, nw, nh, nfps, actual, nfourcc))
        return

    modes = discover(idx)
    if a.quick:
        return
    results = sweep(idx, modes)
    if results:
        long_verify(idx, results)


if __name__ == "__main__":
    main()
