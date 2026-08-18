#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P1 巡线 - 主程序

功能:
  从下视摄像头实时读取画面, 检测池底橙红色引导线,
  输出 偏移量 offset 与 角度 angle 供机器人控制器使用.

  实时反馈:
    - 画面上叠加当前帧率 (FPS)
    - 控制台单行刷新 offset / angle / FPS / 单帧检测耗时
    - 每 N 帧在控制台打印一次各阶段耗时 (perf), 便于定位瓶颈

检测流水线 (由 line_detector 完成):
  原始帧 → 预处理 → HSV颜色分割 → 形态学 → 轮廓提取 → 偏移/角度

用法:
  python time_test.py                        # 默认下视摄像头 index=1
  python time_test.py --camera 0             # 指定摄像头索引
  python time_test.py --resolution 1280 720  # 指定采集分辨率
  python time_test.py --perf-every 120       # 每 120 帧打印一次阶段耗时

采集策略:
  cap.read() 由后台线程执行, 主循环从缓冲取最新帧, 采集与检测解耦,
  主循环不再阻塞等待相机帧周期.

退出:
  按 Ctrl+C 强制结束 (本程序不提供按键退出机制)
"""

import argparse
import importlib.util
import os
import sys
import threading
import time

import cv2

# 显示窗口名 (必须为纯 ASCII, Windows 下中文窗口名易黑屏)
WIN_NAME = "P1 Line Following"


# ================================================================
# 模块加载 (文件夹名含数字/中文, 无法用标准 import)
# ================================================================
def _load_modules():
    """
    动态加载 P1 巡线 相关模块

    Returns:
        (LineDetector类, get_config函数, line_detector模块对象)
        line_detector模块对象用于计时 preprocess 阶段.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载 config
    cfg_path = os.path.join(module_dir, "config.py")
    spec = importlib.util.spec_from_file_location("p1_main_config", cfg_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)

    # 加载 line_detector (其内部会自行加载 0.预处理)
    p1_path = os.path.join(module_dir, "line_detector.py")
    spec = importlib.util.spec_from_file_location("p1_main_line_detector", p1_path)
    ld_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld_mod)

    return ld_mod.LineDetector, cfg_mod.get_config, ld_mod


# ================================================================
# 摄像头打开 (带预热与后端回退)
# ================================================================
def _open_camera(index, width, height):
    """
    打开摄像头并预热

    依次尝试 默认后端 → DSHOW (Windows USB 摄像头 DSHOW 更稳定),
    预热期间丢弃前几帧, 等待相机自动曝光/白平衡稳定, 并排除全黑帧.

    Args:
        index: 摄像头索引
        width: 目标采集宽度
        height: 目标采集高度

    Returns:
        cv2.VideoCapture 或 None (打开失败)
    """
    backends = [cv2.CAP_ANY]
    if sys.platform.startswith('win'):
        backends.append(cv2.CAP_DSHOW)

    for pos, backend in enumerate(backends):
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 采集像素格式设为 MPEG (部分摄像头/采集卡默认 MJPEG 或 YUYV,
        # 强制 MPEG 可降低带宽占用; 若不支持会静默保持原格式)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MPEG'))

        if cap.isOpened() and _warmup(cap):
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"[*] 摄像头 index={index} 已就绪: {actual_w:.0f}x{actual_h:.0f}")
            return cap

        # 当前后端失败, 释放后尝试下一个
        cap.release()
        if pos < len(backends) - 1:
            print("[!] 默认后端打开失败, 改用 DSHOW 后端重试...")

    return None


def _warmup(cap, n_frames=5):
    """
    摄像头预热: 读取并丢弃前几帧, 等待曝光/白平衡稳定.

    Args:
        cap:      已打开的 VideoCapture
        n_frames: 最多尝试的预热帧数

    Returns:
        bool: True=读到有效帧, False=预热后仍为全黑帧
    """
    for _ in range(n_frames):
        ret, frame = cap.read()
        # 亮度均值 > 1.0 视为有效画面 (全黑帧均值约为 0)
        if ret and frame is not None and float(frame.mean()) > 1.0:
            return True
    return False


# ================================================================
# 后台线程摄像头读取器 (采集与处理解耦)
# ================================================================
class CameraReader:
    """
    后台线程摄像头读取器: 采集与检测解耦.

    cap.read() 在独立线程持续抓取最新帧存入单槽缓冲,
    主循环调用 read() 时立即返回最新帧, 不再阻塞等待相机帧周期.

    线程安全: 通过 Condition 保护缓冲槽.
    - 处理比相机慢 → 自动丢弃旧帧, 始终拿到最新帧 (低延迟)
    - 处理比相机快 → read() 等待下一帧, 按相机节奏运行

    注意: 取出的帧为后台线程共享对象, 下游检测/绘制不得原地修改.
    """

    def __init__(self, cap):
        """
        Args:
            cap: 已打开并预热的 cv2.VideoCapture
        """
        self._cap = cap
        self._cond = threading.Condition()
        self._frame = None      # 单槽缓冲: 仅保留最新一帧
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """后台循环: 持续抓帧, 覆盖写入单槽缓冲"""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # 读取失败: 短暂休眠, 避免空转占用 CPU
                time.sleep(0.001)
                continue
            with self._cond:
                self._frame = frame
                self._cond.notify_all()

    def read(self, timeout=1.0):
        """
        获取最新一帧

        Args:
            timeout: 等待新帧的超时 (秒); 超过即返回失败

        Returns:
            (ret, frame): ret=True 取到最新帧; 超时返回 (False, None)
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame is not None, timeout):
                return False, None
            frame = self._frame
            self._frame = None
            return True, frame

    def release(self):
        """停止后台线程并释放摄像头"""
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


# ================================================================
# 阶段计时器 (各阶段耗时统计)
# ================================================================
class StageTimer:
    """
    检测阶段计时器: 包裹各阶段方法, 累积每次调用的耗时 (ms).
    用于周期性在控制台打印各阶段耗时, 定位帧率瓶颈.
    """

    def __init__(self):
        self.times = {}    # label -> list[ms]

    def bind(self, obj, method_name, label):
        """
        把 obj.method_name 替换为计时版, 耗时累积到 self.times[label]

        Args:
            obj:         拥有该方法的对象 (或模块)
            method_name: 方法名
            label:       环节显示名 (如 "预处理")
        """
        original = getattr(obj, method_name)
        timer = self

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000.0
            timer.times.setdefault(label, []).append(dt)
            return result

        setattr(obj, method_name, wrapped)

    def reset(self):
        """清空累积的耗时数据 (每个打印周期结束后调用)"""
        self.times.clear()

    def summary(self):
        """
        按平均耗时降序返回各环节平均耗时

        Returns:
            [(label, 平均ms), ...]
        """
        rows = [(label, sum(ms) / len(ms))
                for label, ms in self.times.items() if ms]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows


def _avg(ms):
    """列表平均 (ms)"""
    return sum(ms) / len(ms) if ms else 0.0


def _print_stage_summary(timer, detect_ms_list, read_ms_list,
                         draw_ms_list, show_ms_list, fps):
    """
    控制台打印各阶段耗时汇总表

    Args:
        timer:          StageTimer (含一个统计周期的累积数据)
        detect_ms_list: 该周期内每帧 detect() 总耗时列表 (ms)
        read_ms_list:   该周期内每帧 reader.read() 耗时列表 (ms, 后台取帧)
        draw_ms_list:   该周期内每帧 draw_result() 耗时列表 (ms)
        show_ms_list:   该周期内每帧 imshow+waitKey 耗时列表 (ms)
        fps:            当前实时帧率 (EMA)
    """
    n = len(detect_ms_list)
    detect_avg = _avg(detect_ms_list)
    read_avg = _avg(read_ms_list)
    draw_avg = _avg(draw_ms_list)
    show_avg = _avg(show_ms_list)
    loop_avg = read_avg + detect_avg + draw_avg + show_avg

    # ---- 各检测阶段耗时 ----
    print(f"\n--- 各阶段耗时 (近 {n} 帧) ---")
    print(f"{'阶段':<10}{'平均ms':>9}{'占检测%':>10}")
    print("-" * 30)
    for label, avg in timer.summary():
        pct = avg / detect_avg * 100.0 if detect_avg > 0 else 0.0
        print(f"{label:<10}{avg:>9.3f}{pct:>9.1f}%")
    print("-" * 30)

    # ---- 每帧总耗时分解 (解释理论/实际 FPS 差距) ----
    print(f"--- 每帧耗时分解 (近 {n} 帧) ---")
    if read_avg > 0:
        print(f"  采集 read    {read_avg:8.2f} ms  → 后台线程取帧, 不阻塞主循环")
    if detect_avg > 0:
        print(f"  检测 detect  {detect_avg:8.2f} ms  → 仅检测    {1000.0 / detect_avg:6.1f} FPS")
    print(f"  绘制 draw    {draw_avg:8.2f} ms")
    print(f"  显示 show    {show_avg:8.2f} ms")
    print(f"  完整循环     {loop_avg:8.2f} ms  → 理论上限 {1000.0 / loop_avg:6.1f} FPS")
    print(f"  实际(EMA)    {fps:8.1f} FPS   [被 最慢环节 / 相机上限 / 显示刷新 卡住]")
    print()   # 空一行, 避免被下一行 \r 状态行覆盖


# ================================================================
# 主程序
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="P1 巡线 - 引导线检测主程序")
    parser.add_argument('--camera', type=int, default=1, metavar='N',
                        help='下视摄像头索引 (默认: 1)')
    parser.add_argument('--resolution', type=int, nargs=2, default=None,
                        metavar=('W', 'H'),
                        help='采集分辨率 (默认取 config.py 中的处理分辨率)')
    parser.add_argument('--perf-every', type=int, default=60, metavar='N',
                        help='每 N 帧打印一次各阶段耗时 (默认: 60)')
    args = parser.parse_args()

    # ---- 加载检测器 ----
    LineDetector, get_config, ld_mod = _load_modules()
    cfg = get_config()
    detector = LineDetector(cfg)

    # 采集分辨率: 默认使用 config 中的处理分辨率
    cam_w = int(cfg.get('image_width', 640))
    cam_h = int(cfg.get('image_height', 480))
    if args.resolution:
        cam_w, cam_h = args.resolution

    # ---- 阶段计时: 包裹各阶段, 自动累计耗时 ----
    timer = StageTimer()
    timer.bind(ld_mod, 'preprocess', '预处理')
    timer.bind(detector, '_color_segment', '颜色分割')
    timer.bind(detector, '_morphology_process', '形态学')
    timer.bind(detector, '_extract_line', '轮廓+角度')
    timer.bind(detector, '_debounce_angle', '角度消抖')

    # ---- 打开摄像头 ----
    print("=" * 55)
    print("  P1 巡线 - 引导线检测")
    print("  按 Ctrl+C 强制结束")
    print("=" * 55)

    cap = _open_camera(args.camera, cam_w, cam_h)
    if cap is None:
        print("[!] 摄像头打开失败, 程序退出")
        return

    # 采集与处理解耦: 后台线程持续抓帧, 主循环不再阻塞在 cap.read()
    reader = CameraReader(cap)

    # ---- 帧率统计 (指数滑动平均, 平滑显示) ----
    fps = 0.0
    prev_t = time.perf_counter()

    # ---- 阶段耗时统计周期 ----
    read_ms_list = []            # 当前周期内每帧 采集 耗时 (ms)
    detect_ms_list = []          # 当前周期内每帧 检测 总耗时 (ms)
    draw_ms_list = []            # 当前周期内每帧 绘制 耗时 (ms)
    show_ms_list = []            # 当前周期内每帧 显示 耗时 (ms)
    frame_count = 0              # 总帧数 (用于触发周期打印)

    # ---- 主循环 ----
    try:
        while True:
            # 帧率统计
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            # 1. 采集 (后台线程取帧, 不阻塞主循环)
            t0 = time.perf_counter()
            ret, frame = reader.read()
            read_ms = (time.perf_counter() - t0) * 1000.0
            read_ms_list.append(read_ms)
            if not ret:
                print("\r[!] 读取帧失败...", end='', flush=True)
                continue
            frame_count += 1

            # 2. 检测 (内部各阶段由 StageTimer 自动计时)
            t0 = time.perf_counter()
            result = detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            detect_ms_list.append(detect_ms)

            # 3. 绘制
            t0 = time.perf_counter()
            vis = detector.draw_result(frame, result, fps=fps)
            draw_ms = (time.perf_counter() - t0) * 1000.0
            draw_ms_list.append(draw_ms)

            '''
            # 4. 显示
            t0 = time.perf_counter()
            cv2.imshow(WIN_NAME, vis)
            cv2.waitKey(1)   # 仅用于刷新显示窗口, 不做按键退出处理
            show_ms = (time.perf_counter() - t0) * 1000.0
            show_ms_list.append(show_ms)
            '''

            # 4. 控制台单行状态
            if result['detected']:
                print(f"\r  offset={result['offset']:+7.1f}px  "
                      f"angle={result['angle']:+6.1f}deg  "
                      f"FPS={fps:5.1f}  detect={detect_ms:6.2f}ms    ",
                      end='', flush=True)
            else:
                print(f"\r  [未检测到引导线]  "
                      f"FPS={fps:5.1f}  detect={detect_ms:6.2f}ms    ",
                      end='', flush=True)

            # 5. 周期性打印各阶段耗时 + 每帧耗时分解
            if frame_count % args.perf_every == 0:
                _print_stage_summary(timer, detect_ms_list,
                                     read_ms_list, draw_ms_list, show_ms_list, fps)
                timer.reset()
                read_ms_list = []
                detect_ms_list = []
                draw_ms_list = []
                show_ms_list = []

    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C, 正在退出...")
        # 退出前打印最后一个周期的耗时汇总
        if detect_ms_list:
            _print_stage_summary(timer, detect_ms_list,
                                 read_ms_list, draw_ms_list, show_ms_list, fps)
    finally:
        # 无论正常/异常退出都释放资源 (reader.release 内部会释放 cap)
        reader.release()
        cv2.destroyAllWindows()
        print("[*] 摄像头已释放, 程序结束")


if __name__ == '__main__':
    main()
