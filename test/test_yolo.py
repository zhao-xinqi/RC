#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_yolo.py - camera 0 高速采集(200fps) + best.pt YOLO 多线程实时识别测试

目标: 以 ~200fps 采集 camera 0 的画面, 多个 worker 线程并行跑 best.pt 检测,
      标注帧存到 ./test/yolo_test。采集线程把每一帧都推入有界队列 (不覆盖丢弃),
      让处理尽量追平采集帧率; 队列满时才丢最旧帧作为兜底。

实时性设计 (依据 ./test/README + 实测):
  - 单帧 predict 约 11ms, 实测纯 GPU 前向就是 ~11.7ms (FP32) —— 瓶颈在 GPU 本身,
    且小模型单帧前向是延迟主导 (启动开销大), FP16 反而更慢 (15.7ms);
  - 纯多线程共享同一块 GPU 帮不上忙 (实测 4 线程单帧才 ~104fps);
  - 只有批处理能榨出 GPU: 1 线程 batch=8 可达 ~540fps, 4线程 batch=4 ~386fps;
  - 因此用 "多线程 + 机会式小批量": 每个 worker 取 1 帧立即处理, 推理前顺手把
    队列里已有的新帧一起带走 (一次前向吃掉一批)。负载轻 = 单帧低延迟;
    负载重 = 自动形成小批量吃满 GPU。延迟被有界队列锁在 ~10-20ms, 适合实时。

结构 (采集线程每帧直接入队, 消除"只留最新帧被覆盖"的丢帧点):
  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────┐   ┌─────────┐
  │ 采集线程       │──▶│ 有界分发队列    │──▶│ N个推理worker线程         │──▶│ 后台保存  │
  │ 独占read,每帧入队│   │ 满才丢最旧帧    │   │ 各自独立模型, 机会式小批量predict│   │ imwrite │
  └──────────────┘   └──────────────┘   └────────────┬────────────┘   └─────────┘
                                                     │ 每个worker发布最新标注帧
                                                     ▼
                                         主循环按固定节拍弹窗 (~60fps)

用法 (需在 YOLO conda 环境下运行, 含 ultralytics + torch):
  conda activate YOLO
  python test_yolo.py                          # camera 0, 默认无限运行+弹窗 (Esc/Ctrl+C停止)
  python test_yolo.py --workers 2 --batch 4    # 2 worker, 每worker最多批4帧
  python test_yolo.py --workers 8              # 调 worker 线程数 (观察吞吐变化)
  python test_yolo.py --device 1               # 换 camera 1 (实测只有~26fps)
  python test_yolo.py --duration 5             # 固定测5秒, 到时自动退出
  python test_yolo.py --duration 0             # 无限运行, 直到 Esc (弹窗) 或 Ctrl+C
  python test_yolo.py --conf 0.5               # 提高置信度阈值
  python test_yolo.py --no-save                # 不存图, 只弹窗实时看 (Esc退出)
  python test_yolo.py --no-show                # 不弹窗 (headless), 只存标注帧
"""

import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np

# ================================================================
# 可调参数 (默认值, 均可用命令行覆盖)
# ================================================================

CAMERA_INDEX = 0    # 外接高速摄像头当前索引是 0 (README: Windows 枚举从 1 变到 0)
FORMAT = "MJPG"     # 必须 MJPG 才能上 200fps (YUYV 顶死 USB2.0 只有 60fps)
DEFAULT_W = 640     # 实测中 "最高分辨率 + 最高帧率" 的最优组合
DEFAULT_H = 480
FPS_TARGET = 1000   # 请求超高帧率, 逼相机协商出上限 (协商出 400, 链路实测 ~204)
CONF_THRESH = 0.25  # YOLO 置信度阈值
WORKERS = 8         # 推理 worker 线程数 (共享同一块 GPU)
BATCH_SIZE = 4      # 每个 worker 机会式批处理的最大帧数 (取到帧后顺手带上队列里已有的)
LOG_INTERVAL = 1.0  # 终端 fps 打印间隔(秒)
SHOW_INTERVAL = 1.0 / 60   # 弹窗节流: 主循环每 SHOW_INTERVAL 秒显示一帧最新标注帧 (~60fps)
GET_TIMEOUT = 0.2   # worker 取帧超时(秒): 空队列时回去检查 stop_evt, 保证能退出

# 以脚本所在目录为基准, 保证无论从哪个 cwd 运行都能找到模型和输出目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "best.pt")   # ./test/best.pt
SAVE_DIR = os.path.join(_SCRIPT_DIR, "yolo_test")   # ./test/yolo_test


# ================================================================
# 基础工具
# ================================================================

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
    """按 MSMF + MJPG + 640x480 + 高帧率请求 打开相机, 返回 cap 或 None.

    用 cap.get() 读回协商值只是为了打印参考; 真实帧率信实测 (perf_counter),
    不信 cap.get(FPS) (README: MSMF 下读回值不可靠).
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FORMAT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    return cap


# ================================================================
# 采集线程: 独占 cap.read(), 每帧推入有界分发队列
# ================================================================

class Grabber:
    """采集线程体: 独占 read(), 每读到一帧就推入分发队列 + 累计帧数.

    直接把每帧推进有界队列 (而非只留最新一帧), 消除"只存最新被覆盖"这一
    丢帧点: 只要相机读出来, 就进队列等 worker 处理。队列满时丢最旧帧
    (dispatch 内的兜底), 既保持实时又不阻塞采集。采集链路因此不受 YOLO 处理
    速度影响 (稳定 ~204fps)。
    """

    def __init__(self, cap, out_q, stop_evt):
        self.cap = cap
        self.lock = threading.Lock()
        self.out_q = out_q
        self.stop_evt = stop_evt
        self.count = 0      # 累计成功读取帧数

    def run(self):
        """采集循环, 在独立线程中运行 (daemon)."""
        while not self.stop_evt.is_set():
            ok, f = self.cap.read()
            if not ok:
                continue
            if self.stop_evt.is_set():
                break      # 退出前最后读的一帧不再入队
            dispatch(self.out_q, f)   # 入有界队列, 满则丢最旧 (丢帧只发生在这里)
            with self.lock:
                self.count += 1


# ================================================================
# 模型加载与单帧推理
# ================================================================

def load_models(count, path, conf, device):
    """加载 count 个独立 YOLO 实例, 返回 (模型列表, 实际推理设备).

    每个 worker 线程持有一个独立实例, 避免多个线程并发调用同一 predict
    的内部状态竞争. 模型很小 (6MB), 多开几份内存开销可忽略.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "[!] 未安装 ultralytics, 请先在 YOLO conda 环境下运行:\n"
            "    conda activate YOLO")

    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    models = []
    for _ in range(count):
        m = YOLO(path)
        m.conf = conf   # 给 predict() 默认置信度阈值
        models.append(m)
    print("[*] 加载模型: %s x%d, 类别: %s, 推理设备: %s"
          % (path, count, list(models[0].names.values()), device))
    return models, device


def run_batch(model, frames, device):
    """对一帧集合做一次批量推理, 返回 (标注帧列表, 每帧检测框数量列表).

    批处理是吃满这块 GPU 的关键 (实测单帧~11.7ms, 4帧一批也才~11ms).
    plot() 直接得到画好检测框/标签/置信度的帧, 画框开销实测可忽略.
    """
    results = model.predict(frames, verbose=False, device=device)
    annotated_list = [r.plot() for r in results]
    det_counts = [len(r.boxes) for r in results]
    return annotated_list, det_counts


# ================================================================
# 线程安全统计
# ================================================================

class Stats:
    """多 worker 共享的处理统计 (锁保护)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0    # 已跑过 YOLO 的帧数
        self.det_frames = 0   # 检出目标的帧数

    def add(self, n_det):
        """记录一帧处理完成, n_det 为该帧检测框数量."""
        with self.lock:
            self.processed += 1
            if n_det:
                self.det_frames += 1

    def snapshot(self):
        """返回 (processed, det_frames)."""
        with self.lock:
            return self.processed, self.det_frames


# ================================================================
# 共享"最新标注帧": 让弹窗不卡的关键
# ================================================================

class LatestFrame:
    """所有 worker 共享的最新已标注帧.

    每个 worker 处理完就把结果 publish 进来, 主循环按固定时间节拍
    (SHOW_INTERVAL) 取最新一张显示。显示帧率因此只由主循环节拍决定 (~60fps),
    不再受"只显示单个 worker 自己的帧"拖累 (8 worker 时旧方案仅 ~11fps, 看着卡).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None      # 最新的标注帧 (numpy BGR), 尚未发布时为 None

    def publish(self, frame):
        """发布一帧最新标注图 (任意 worker 调用, 不阻塞)."""
        with self.lock:
            self.frame = frame

    def latest(self):
        """取当前最新标注帧, 还没有时返回 None."""
        with self.lock:
            return self.frame


# ================================================================
# 后台保存线程: imwrite 不占推理时间
# ================================================================

class Saver:
    """后台保存线程: worker 只把标注帧丢进队列, 保存线程负责写盘.

    文件名序号由保存线程自己分配, 避免多 worker 并发取名冲突.
    """

    def __init__(self, out_dir):
        self.q = queue.Queue()
        self.out_dir = out_dir
        self.saved = 0     # 已写入张数 (仅保存线程访问, 结束时主线程读)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            annotated = self.q.get()
            if annotated is None:      # None 为退出哨兵
                break
            path = os.path.join(self.out_dir, "frame_%05d.jpg" % self.saved)
            cv2.imwrite(path, annotated)
            self.saved += 1
            self.q.task_done()

    def put(self, annotated):
        """把一帧标注图排入保存队列, 立即返回 (不阻塞推理)."""
        self.q.put(annotated)

    def join(self):
        """等队列全部写完, 再退出保存线程 (结束时调用)."""
        self.q.join()
        self.q.put(None)
        self.thread.join()


# ================================================================
# 分发与 worker
# ================================================================

def dispatch(q, frame):
    """有界队列入队: 满了丢最旧帧, 保证处理延迟不堆积 (实时关键).

    采集 200fps > 处理吞吐时, 队列始终保持容量上限, 每来一帧丢一帧最旧的,
    内存占用与延迟恒定, 不会越积越多.
    """
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put(frame)


def make_worker(model, device, dispatch_q, saver, stats, batch, stop_evt, latest):
    """构造一个推理 worker 的线程函数 (闭包绑定各自模型).

    机会式小批量: 取 1 帧立即处理, 推理前顺手把队列里已有的新帧一起带走
    (最多 batch-1 帧), 一次前向吃掉一批 -> 负载轻时低延迟, 负载重时吃满 GPU.
    每处理完一帧就把标注结果 publish 到 latest, 弹窗统一由主循环按时间节拍
    显示 (cv2.imshow 非线程安全, 只允许主循环调用).
    退出: get 带超时, 每 GET_TIMEOUT 秒检查一次 stop_evt, 不依赖收尾哨兵
    (哨兵方案在队列满时会被卡死, 见 shutdown 注释).
    """
    def work():
        while not stop_evt.is_set():
            try:
                f = dispatch_q.get(timeout=GET_TIMEOUT)
            except queue.Empty:
                continue        # 队列暂时空: 回去检查 stop_evt, 支持被强制停止
            frames = [f]
            for _ in range(batch - 1):
                try:
                    nf = dispatch_q.get_nowait()
                except queue.Empty:
                    break                     # 队列空了, 就只处理已拿到的
                frames.append(nf)

            annotated_list, det_counts = run_batch(model, frames, device)
            for annotated, n_det in zip(annotated_list, det_counts):
                stats.add(n_det)
                if saver is not None:
                    saver.put(annotated)
                latest.publish(annotated)     # 发布最新标注帧供弹窗显示
    return work


# ================================================================
# 主流程
# ================================================================

def main():
    ap = argparse.ArgumentParser(
        description="camera 高速采集(200fps) + best.pt YOLO 多线程实时识别测试")
    ap.add_argument("--device", type=int, default=CAMERA_INDEX,
                    help="相机索引, 默认%d (camera 1 实测仅~26fps)" % CAMERA_INDEX)
    ap.add_argument("--w", type=int, default=DEFAULT_W, help="分辨率宽, 默认640")
    ap.add_argument("--h", type=int, default=DEFAULT_H, help="分辨率高, 默认480")
    ap.add_argument("--duration", type=float, default=0,
                    help="测试时长(秒), 默认0 = 无限运行直到 Ctrl+C/Esc; >0 到时自动退出")
    ap.add_argument("--model", default=MODEL_PATH,
                    help="模型路径, 默认 ./test/best.pt")
    ap.add_argument("--conf", type=float, default=CONF_THRESH,
                    help="置信度阈值, 默认%.2f" % CONF_THRESH)
    ap.add_argument("--device-idx", dest="ul_device", default="auto",
                    help="推理设备: auto/cpu/cuda索引, 默认auto")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="推理 worker 线程数, 默认%d (共享同一块GPU)" % WORKERS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE,
                    help="每worker机会式批处理最大帧数, 默认%d" % BATCH_SIZE)
    ap.add_argument("--out", default=SAVE_DIR,
                    help="标注帧保存目录, 默认 ./test/yolo_test")
    ap.add_argument("--no-save", action="store_true", help="不保存标注帧")
    ap.add_argument("--no-show", dest="show", action="store_false", default=True,
                    help="不弹窗 (headless), 只保存标注帧")
    a = ap.parse_args()

    # ---- 1. 打开相机 ----
    cap = open_camera(a.device, a.w, a.h)
    if cap is None:
        print("[!] 无法打开 camera %d (%s %dx%d)" % (a.device, FORMAT, a.w, a.h))
        return 1
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfps = cap.get(cv2.CAP_PROP_FPS)
    nfourcc = fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    print("== camera %d 请求 %s %dx%d -> 协商 %dx%d @%.0ffps (fourcc=%s) =="
          % (a.device, FORMAT, a.w, a.h, nw, nh, nfps, nfourcc))

    # ---- 2. 加载 N 个独立模型实例 (每个 worker 一个) ----
    models, ul_device = load_models(a.workers, a.model, a.conf, a.ul_device)

    # 预热每个模型一次推理 (黑帧): 首次 predict 要初始化 CUDA 上下文/算子,
    # 不预热会把测试窗口前半段吃掉 (实测首帧耗时 ~3s, 稳态只要 ~12ms)
    for m in models:
        m.predict(np.zeros((480, 480, 3), dtype=np.uint8),
                  verbose=False, device=ul_device)

    # ---- 3. 分发队列 + 采集线程 (每读到一帧直接入队) ----
    # 有界队列: 满则丢最旧帧保实时; 容量留足 workers*batch*2,
    # 让 worker 补批时队列里能有兄弟帧可拿, 同时吸收瞬时突发
    stop_evt = threading.Event()
    dispatch_q = queue.Queue(maxsize=a.workers * a.batch * 2)
    g = Grabber(cap, dispatch_q, stop_evt)
    threading.Thread(target=g.run, daemon=True).start()

    # ---- 4. 输出目录 ----
    out_dir = a.out
    saver = None
    if a.no_save:
        print("[*] --no-save: 不保存标注帧")
    else:
        os.makedirs(out_dir, exist_ok=True)
        # 清掉上一次运行留下的 frame_*.jpg, 保证每次运行输出干净
        for name in os.listdir(out_dir):
            if name.startswith("frame_") and name.endswith(".jpg"):
                os.remove(os.path.join(out_dir, name))
        saver = Saver(out_dir)
        print("[*] 标注帧保存到: %s" % out_dir)

    # ---- 5. 启动 N 个推理 worker ----
    stats = Stats()
    latest = LatestFrame()      # 共享最新标注帧, 弹窗由此取帧显示
    workers = []
    for model in models:
        t = threading.Thread(
            target=make_worker(model, ul_device, dispatch_q, saver, stats,
                               a.batch, stop_evt, latest),
            daemon=True)
        t.start()
        workers.append(t)

    # ---- 6. 主循环: 只统计与打印帧率 (采集线程已直接把帧推入队列) ----
    # duration=0 时无限运行, 直到 Esc (弹窗) 或 Ctrl+C 强制停止
    t0 = time.perf_counter()
    run_until = t0 + a.duration if a.duration > 0 else None
    last_cap = 0            # 上次采样的采集计数
    t_last = t0

    stop_note = "Ctrl+C 或 Esc" if a.show else "Ctrl+C"
    if a.duration > 0:
        print("[*] 测试 %g 秒, %d 个 worker x 最多批%d帧 ... (%s 可提前退出)"
              % (a.duration, a.workers, a.batch, stop_note))
    else:
        print("[*] 无限运行, %d 个 worker x 最多批%d帧 ... (%s 强制停止)"
              % (a.workers, a.batch, stop_note))

    try:
        last_show = 0.0         # 上次弹窗时刻 (perf_counter)
        while not stop_evt.is_set():
            if run_until is not None and time.perf_counter() >= run_until:
                break
            time.sleep(0.001)   # 关键: 让出 GIL, 否则死循环饿死采集线程 (~35fps)
            with g.lock:
                cur_cap = g.count

            # 弹窗: 主循环按固定节拍显示"最新标注帧" (~60fps 跟手, 不再卡顿)
            if a.show:
                now = time.perf_counter()
                if now - last_show >= SHOW_INTERVAL:
                    cur = latest.latest()
                    if cur is not None:
                        cv2.imshow("test_yolo", cur)
                        if cv2.waitKey(1) & 0xFF == 27:   # Esc 提前退出
                            stop_evt.set()
                    last_show = now

            # 每秒打印一次 采集/处理 fps
            now = time.perf_counter()
            if now - t_last >= LOG_INTERVAL:
                processed, det_frames = stats.snapshot()
                cap_fps = (cur_cap - last_cap) / (now - t_last)
                proc_fps = processed / (now - t0)
                print("采集 %6.1f fps | 处理 %6.1f fps | 已处理 %d 帧 | 检出 %d 帧 | 队列待处理 %d"
                      % (cap_fps, proc_fps, processed, det_frames, dispatch_q.qsize()),
                      flush=True)
                last_cap = cur_cap
                t_last = now
    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C, 停止测试并保存已处理帧 ...", flush=True)

    # ---- 7. 收尾: 停采集, 通知 worker 自行退出, 最后等保存写完 ----
    # 不用哨兵: 若队列在收尾时是满的 (采集>处理), put 哨兵会永久阻塞。
    # worker 用超时 get 感知 stop_evt 自行退出; 队列里未开始的帧直接丢弃 (停止语义)。
    stop_evt.set()          # 停采集线程 + 通知 worker 退出
    for t in workers:
        t.join(timeout=5)   # worker 最多 GET_TIMEOUT 秒后感知到 stop_evt 并退出
    if saver is not None:
        saver.join()        # 等已发布标注帧全部写完盘

    elapsed = time.perf_counter() - t0
    with g.lock:
        final_cap = g.count
    processed, det_frames = stats.snapshot()
    cap_fps = final_cap / elapsed if elapsed > 0 else 0.0
    proc_fps = processed / elapsed if elapsed > 0 else 0.0
    print("== 汇总 (%.2fs) ==" % elapsed)
    print("  采集帧率      : %.1f fps (累计 %d 帧)" % (cap_fps, final_cap))
    print("  处理帧率      : %.1f fps (共处理 %d 帧, %d 个 worker x 批%d)"
          % (proc_fps, processed, a.workers, a.batch))
    print("  检出目标帧    : %d / %d" % (det_frames, processed))
    if saver is not None:
        print("  已保存标注帧  : %d 张 -> %s/" % (saver.saved, out_dir))
    else:
        print("  已保存标注帧  : 0 (--no-save)")

    cap.release()
    if a.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
