"""
main_bottom.py - 下视摄像头功能测试入口

功能:
  以下视摄像头的视频 (或视频文件) 作为输入源, 采用多线程 + 帧级同步,
  同时调用三个任务模块, 实现对 引导线 / 目标球 / 收集框 的识别:
    P1 巡线 -> LineDetector  (HSV 橙红色分割 + 最小二乘拟合, 输出偏移/角度)
    P4 抓球 -> BallDetector  (完全复用 P2 撞球的球检测, 即 main_front 中的适配器)
    P5 放球 -> BoxDetector   (红色闭合矩形收集框, P3 穿门同样复用其内核)

  三个任务的检测结果叠加绘制到同一帧显示 (单窗口, 详见 _compose).

所有配置参数集中在 main_bottom_config.py, 命令行参数可覆盖配置中的同名项.

线程模型 (主线程 + 3 个工作线程, 两个 Barrier 完成一次帧同步):
  主线程     : 读帧 -> 发布 -> 回收结果 -> 汇总显示
  P1 线程    : LineDetector.detect(frame)
  P4 线程    : BallDetector.detect(frame)
  P5 线程    : BoxDetector.detect(frame)
  barrier_frame  : 新一帧就绪, 四方同时出发, 保证 P1/P4/P5 处理的是同一帧
  barrier_result : P1/P4/P5 检测完成, 四方会合, 主线程统一汇总显示

运行示例:
  python main_bottom.py --source D:\\RC\\bottom.mp4
  python main_bottom.py --source 0 --target red
  python main_bottom.py --source bottom.mp4 --output out.mp4
  python main_bottom.py --source bottom.mp4 --backend hbm   # RDK 板端 BPU 推理
"""

import argparse
import importlib.util
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ================================================================
# 常量
# ================================================================
_PROJ_ROOT = Path(__file__).resolve().parent          # V3.0 根目录
_P1_DIR = _PROJ_ROOT / "P1 巡线"
_P2_DIR = _PROJ_ROOT / "P2 撞球"    # P4 复用 P2, 默认模型位于此目录
_P5_DIR = _PROJ_ROOT / "P5 放球"
_FRONT_PATH = _PROJ_ROOT / "main_front.py"           # 复用其 P2 球检测适配器
WIN_NAME = "BottomCam | P1 line + P4 ball + P5 box"


# ================================================================
# 通用工具
# ================================================================

def load_module_by_path(module_name: str, path: Path):
    """按文件路径动态加载 Python 模块 (兼容中文/数字目录名)。

    各任务文件夹名含数字与中文, 无法用标准 import 语句,
    故使用 importlib.util.spec_from_file_location 按路径加载。
    """
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 复用 main_front 的 P2 球检测适配器 (P4 抓球完全复用 P2 撞球, 不重复实现)
_front_mod = load_module_by_path("main_front", _FRONT_PATH)
BallDetector = _front_mod.BallDetector            # P4 球检测 = P2 球检测
load_labels = _front_mod.load_labels
strip_path_quotes = _front_mod.strip_path_quotes
build_preprocess_config = _front_mod.build_preprocess_config


# 加载本文件同目录的配置模块 (main_bottom_config.py)
_CFG_PATH = _PROJ_ROOT / "main_bottom_config.py"
_cfg_mod = load_module_by_path("bottom_main_config", _CFG_PATH)
DEFAULT_CONFIG = _cfg_mod.DEFAULT_CONFIG
get_config = _cfg_mod.get_config


# ================================================================
# P1 巡线 - 引导线检测器创建
# ================================================================

def create_line_detector(overrides=None):
    """创建 P1 巡线检测器 (池底橙红色引导线, 输出偏移/角度)。

    Args:
        overrides: dict, 对 P1 巡线/config.py 的参数覆盖
    """
    ld_mod = load_module_by_path("P1_line_detector", _P1_DIR / "line_detector.py")
    return ld_mod.LineDetector(ld_mod.get_config(overrides))


def create_box_detector(work_size, overrides=None):
    """创建 P5 收集框检测器 (红色闭合矩形, 与 P3 穿门同内核)。

    Args:
        work_size: (宽, 高), 参考中心点默认取画面中心
        overrides: dict, 对 P5 放球/config.py 的参数覆盖 (未设置 camera_center 时自动对齐)
    """
    w, h = work_size
    cfg = dict(overrides or {})
    cfg.setdefault("camera_center", (w / 2.0, h / 2.0))
    bd_mod = load_module_by_path("P5_box_detector", _P5_DIR / "box_detector.py")
    return bd_mod.BoxDetector(bd_mod.get_config(cfg))


# ================================================================
# 下视摄像头多线程测试器
# ================================================================

class BottomCamTester:
    """下视摄像头三任务并发测试器。

    线程模型:
      主线程       读帧 -> 发布 -> 回收 -> 汇总显示
      P1 工作线程  LineDetector.detect
      P4 工作线程  BallDetector.detect (复用 P2)
      P5 工作线程  BoxDetector.detect

    同步方式 (threading.Barrier, 参与方 = 4):
      _barrier_frame  : 主线程发布新帧后与三个工作线程会合, 四方同时开跑,
                        保证 P1/P4/P5 处理的是同一帧 (帧级同步).
      _barrier_result : 三个工作线程完成检测后与主线程会合, 主线程统一汇总显示.
    """

    def __init__(self, source, line_detector, ball_detector, box_detector,
                 work_size=(640, 480), show=True, output=None):
        self.source = source
        self.line = line_detector
        self.ball = ball_detector
        self.box = box_detector
        self.work_w, self.work_h = work_size
        self.show = show
        self.output = output

        self._stop = threading.Event()
        self._barrier_frame = threading.Barrier(4)
        self._barrier_result = threading.Barrier(4)
        self._result_lock = threading.Lock()
        self._cur_frame = None
        self._results = {"line": None, "ball": [], "box": None, "err": {}}

    # ---------- 工作线程 ----------

    def _p1_worker(self):
        """P1 巡线工作线程。"""
        while not self._stop.is_set():
            if not self._wait(self._barrier_frame):
                break
            frame = self._cur_frame.copy()
            try:
                self._put_result("line", self.line.detect(frame))
            except Exception as exc:
                self._put_error("P1", exc)
            if not self._wait(self._barrier_result):
                break

    def _p4_worker(self):
        """P4 抓球工作线程 (完全复用 P2 的球检测)。"""
        while not self._stop.is_set():
            if not self._wait(self._barrier_frame):
                break
            frame = self._cur_frame.copy()
            try:
                self._put_result("ball", self.ball.detect(frame))
            except Exception as exc:
                self._put_error("P4", exc)
            if not self._wait(self._barrier_result):
                break

    def _p5_worker(self):
        """P5 放球工作线程 (红色闭合矩形收集框)。"""
        while not self._stop.is_set():
            if not self._wait(self._barrier_frame):
                break
            frame = self._cur_frame.copy()
            try:
                self._put_result("box", self.box.detect(frame))
            except Exception as exc:
                self._put_error("P5", exc)
            if not self._wait(self._barrier_result):
                break

    # ---------- 同步与共享状态 ----------

    def _wait(self, barrier) -> bool:
        """等待会合点; 线程被终止 (Barrier 被破坏) 时返回 False。"""
        try:
            barrier.wait()
            return True
        except threading.BrokenBarrierError:
            self._stop.set()
            return False

    def _publish(self, frame, frame_idx):
        """发布新帧供三个工作线程处理, 并清空上一帧结果。"""
        self._cur_frame = frame
        with self._result_lock:
            self._results = {"line": None, "ball": [], "box": None, "err": {}}

    def _collect(self):
        """回收 P1/P4/P5 的检测结果。"""
        with self._result_lock:
            return (self._results.get("line"),
                    self._results.get("ball", []),
                    self._results.get("box"),
                    self._results.get("err", {}))

    def _put_result(self, key, value):
        with self._result_lock:
            self._results[key] = value

    def _put_error(self, name, exc):
        with self._result_lock:
            self._results["err"][name] = repr(exc)
        print(f"[WARN] P{name} 检测异常: {exc!r}")

    # ---------- 主流程 ----------

    def run(self):
        """启动多线程流水线并开始处理视频源。"""
        cap = self._open_source()
        writer = self._create_writer(cap)
        threads = [
            threading.Thread(target=self._p1_worker, name="P1-line", daemon=True),
            threading.Thread(target=self._p4_worker, name="P4-ball", daemon=True),
            threading.Thread(target=self._p5_worker, name="P5-box", daemon=True),
        ]
        for t in threads:
            t.start()

        if self.show:
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            # 单帧叠加显示, 窗口放大 2 倍便于观察 (不超 1440x1080)
            cv2.resizeWindow(WIN_NAME, min(self.work_w * 2, 1440),
                             min(self.work_h * 2, 1080))

        try:
            self._main_loop(cap, writer)
        finally:
            self._shutdown(threads, cap, writer)

    def _open_source(self):
        """打开视频文件或摄像头 (自动去除路径误带的引号)。"""
        source = strip_path_quotes(self.source) if isinstance(self.source, str) else self.source
        try:
            source_int = int(source)
        except (ValueError, TypeError):
            source_int = None
        cap = cv2.VideoCapture(source_int if source_int is not None else source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")
        if source_int is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.work_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.work_h)
        return cap

    def _create_writer(self, cap):
        """创建结果视频写入器 (未配置 --output 时返回 None)。"""
        if not self.output:
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # 单帧叠加结果, 尺寸与工作分辨率一致
        writer = cv2.VideoWriter(
            self.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (self.work_w, self.work_h))
        if not writer.isOpened():
            print(f"[WARN] 无法创建输出视频: {self.output}")
            return None
        print(f"[MAIN] 结果视频将保存到: {self.output}")
        return writer

    def _main_loop(self, cap, writer):
        """主循环: 读帧 -> 发布 -> 帧同步 -> 回收 -> 汇总显示。"""
        frame_idx = 0
        fps = 0.0
        while not self._stop.is_set():
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print(f"[MAIN] 视频读取结束, 共处理 {frame_idx} 帧")
                break

            frame = self._to_work_size(frame)
            self._publish(frame, frame_idx)

            # 帧交接会合: 等 P1/P4 线程拿到本帧
            if not self._wait(self._barrier_frame):
                break
            # 结果回收会合: 等 P1/P4 完成本帧检测
            if not self._wait(self._barrier_result):
                break

            line_res, ball_dets, box_res, err = self._collect()
            if frame_idx % 10 == 0 or not self.show:
                self._print_summary(frame_idx, line_res, ball_dets, box_res)

            vis = self._compose(frame, frame_idx, line_res, ball_dets, box_res, fps, err)
            if writer is not None:
                writer.write(vis)
            if self.show:
                cv2.imshow(WIN_NAME, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[MAIN] 用户按下 q, 退出")
                    break

            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            frame_idx += 1

    def _to_work_size(self, frame):
        """将帧缩放到统一工作分辨率, 保证各检测坐标一致。"""
        h, w = frame.shape[:2]
        if (w, h) != (self.work_w, self.work_h):
            frame = cv2.resize(frame, (self.work_w, self.work_h),
                               interpolation=cv2.INTER_LINEAR)
        return frame

    def _print_summary(self, frame_idx, line_res, ball_dets, box_res):
        """控制台输出当前帧的 线/球/框 检测概况。"""
        if line_res is not None and line_res["detected"]:
            line_txt = (f"line offset={line_res['offset']:+.1f}px "
                        f"angle={line_res['angle']:+.1f}deg")
        elif line_res is None:
            line_txt = "line=ERR"
        else:
            line_txt = "line=None"

        if box_res is not None and box_res["detected"]:
            box_txt = f"box offset={box_res['offset']:+.1f}px"
        elif box_res is None:
            box_txt = "box=ERR"
        else:
            box_txt = "box=None"

        print(f"[Frame {frame_idx:4d}] ball={len(ball_dets)} | {line_txt} | {box_txt}")

    def _compose(self, frame, frame_idx, line_res, ball_dets, box_res, fps, err):
        """把 P1/P4/P5 三个任务的检测结果叠加绘制到同一帧 (单窗口显示)。

        图形叠加: P1 重心/偏移箭头/拟合方向, P4 球外接框, P5 框口矩形/中心/偏移箭头.
        状态文字: 三任务分行显示于左上角, 各检测线程异常提示在左下角.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx_img = w // 2

        # 画面垂直中心参考线 (帮助观察各偏移量)
        cv2.line(vis, (cx_img, 0), (cx_img, h), (255, 255, 255), 1)

        # ---- P1 巡线: 重心 + 偏移箭头 + 拟合方向 ----
        if line_res is not None and line_res["detected"]:
            cx, cy = int(line_res["center"][0]), int(line_res["center"][1])
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.arrowedLine(vis, (cx_img, cy), (cx, cy), (255, 255, 0), 2, tipLength=0.3)
            ang = np.radians(line_res["angle"])
            length = 80
            dx, dy = int(length * np.sin(ang)), int(length * np.cos(ang))
            cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 0, 255), 2)

        # ---- P4 抓球: 外接框 + 中心点 ----
        for det in ball_dets:
            # bbox/center 可能为浮点, cv2 绘制需整型
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            cx, cy = int(det["center"][0]), int(det["center"][1])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(vis, f"{det['label']} {det['score']:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---- P5 放球: 框口矩形 + 中心 + 偏移箭头 ----
        if box_res is not None and box_res["detected"]:
            x1, y1, x2, y2 = box_res["bbox"]
            cx, cy = int(box_res["center"][0]), int(box_res["center"][1])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
            cv2.arrowedLine(vis, (cx_img, cy), (cx, cy), (0, 255, 0), 2, tipLength=0.3)

        # ---- 状态文字 (三任务分行, 避免重叠) ----
        y = 20
        cv2.putText(vis, f"BottomCam | frame={frame_idx} | fps={fps:.1f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 24

        # P1 巡线状态
        if line_res is not None and line_res["detected"]:
            txt = (f"P1 line | offset={line_res['offset']:+.1f}px "
                   f"angle={line_res['angle']:+.1f}deg")
            color = (255, 255, 0)
        else:
            txt, color = ("P1 line | ERR" if line_res is None else "P1 line | None"), (0, 0, 255)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 24

        # P4 抓球状态
        cv2.putText(vis, f"P4 ball | n={len(ball_dets)}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        y += 24

        # P5 放球状态
        if box_res is not None and box_res["detected"]:
            txt = f"P5 box | offset={box_res['offset']:+.1f}px angle={box_res['angle']:+.1f}deg"
            color = (0, 255, 0)
        else:
            txt, color = ("P5 box | ERR" if box_res is None else "P5 box | None"), (0, 0, 255)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 各检测线程异常提示 (左下角)
        for i, name in enumerate(("P1", "P4", "P5")):
            if err.get(name):
                cv2.putText(vis, f"{name} ERR: {err[name]}", (10, h - 20 - i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return vis

    def _shutdown(self, threads, cap, writer):
        """终止工作线程并释放资源。"""
        self._stop.set()
        # 打破两个 Barrier, 让阻塞等待的工作线程立即退出
        for barrier in (self._barrier_frame, self._barrier_result):
            try:
                barrier.abort()
            except Exception:
                pass
        for t in threads:
            t.join(timeout=3)
        cap.release()
        if writer is not None:
            writer.release()
        if self.show:
            cv2.destroyAllWindows()
        print("[MAIN] 资源已释放, 测试结束")

    def request_stop(self):
        """请求停止本测试器 (供 main_RDK 协调双摄像头同时退出)。

        设置停止标志并打破两个 Barrier, 使阻塞中的工作线程/主循环立即退出。
        仅新增方法, 不影响原有调用方式, 向后兼容。
        """
        self._stop.set()
        for barrier in (self._barrier_frame, self._barrier_result):
            try:
                barrier.abort()
            except Exception:
                pass


# ================================================================
# 命令行入口
# ================================================================

def parse_args():
    """解析命令行参数; 未指定的参数使用 main_bottom_config.py 中的默认值。"""
    parser = argparse.ArgumentParser(
        description="下视摄像头三任务并发测试 (P1 巡线 + P4 抓球 + P5 放球)")
    parser.add_argument("--source", type=str, default=None,
                        help="视频文件路径或摄像头编号, 例如 0 / bottom.mp4")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["ultralytics", "hbm"],
                        help="P4 球检测后端: ultralytics(.pt 开发机) / hbm(.hbm RDK 板端)")
    parser.add_argument("--model", type=str, default=None,
                        help="P4 球检测模型路径 (默认按后端取 P2 撞球 下的默认模型)")
    parser.add_argument("--label-file", type=str, default=None,
                        help="hbm 后端类别名称文件 (每行一个类别名)")
    parser.add_argument("--target", type=str, nargs="+", default=None,
                        help="P4 只保留的目标类别, 例如 --target red; 不指定则保留全部")
    parser.add_argument("--conf", type=float, default=None, help="P4 置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="P4 NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="ultralytics 后端推理分辨率")
    parser.add_argument("--no-preprocess", action="store_true", default=None,
                        help="关闭 P4 的水下预处理")
    parser.add_argument("--width", type=int, default=None, help="工作分辨率宽")
    parser.add_argument("--height", type=int, default=None, help="工作分辨率高")
    parser.add_argument("--output", type=str, default=None,
                        help="结果视频保存路径 (可选)")
    parser.add_argument("--show", action="store_true", default=None,
                        help="显示检测窗口")
    parser.add_argument("--no-show", action="store_false", dest="show", default=None,
                        help="不显示窗口")
    return parser.parse_args()


def build_config(opt):
    """合并配置: 以 main_bottom_config.py 为基准, 命令行参数覆盖同名项。"""
    cfg = get_config()
    # 输入源
    if opt.source is not None:
        cfg["source"] = opt.source
    if opt.width is not None:
        cfg["work_width"] = opt.width
    if opt.height is not None:
        cfg["work_height"] = opt.height
    # P4 球检测 (复用 P2)
    if opt.backend is not None:
        cfg["p4_backend"] = opt.backend
    if opt.model is not None:
        cfg["p4_model"] = opt.model
    if opt.label_file is not None:
        cfg["p4_label_file"] = opt.label_file
    if opt.target is not None:
        cfg["p4_target_classes"] = opt.target
    if opt.conf is not None:
        cfg["p4_conf"] = opt.conf
    if opt.iou is not None:
        cfg["p4_iou"] = opt.iou
    if opt.imgsz is not None:
        cfg["p4_imgsz"] = opt.imgsz
    if opt.no_preprocess is not None:
        cfg["p4_enable_preprocess"] = not opt.no_preprocess
    # 显示与输出
    if opt.show is not None:
        cfg["show"] = opt.show
    if opt.output is not None:
        cfg["output"] = opt.output
    return cfg


def resolve_model_path(cfg):
    """解析 P4 模型路径: 未配置时按后端取默认模型, 相对路径基于 P2 目录展开。"""
    model = cfg["p4_model"]
    if model is None:
        default_name = (cfg["p4_default_model_pt"] if cfg["p4_backend"] == "ultralytics"
                        else cfg["p4_default_model_hbm"])
        return str(_P2_DIR / default_name)
    if os.path.isabs(model):
        return model
    return str(_P2_DIR / model)


def main():
    opt = parse_args()
    cfg = build_config(opt)
    work_size = (cfg["work_width"], cfg["work_height"])

    # ---- P1 巡线检测器 ----
    line_detector = create_line_detector(cfg.get("p1_overrides"))
    print(f"[P1] 巡线检测器加载完成, 引导线 HSV: "
          f"{cfg['p1_overrides']['hsv_lower']}~{cfg['p1_overrides']['hsv_upper']}")

    # ---- P4 抓球检测器 (完全复用 P2) ----
    model_path = resolve_model_path(cfg)
    ball_detector = BallDetector(
        model_path=model_path,
        backend=cfg["p4_backend"],
        conf=cfg["p4_conf"],
        iou=cfg["p4_iou"],
        imgsz=cfg["p4_imgsz"],
        target_classes=cfg["p4_target_classes"],
        label_file=cfg["p4_label_file"],
        enable_preprocess=cfg["p4_enable_preprocess"],
        preprocess_cfg=cfg,
    )
    print(f"[P4] 球检测后端: {cfg['p4_backend']} (完全复用 P2), 模型: {model_path}, "
          f"目标类别: {cfg['p4_target_classes'] or '全部'}")

    # ---- P5 放球检测器 (红色闭合矩形收集框) ----
    box_detector = create_box_detector(work_size, cfg.get("p5_overrides"))
    print(f"[P5] 收集框检测器加载完成, camera_center: {work_size[0] / 2:.0f}, {work_size[1] / 2:.0f}")

    # ---- 启动多线程测试 ----
    tester = BottomCamTester(
        source=cfg["source"],
        line_detector=line_detector,
        ball_detector=ball_detector,
        box_detector=box_detector,
        work_size=work_size,
        show=cfg["show"],
        output=cfg["output"],
    )
    tester.run()


if __name__ == "__main__":
    main()
