"""
main_front.py - 前视摄像头功能测试入口

功能:
  以前视摄像头的视频 (或视频文件) 作为输入源, 采用多线程 + 帧级同步,
  同时调用两个任务模块, 实现对 目标球 与 门框 的识别:
    P2 撞球 -> BallDetector  (YOLO, 支持 ultralytics .pt / HBM .hbm 两种后端)
    P3 穿门 -> DoorDetector  (红色方形门框检测, 复用 P5 BoxDetector 内核)

所有配置参数集中在 main_front_config.py, 命令行参数可覆盖配置中的同名项.

线程模型 (主线程 + 2 个工作线程, 两个 Barrier 完成一次帧同步):
  主线程     : 读帧 -> 发布 -> 回收结果 -> 汇总显示
  P2 线程    : BallDetector.detect(frame)
  P3 线程    : DoorDetector.detect(frame)
  barrier_frame  : 新一帧就绪, 三线程同时出发, 保证 P2/P3 处理的是同一帧
  barrier_result : P2/P3 检测完成, 三线程会合, 主线程统一汇总显示

运行示例:
  python main_front.py --source test.mp4
  python main_front.py --source 0 --target red
  python main_front.py --source test.mp4 --output out.mp4
  python main_front.py --source test.mp4 --backend hbm   # RDK 板端 BPU 推理
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
_P2_DIR = _PROJ_ROOT / "P2 撞球"
_P3_DIR = _PROJ_ROOT / "P3 穿门"
_PREP_PATH = _PROJ_ROOT / "0.预处理" / "preprocessor.py"
WIN_NAME = "FrontCam | P2 ball + P3 door"


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


# 加载本文件同目录的配置模块 (main_front_config.py)
_CFG_PATH = _PROJ_ROOT / "main_front_config.py"
_cfg_mod = load_module_by_path("front_main_config", _CFG_PATH)
DEFAULT_CONFIG = _cfg_mod.DEFAULT_CONFIG
get_config = _cfg_mod.get_config


def _normalize_name(name: str) -> str:
    """标准化类别名, 兼容 大小写/空格/下划线/短横线 差异。"""
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def load_labels(label_file):
    """读取类别名称文件 (每行一个类别名)。"""
    if not label_file or not os.path.exists(label_file):
        return []
    with open(label_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def strip_path_quotes(path):
    """去除路径两端的引号 (配置中误带 " 或 ' 时仍能正常打开文件)。"""
    path = str(path).strip()
    while len(path) >= 2 and path[0] in "\"'" and path[-1] == path[0]:
        path = path[1:-1].strip()
    return path


def build_preprocess_config(cfg):
    """从配置中提取 P2 水下预处理参数字典。

    兼容两种键名:
      - 平铺键名 (preprocess_width / preprocess_red_boost / ...)  : main_front_config.py
      - 预处理键名 (image_width / red_boost / ...)                : 0.预处理 参数
    宽度/高度缺省时跟随工作分辨率。
    """
    def _pick(*names, default):
        for name in names:
            value = cfg.get(name)
            if value is not None:
                return value
        return default

    return {
        "image_width": _pick("preprocess_width", "image_width",
                             default=cfg.get("work_width", 640)),
        "image_height": _pick("preprocess_height", "image_height",
                              default=cfg.get("work_height", 480)),
        "enable_resize": _pick("preprocess_enable_resize", "enable_resize", default=True),
        "enable_color_correct": _pick("preprocess_enable_color_correct",
                                      "enable_color_correct", default=True),
        "red_boost": _pick("preprocess_red_boost", "red_boost", default=1.2),
        "enable_gaussian": _pick("preprocess_enable_gaussian", "enable_gaussian", default=True),
        "gaussian_kernel": _pick("preprocess_gaussian_kernel", "gaussian_kernel", default=5),
        "enable_clahe": _pick("preprocess_enable_clahe", "enable_clahe", default=True),
        "clahe_clip": _pick("preprocess_clahe_clip", "clahe_clip", default=2.0),
        "clahe_tile": _pick("preprocess_clahe_tile", "clahe_tile", default=(8, 8)),
    }


# ================================================================
# P2 撞球 - 球检测适配器
# ================================================================

class BallDetector:
    """P2 撞球 - 目标球检测适配器。

    对上层提供统一接口 detect(frame), 返回统一格式的检测结果列表:
        [{'label': str, 'score': float,
          'bbox': (x1, y1, x2, y2), 'center': (cx, cy)}, ...]

    支持两种推理后端:
      - "ultralytics": 开发机/PC 测试, 加载 Ultralytics YOLO .pt 模型
      - "hbm":         RDK 板端, 加载 HBM 运行时 .hbm 模型 (BPU 推理)
    """

    def __init__(self, model_path, backend="ultralytics", conf=0.25, iou=0.45,
                 imgsz=640, target_classes=None, label_file=None,
                 enable_preprocess=True, preprocess_cfg=None):
        self.backend = backend
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.target_classes = [_normalize_name(t) for t in (target_classes or [])]

        # 通用水下预处理 (可选, 开启可增强红色球检出)
        self.enable_preprocess = enable_preprocess
        if enable_preprocess:
            self.preprocess = load_module_by_path("front_preprocessor", _PREP_PATH).preprocess
            self.preprocess_cfg = build_preprocess_config(preprocess_cfg or {})

        # 按后端初始化模型
        if backend == "ultralytics":
            self._init_ultralytics(model_path)
        elif backend == "hbm":
            self._init_hbm(model_path, label_file)
        else:
            raise ValueError(f"未知的球检测后端: {backend}")

    # ---------- 后端初始化 ----------

    def _init_ultralytics(self, model_path):
        """加载 Ultralytics YOLO .pt 模型 (类别名由模型自带)。"""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics, 请先执行: pip install ultralytics") from exc
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到球检测模型: {model_path}")
        self.model = YOLO(model_path)
        self._backend_detect = self._detect_ultralytics

    def _init_hbm(self, model_path, label_file):
        """加载 HBM 运行时 .hbm 模型 (仅 RDK 板端环境可用)。"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到球检测模型: {model_path}")
        try:
            yolo_mod = load_module_by_path("P2_yolo_detect", _P2_DIR / "yolo_detect.py")
        except ImportError as exc:
            raise RuntimeError(
                "hbm 后端需要 RDK 板端环境 (hbm_runtime); 开发机请改用 ultralytics 后端"
            ) from exc
        config = yolo_mod.YoloDetectConfig(
            model_path=model_path,
            score_thres=self.conf,
            nms_thres=self.iou,
        )
        self.model = yolo_mod.YoloDetect(config)
        # 特征图网格大小按模型实际输入分辨率自动计算 (320x320 -> [40,20,10])
        self.model.cfg.anchor_sizes = [self.model.input_h // s for s in self.model.cfg.strides]
        # 类别名: 优先读取 label 文件, 否则使用类别 id 字符串
        self.labels = load_labels(label_file)
        self._backend_detect = self._detect_hbm

    # ---------- 检测主入口 ----------

    def detect(self, frame):
        """检测一帧中的目标球, 返回统一格式的检测结果列表。"""
        img = frame
        if self.enable_preprocess:
            img = self.preprocess(frame, self.preprocess_cfg)
        return self._backend_detect(img)

    def _detect_ultralytics(self, img):
        """ultralytics 后端: 推理并提取统一格式检测结果。"""
        result = self.model(img, conf=self.conf, iou=self.iou,
                            imgsz=self.imgsz, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if not self._is_target(label):
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(self._make_detection(label, box.conf.item(), x1, y1, x2, y2))
        return detections

    def _detect_hbm(self, img):
        """hbm 后端: 调用 P2 YoloDetect 推理并统一格式。"""
        boxes, scores, cls_ids = self.model.predict(img)
        detections = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            label = self.labels[cls_id] if self.labels and cls_id < len(self.labels) else str(int(cls_id))
            if not self._is_target(label):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(self._make_detection(label, score, x1, y1, x2, y2))
        return detections

    @staticmethod
    def _make_detection(label, score, x1, y1, x2, y2):
        """构造统一格式的单个检测结果字典。"""
        return {
            "label": label,
            "score": float(score),
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
        }

    def _is_target(self, label):
        """按配置的目标类别过滤; 未配置时保留全部。"""
        if not self.target_classes:
            return True
        return _normalize_name(label) in self.target_classes


def draw_ball_result(frame, detections, title="P2 Ball"):
    """在帧上绘制球检测结果 (外接框 + 中心点 + 类别置信度)。

    Args:
        frame:      原始 BGR 图像
        detections: 统一格式检测结果列表
        title:      面板标题文字 (默认 "P2 Ball"; main_bottom 复用时可传 "P4 Ball")
    """
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(vis, f"{det['label']} {det['score']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"{title} | n={len(detections)}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return vis


# ================================================================
# P3 穿门 - 门检测器创建
# ================================================================

def create_door_detector(work_size, overrides=None):
    """创建 P3 门检测器 (红色方形门框, 复用 P5 BoxDetector 内核)。

    门框为四边闭合矩形 (方形, 内部空心), 与 P5 收集框同形; 两者靠比赛阶段
    (TASK_STAGE) 区分, 前视测试阶段不会同时出现.

    Args:
        work_size: (宽, 高), 参考中心点默认取画面中心
        overrides: dict, 对 P3 config.py 的参数覆盖 (未设置 camera_center 时自动对齐)
    """
    w, h = work_size
    cfg = dict(overrides or {})
    cfg.setdefault("camera_center", (w / 2.0, h / 2.0))
    dd_mod = load_module_by_path("P3_door_detector", _P3_DIR / "door_detector.py")
    return dd_mod.DoorDetector(dd_mod.get_config(cfg))


# ================================================================
# 前视摄像头多线程测试器
# ================================================================

class FrontCamTester:
    """前视摄像头双任务并发测试器。

    线程模型:
      主线程       读帧 -> 发布 -> 回收 -> 汇总显示
      P2 工作线程  BallDetector.detect
      P3 工作线程  DoorDetector.detect

    同步方式 (threading.Barrier, 参与方 = 3):
      _barrier_frame  : 主线程发布新帧后与两个工作线程会合, 三线程同时开跑,
                        保证 P2/P3 处理的是同一帧 (帧级同步).
      _barrier_result : 两个工作线程完成检测后与主线程会合, 主线程统一汇总显示.
    """

    def __init__(self, source, ball_detector, door_detector,
                 work_size=(640, 480), show=True, output=None):
        self.source = source
        self.ball = ball_detector
        self.door = door_detector
        self.work_w, self.work_h = work_size
        self.show = show
        self.output = output

        self._stop = threading.Event()
        self._barrier_frame = threading.Barrier(3)
        self._barrier_result = threading.Barrier(3)
        self._result_lock = threading.Lock()
        self._cur_frame = None
        self._results = {"ball": [], "door": None, "err": {}}

    # ---------- 工作线程 ----------

    def _p2_worker(self):
        """P2 球检测工作线程。"""
        while not self._stop.is_set():
            if not self._wait(self._barrier_frame):
                break
            frame = self._cur_frame.copy()
            try:
                self._put_result("ball", self.ball.detect(frame))
            except Exception as exc:
                self._put_error("P2", exc)
            if not self._wait(self._barrier_result):
                break

    def _p3_worker(self):
        """P3 门检测工作线程。"""
        while not self._stop.is_set():
            if not self._wait(self._barrier_frame):
                break
            frame = self._cur_frame.copy()
            try:
                self._put_result("door", self.door.detect(frame))
            except Exception as exc:
                self._put_error("P3", exc)
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
        """发布新帧供两个工作线程处理, 并清空上一帧结果。"""
        self._cur_frame = frame
        with self._result_lock:
            self._results = {"ball": [], "door": None, "err": {}}

    def _collect(self):
        """回收 P2/P3 的检测结果。"""
        with self._result_lock:
            return (self._results.get("ball", []),
                    self._results.get("door"),
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
            threading.Thread(target=self._p2_worker, name="P2-ball", daemon=True),
            threading.Thread(target=self._p3_worker, name="P3-door", daemon=True),
        ]
        for t in threads:
            t.start()

        if self.show:
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_NAME, min(self.work_w * 3, 1440), self.work_h)

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
        # 三面板 (原始 | P2 | P3) 拼接后的宽度
        writer = cv2.VideoWriter(
            self.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (self.work_w * 3, self.work_h))
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

            # 帧交接会合: 等 P2/P3 线程拿到本帧
            if not self._wait(self._barrier_frame):
                break
            # 结果回收会合: 等 P2/P3 完成本帧检测
            if not self._wait(self._barrier_result):
                break

            ball_dets, door_res, err = self._collect()
            if frame_idx % 10 == 0 or not self.show:
                self._print_summary(frame_idx, ball_dets, door_res)

            vis = self._compose(frame, frame_idx, ball_dets, door_res, fps, err)
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
        """将帧缩放到统一工作分辨率, 保证各面板坐标一致。"""
        h, w = frame.shape[:2]
        if (w, h) != (self.work_w, self.work_h):
            frame = cv2.resize(frame, (self.work_w, self.work_h),
                               interpolation=cv2.INTER_LINEAR)
        return frame

    def _print_summary(self, frame_idx, ball_dets, door_res):
        """控制台输出当前帧的 球/门 检测概况 (门含类型/偏移/倾斜角/长宽比)。"""
        if door_res is not None and door_res["detected"]:
            door_txt = (f"door={door_res['door_type']} "
                        f"offset={door_res['offset']:+.1f}px "
                        f"angle={door_res.get('angle', 0.0):+.1f}deg "
                        f"aspect={door_res.get('aspect', 0.0):.2f}")
        elif door_res is None:
            door_txt = "door=ERR"
        else:
            door_txt = "door=None"
        print(f"[Frame {frame_idx:4d}] ball={len(ball_dets)} | {door_txt}")

    def _compose(self, frame, frame_idx, ball_dets, door_res, fps, err):
        """拼接 原始画面 | P2 球检测 | P3 门检测 三面板显示。"""
        original = frame.copy()
        cv2.putText(original, f"FrontCam | frame={frame_idx} | fps={fps:.1f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if err.get("P2"):
            cv2.putText(original, "P2 ERR", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if err.get("P3"):
            cv2.putText(original, "P3 ERR", (10, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        p2_vis = draw_ball_result(frame, ball_dets)

        if door_res is not None:
            p3_vis = self.door.draw_result(frame, door_res, fps=fps)
        else:
            p3_vis = frame.copy()
            cv2.putText(p3_vis, "P3 Door ERROR", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return np.hstack([original, p2_vis, p3_vis])

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
    """解析命令行参数; 未指定的参数使用 main_front_config.py 中的默认值。"""
    parser = argparse.ArgumentParser(
        description="前视摄像头双任务并发测试 (P2 撞球 + P3 穿门)")
    parser.add_argument("--source", type=str, default=None,
                        help="视频文件路径或摄像头编号, 例如 0 / test.mp4")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["ultralytics", "hbm"],
                        help="P2 球检测后端: ultralytics(.pt 开发机) / hbm(.hbm RDK 板端)")
    parser.add_argument("--model", type=str, default=None,
                        help="P2 球检测模型路径 (默认按后端取 P2 撞球 下的默认模型)")
    parser.add_argument("--label-file", type=str, default=None,
                        help="hbm 后端类别名称文件 (每行一个类别名)")
    parser.add_argument("--target", type=str, nargs="+", default=None,
                        help="P2 只保留的目标类别, 例如 --target red; 不指定则保留全部")
    parser.add_argument("--conf", type=float, default=None, help="P2 置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="P2 NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="ultralytics 后端推理分辨率")
    parser.add_argument("--no-preprocess", action="store_true", default=None,
                        help="关闭 P2 的水下预处理")
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
    """合并配置: 以 main_front_config.py 为基准, 命令行参数覆盖同名项。"""
    cfg = get_config()
    # 输入源
    if opt.source is not None:
        cfg["source"] = opt.source
    if opt.width is not None:
        cfg["work_width"] = opt.width
    if opt.height is not None:
        cfg["work_height"] = opt.height
    # P2 球检测
    if opt.backend is not None:
        cfg["p2_backend"] = opt.backend
    if opt.model is not None:
        cfg["p2_model"] = opt.model
    if opt.label_file is not None:
        cfg["p2_label_file"] = opt.label_file
    if opt.target is not None:
        cfg["p2_target_classes"] = opt.target
    if opt.conf is not None:
        cfg["p2_conf"] = opt.conf
    if opt.iou is not None:
        cfg["p2_iou"] = opt.iou
    if opt.imgsz is not None:
        cfg["p2_imgsz"] = opt.imgsz
    if opt.no_preprocess is not None:
        cfg["p2_enable_preprocess"] = not opt.no_preprocess
    # 显示与输出
    if opt.show is not None:
        cfg["show"] = opt.show
    if opt.output is not None:
        cfg["output"] = opt.output
    return cfg


def resolve_model_path(cfg):
    """解析 P2 模型路径: 未配置时按后端取默认模型, 相对路径基于 P2 目录展开。"""
    model = cfg["p2_model"]
    if model is None:
        default_name = (cfg["p2_default_model_pt"] if cfg["p2_backend"] == "ultralytics"
                        else cfg["p2_default_model_hbm"])
        return str(_P2_DIR / default_name)
    if os.path.isabs(model):
        return model
    return str(_P2_DIR / model)


def main():
    opt = parse_args()
    cfg = build_config(opt)
    work_size = (cfg["work_width"], cfg["work_height"])

    # ---- P2 球检测器 ----
    model_path = resolve_model_path(cfg)
    ball_detector = BallDetector(
        model_path=model_path,
        backend=cfg["p2_backend"],
        conf=cfg["p2_conf"],
        iou=cfg["p2_iou"],
        imgsz=cfg["p2_imgsz"],
        target_classes=cfg["p2_target_classes"],
        label_file=cfg["p2_label_file"],
        enable_preprocess=cfg["p2_enable_preprocess"],
        preprocess_cfg=cfg,
    )
    print(f"[P2] 球检测后端: {cfg['p2_backend']}, 模型: {model_path}, "
          f"目标类别: {cfg['p2_target_classes'] or '全部'}")

    # ---- P3 门检测器 ----
    door_detector = create_door_detector(work_size, cfg.get("p3_overrides"))
    print(f"[P3] 门检测器加载完成, camera_center: {work_size[0] / 2:.0f}, {work_size[1] / 2:.0f}")

    # ---- 启动多线程测试 ----
    tester = FrontCamTester(
        source=cfg["source"],
        ball_detector=ball_detector,
        door_detector=door_detector,
        work_size=work_size,
        show=cfg["show"],
        output=cfg["output"],
    )
    tester.run()


if __name__ == "__main__":
    main()
