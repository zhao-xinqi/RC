"""
main_RDK.py - RDK S100 双摄像头多线程主程序

功能:
  在 RDK S100 板端同时打开前视 + 下视两个摄像头, 多线程并发处理并叠加显示:
    前视窗口 : P2 撞球 + P3 穿门 (两个任务线程结果叠加到一个窗口)
    下视窗口 : P1 巡线 + P4 抓球 + P5 放球 (三个任务线程结果叠加到一个窗口)

线程模型 (多线程同时运行):
  采集线程 x2 : 前视(/dev/video0) 与 下视(/dev/video1) 各自独立采集最新帧
  任务线程 x5 : 前视帧交给 P2(撞球)+P3(穿门), 下视帧交给 P1(巡线)+P4(抓球)+P5(放球)
                每个任务线程独立运行"0.预处理"(带独立开关 TASK_PREPROCESS_ENABLE) + 检测
  主线程      : 汇总结果, 前视双任务叠加显示一个窗口, 下视三任务叠加显示一个窗口

生态说明 (RDK S100):
  - P2/P4 球检测使用 hbm 模型 (BPU 推理), 需要板端 hbm_runtime
  - 摄像头为 V4L2 设备 (/dev/video0 /dev/video1), MJPG 编码降低 USB 带宽压力
  - 无桌面环境下 cv2.imshow 不可用, 自动降级为仅数据处理 (控制台仍可查看概况)

运行:
  python main_RDK.py
"""

# ================================================================
# 库导入
# ================================================================
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
_PREP_PATH = _PROJ_ROOT / "0.预处理" / "preprocessor.py"
_P1_DIR = _PROJ_ROOT / "P1 巡线"
_P2_DIR = _PROJ_ROOT / "P2 撞球"
_P3_DIR = _PROJ_ROOT / "P3 穿门"
_P4_DIR = _PROJ_ROOT / "P2 撞球"    # P4 复用 P2, 默认模型位于此目录
_P5_DIR = _PROJ_ROOT / "P5 放球"


# ================================================================
# 基本配置（导入main_RDK_config.py）
# ================================================================
from main_RDK_config import get_config
CFG = get_config()

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


# ================================================================
# 辅助函数
# ================================================================

# 加载共享的 0.预处理 模块 (所有任务线程共用同一函数, 参数各自独立)
_prep_mod = load_module_by_path("preprocessor", _PREP_PATH)
preprocess = _prep_mod.preprocess

# P2 撞球主模块路径 (hbm 球检测推理路径复用 P2 撞球/main.py)
_P2_MAIN_PATH = _P2_DIR / "main.py"

# 各任务线程 0.预处理 独立开关 (True=该线程执行 0.预处理, False=跳过)
# 开关来自 main_RDK_config.py 的 p1..p5_enable_preprocess 键, 各任务独立控制.
TASK_PREPROCESS_ENABLE = {
    'p1': bool(CFG.get('p1_enable_preprocess', True)),   # P1 巡线 (下视)
    'p2': bool(CFG.get('p2_enable_preprocess', True)),   # P2 撞球 (前视)
    'p3': bool(CFG.get('p3_enable_preprocess', True)),   # P3 穿门 (前视)
    'p4': bool(CFG.get('p4_enable_preprocess', True)),   # P4 抓球 (下视, 复用 P2)
    'p5': bool(CFG.get('p5_enable_preprocess', True)),   # P5 放球 (下视)
}

# 0.预处理 各步骤的共享默认值; 若 main_RDK_config.py 中存在
# "任务前缀 + 键名" (如 'p1_enable_clahe') 则优先使用配置值.
_PREPROCESS_DEFAULT = {
    'enable_resize': False,          # 摄像头已直出 640x480, 无需重复缩放
    'enable_color_correct': True,    # 水下颜色校正 (红光补偿)
    'red_boost': 1.2,                # 红色通道增强系数
    'enable_gaussian': True,         # 高斯去噪
    'gaussian_kernel': 5,            # 高斯核大小 (奇数)
    'enable_clahe': True,            # CLAHE 自适应直方图均衡
    'clahe_clip': 2.0,               # CLAHE 对比度限幅
    'clahe_tile': (8, 8),            # CLAHE 网格大小
}

# 前视 / 下视 显示窗口名
FRONT_WIN_NAME = "FrontCam | P2 ball + P3 door"
BOTTOM_WIN_NAME = "BottomCam | P1 line + P4 ball + P5 box"


def build_preprocess_config(cfg, task_key):
    """构建某任务线程的 0.预处理 参数字典 (各线程独立配置).

    Args:
        cfg:      main_RDK_config 配置
        task_key: 'p1'/'p2'/'p3'/'p4'/'p5'

    键名约定: cfg 中键 = task_key + '_' + 参数名, 如 'p1_enable_clahe';
    未配置的键使用 _PREPROCESS_DEFAULT 共享默认值.
    """
    prep_cfg = {
        'image_width': cfg.get('work_width', 640),
        'image_height': cfg.get('work_height', 480),
    }
    prefix = task_key + '_'
    for key, default in _PREPROCESS_DEFAULT.items():
        prep_cfg[key] = cfg.get(prefix + key, default)
    return prep_cfg


def _disable_internal_preprocess(task_cfg):
    """关闭任务模块内部的 0.预处理 步骤.

    原因: 0.预处理 由主程序按线程开关统一执行, 若任务模块内部再执行一次,
         会对同一帧重复处理 (浪费算力且可能过度增强).
    实现: 将 preprocess() 的四个步骤开关全部置 False, 使其退化为仅拷贝原图.
    """
    for key in ('enable_resize', 'enable_color_correct', 'enable_gaussian', 'enable_clahe'):
        task_cfg[key] = False
    return task_cfg


def build_line_detector_config(cfg):
    """从 CFG 的 p1_* 键构建 P1 巡线检测器配置 (内部预处理已关闭)."""
    return _disable_internal_preprocess({
        'image_width': cfg['work_width'],
        'image_height': cfg['work_height'],
        'hsv_lower': cfg['p1_hsv_lower'],
        'hsv_upper': cfg['p1_hsv_upper'],
        'hsv_lower2': cfg['p1_hsv_lower2'],
        'hsv_upper2': cfg['p1_hsv_upper2'],
        'enable_red_wrap': cfg['p1_enable_red_wrap'],
        'morph_open_kernel': cfg['p1_morph_open_kernel'],
        'morph_close_kernel': cfg['p1_morph_close_kernel'],
        'min_contour_area': cfg['p1_min_contour_area'],
        'enable_angle_debounce': cfg['p1_enable_angle_debounce'],
        'angle_deadzone': cfg['p1_angle_deadzone'],
        'angle_smooth_window': cfg['p1_angle_smooth_window'],
    })


def build_door_config(cfg, work_size):
    """从 CFG 的 p3_* 键构建 P3 穿门检测器配置 (内部预处理已关闭)."""
    w, h = work_size
    return _disable_internal_preprocess({
        'image_width': w,
        'image_height': h,
        'camera_center': (w / 2.0, h / 2.0),
        'hsv_lower': cfg['p3_hsv_lower'],
        'hsv_upper': cfg['p3_hsv_upper'],
        'hsv_lower2': cfg['p3_hsv_lower2'],
        'hsv_upper2': cfg['p3_hsv_upper2'],
        'morph_open_kernel': cfg['p3_morph_open_kernel'],
        'morph_close_kernel': cfg['p3_morph_close_kernel'],
        'min_contour_area': cfg['p3_min_contour_area'],
        'min_box_width_ratio': cfg['p3_min_box_width_ratio'],
        'max_box_width_ratio': cfg['p3_max_box_width_ratio'],
        'min_box_height_ratio': cfg['p3_min_box_height_ratio'],
        'max_box_height_ratio': cfg['p3_max_box_height_ratio'],
        'min_aspect_ratio': cfg['p3_min_aspect_ratio'],
        'max_aspect_ratio': cfg['p3_max_aspect_ratio'],
        'edge_fill_threshold': cfg['p3_edge_fill_threshold'],
        'interior_empty_threshold': cfg['p3_interior_empty_threshold'],
        'edge_inset_ratio': cfg['p3_edge_inset_ratio'],
        'interior_shrink_ratio': cfg['p3_interior_shrink_ratio'],
        'aligned_ratio': cfg['p3_aligned_ratio'],
        'door_top_norm_threshold': cfg['p3_door_top_norm_threshold'],
    })


def build_box_config(cfg, work_size):
    """从 CFG 的 p5_* 键构建 P5 放球检测器配置 (内部预处理已关闭)."""
    w, h = work_size
    return _disable_internal_preprocess({
        'image_width': w,
        'image_height': h,
        'camera_center': (w / 2.0, h / 2.0),
        'hsv_lower': cfg['p5_hsv_lower'],
        'hsv_upper': cfg['p5_hsv_upper'],
        'hsv_lower2': cfg['p5_hsv_lower2'],
        'hsv_upper2': cfg['p5_hsv_upper2'],
        'morph_open_kernel': cfg['p5_morph_open_kernel'],
        'morph_close_kernel': cfg['p5_morph_close_kernel'],
        'min_contour_area': cfg['p5_min_contour_area'],
        'min_box_width_ratio': cfg['p5_min_box_width_ratio'],
        'max_box_width_ratio': cfg['p5_max_box_width_ratio'],
        'min_box_height_ratio': cfg['p5_min_box_height_ratio'],
        'max_box_height_ratio': cfg['p5_max_box_height_ratio'],
        'min_aspect_ratio': cfg['p5_min_aspect_ratio'],
        'max_aspect_ratio': cfg['p5_max_aspect_ratio'],
        'edge_fill_threshold': cfg['p5_edge_fill_threshold'],
        'interior_empty_threshold': cfg['p5_interior_empty_threshold'],
        'edge_inset_ratio': cfg['p5_edge_inset_ratio'],
        'interior_shrink_ratio': cfg['p5_interior_shrink_ratio'],
        'aligned_ratio': cfg['p5_aligned_ratio'],
    })


# ---- 检测器工厂: 动态加载各任务模块 (不修改其内部实现) ----

def create_line_detector(cfg):
    """创建 P1 巡线检测器 (动态加载 P1 巡线/line_detector.py)."""
    mod = load_module_by_path("P1_line_detector", _P1_DIR / "line_detector.py")
    return mod.LineDetector(mod.get_config(build_line_detector_config(cfg)))


def create_door_detector(cfg, work_size):
    """创建 P3 穿门检测器 (动态加载 P3 穿门/door_detector.py)."""
    mod = load_module_by_path("P3_door_detector", _P3_DIR / "door_detector.py")
    return mod.DoorDetector(mod.get_config(build_door_config(cfg, work_size)))


def create_box_detector(cfg, work_size):
    """创建 P5 放球检测器 (动态加载 P5 放球/box_detector.py)."""
    mod = load_module_by_path("P5_box_detector", _P5_DIR / "box_detector.py")
    return mod.BoxDetector(mod.get_config(build_box_config(cfg, work_size)))


# ---- P2/P4 球检测 (hbm, 复用 P2 撞球/main.py 的推理路径) ----

class _BallDetector:
    """hbm 球检测适配器 (P2/P4 共用, 复用 P2 撞球/main.py 已完成功能).

    对外接口 detect(frame) -> 统一格式结果列表:
        [{'label': str, 'score': float, 'bbox': (x1,y1,x2,y2), 'center': (cx,cy)}, ...]
    内部由 P2 的 create_detector 完成 anchor_sizes 修正与 BPU 调度设置.
    """

    def __init__(self, p2_mod, p2_cfg, model_path, label_file=None):
        self._p2m = p2_mod
        self._cfg = dict(p2_cfg)
        self._labels = p2_mod.load_labels(label_file)
        self._model = p2_mod.create_detector(self._cfg, model_path)

    def detect(self, frame):
        """在单帧上检测目标球, 返回统一格式结果列表."""
        boxes, scores, cls_ids = self._model.predict(frame)
        boxes, scores, cls_ids = self._p2m.filter_target_classes(
            boxes, scores, cls_ids, self._cfg, self._labels)

        dets = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            label = self._labels[cls_id] if cls_id < len(self._labels) else str(int(cls_id))
            dets.append({
                "label": label,
                "score": float(score),
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })
        return dets


def create_ball_detector(cfg, bpu_cores, task_key='p2'):
    """创建 hbm 球检测器 (P2 与 P4 各实例化一次, 避免两线程并发调用同一 BPU 模型).

    Args:
        cfg:       main_RDK_config 配置
        bpu_cores: 本实例使用的 BPU 核心索引列表
        task_key:  'p2'(前视撞球) 或 'p4'(下视抓球).
                   目标类别/标签配置按任务分别读取 p2_* 与 p4_* 键;
                   p4_* 未配置时回退到 p2_* (P4 复用 P2 模型).

    目标过滤规则 (默认为无限制, 检测全部类别):
        1. 设置了 {task}_target_class_ids → 按类别 id 过滤 (无需 label 文件)
        2. 设置了 {task}_target_class_names → 按类别名称过滤 (需 {task}_label_file)
        3. 设置了 front_target / bottom_target (单类别便捷写法) → 等同一条名称过滤
        4. 以上均未设置 → 保留全部检测结果
    """
    # 惰性加载 P2 主模块 (需要 RDK 板端 hbm_runtime)
    p2_mod = load_module_by_path("P2_main", _P2_MAIN_PATH)

    p2_cfg = p2_mod.get_config()
    p2_cfg['score_thres'] = cfg['p2_score_thres']
    p2_cfg['nms_thres'] = cfg['p2_nms_thres']
    p2_cfg['priority'] = cfg['p2_priority']
    p2_cfg['bpu_cores'] = list(bpu_cores)

    # ---- 目标类别过滤: 按任务读取配置 (p4_* 缺省回退 p2_*, 兼容旧配置) ----
    label_file = cfg.get(f'{task_key}_label_file', cfg.get('p2_label_file'))
    target_ids = list(cfg.get(f'{task_key}_target_class_ids',
                              cfg.get('p2_target_class_ids', [])) or [])
    target_names = list(cfg.get(f'{task_key}_target_class_names',
                                cfg.get('p2_target_class_names', [])) or [])

    # 便捷单类别限制: front_target(前视/P2) / bottom_target(下视/P4), None=不限制
    single = cfg.get('front_target') if task_key == 'p2' else cfg.get('bottom_target')
    if not target_ids and not target_names and single:
        target_names = [str(single)]

    p2_cfg['target_class_ids'] = target_ids
    p2_cfg['target_class_names'] = target_names

    # 按类别名过滤必须依赖 label 文件; 缺失时警告并保留全部, 避免运行时报错
    if target_names and not label_file:
        print(f"[警告] {task_key.upper()} 设置了目标类别 {target_names}, "
              f"但未提供 {task_key}_label_file, 无法按类别名过滤, 将保留全部检测结果")
        p2_cfg['target_class_names'] = []

    model_path = cfg.get('p2_model_path') or 'best_nashe_320x320_nv12.hbm'
    model_path = p2_mod.resolve_model_path(model_path)

    # label 文件相对路径按 P2 撞球目录解析 (通常与模型同目录)
    if label_file and not os.path.isabs(label_file):
        label_file = str(_P2_DIR / label_file)
    return _BallDetector(p2_mod, p2_cfg, model_path, label_file)


# ---- 线程基础设施 ----

class _LatestFrameStore:
    """线程安全的"最新帧"容器: 采集线程写入, 任务线程读取.

    采用"最新帧覆盖"策略 (自动丢弃过旧帧), 各任务线程独立拉取,
    不因某一线程较慢而阻塞其它线程 (符合"每个线程独立"的要求).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def put(self, frame):
        """写入最新帧 (仅保存引用, 读取方不得修改该帧)."""
        with self._lock:
            self._frame = frame

    def get(self):
        """读取最新帧引用; 尚无帧时返回 None."""
        with self._lock:
            return self._frame


class _ResultStore:
    """线程安全的检测结果容器: 任务线程写入, 主线程叠加显示时读取."""

    def __init__(self, task_names):
        self._lock = threading.Lock()
        self._results = {name: None for name in task_names}
        self._errors = {}

    def put(self, task, result):
        """写入某任务最新检测结果, 并清除该任务的历史错误."""
        with self._lock:
            self._results[task] = result
            self._errors.pop(task, None)

    def put_error(self, task, exc):
        """记录某任务线程的异常信息 (不影响其它线程)."""
        with self._lock:
            self._errors[task] = repr(exc)

    def snapshot(self):
        """返回 (结果字典, 错误字典) 副本, 供显示线程叠加."""
        with self._lock:
            return dict(self._results), dict(self._errors)


def _capture_worker(cap, frame_store, stop_event):
    """摄像头采集线程: 循环读帧并写入最新帧容器.

    Args:
        cap:         已打开的 VideoCapture
        frame_store: 最新帧容器 (写入)
        stop_event:  停止事件
    """
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)   # 读取失败 (可能暂时无帧), 稍等重试
            continue
        frame_store.put(frame)


def _task_worker(task_name, frame_store, detect_fn,
                 enable_preprocess, preprocess_cfg, result_store, stop_event):
    """通用任务线程主循环 (每个任务线程独立运行).

    流程: 拉取最新帧 -> (可选)执行 0.预处理 -> 调用任务检测 -> 写入结果.
    单个任务异常仅记录到 result_store, 不影响其它线程.

    Args:
        task_name:         任务名 'p1'/'p2'/'p3'/'p4'/'p5'
        frame_store:       本线程所属摄像头的 最新帧容器
        detect_fn:         检测函数, 输入一帧, 返回该任务结果
        enable_preprocess: 本线程 0.预处理 独立开关
        preprocess_cfg:    本线程 0.预处理 参数字典
        result_store:      结果容器 (写入)
        stop_event:        停止事件
    """
    while not stop_event.is_set():
        frame = frame_store.get()
        if frame is None:
            time.sleep(0.02)   # 摄像头尚未出帧, 稍等再取
            continue

        try:
            if enable_preprocess:
                # 0.预处理 (共享模块), 各线程独立开关与参数
                frame = preprocess(frame, preprocess_cfg)
            result = detect_fn(frame)
            result_store.put(task_name, result)
        except Exception as exc:
            # 记录错误但继续运行, 避免单个任务故障拖垮其它线程
            result_store.put_error(task_name, exc)
            time.sleep(0.1)


# ---- 叠加显示 (各任务结果绘制到同一帧, 单窗口) ----

def _draw_errors(vis, errors, task_keys):
    """在画面左下角绘制各任务线程的错误提示 (红色小字)."""
    if not errors:
        return
    h = vis.shape[0]
    for i, key in enumerate(task_keys):
        if errors.get(key):
            cv2.putText(vis, f"{key.upper()} ERR: {errors[key]}",
                        (10, h - 20 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def compose_front_view(frame, ball_dets, door_result, fps, errors=None):
    """前视叠加视图: P2 球检测 + P3 门检测 叠加到同一帧 (单窗口显示).

    Args:
        frame:       原始前视帧 (BGR)
        ball_dets:   P2 检测结果列表 [{'label','score','bbox','center'}, ...]
        door_result: P3 检测结果字典 (None=尚无结果)
        fps:         显示帧率
        errors:      任务线程错误字典 (可选)
    """
    vis = frame.copy()

    # ---- P2 撞球: 外接框 + 中心点 ----
    for det in (ball_dets or []):
        x1, y1, x2, y2 = (int(v) for v in det['bbox'])
        cx, cy = int(det['center'][0]), int(det['center'][1])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)   # 黄色外接框
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)              # 绿色中心点
        cv2.putText(vis, f"{det['label']} {det['score']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---- P3 穿门: 门框 + 中心 ----
    if door_result is not None and door_result['detected']:
        x1, y1, x2, y2 = door_result['bbox']
        cx, cy = int(door_result['center'][0]), int(door_result['center'][1])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)     # 绿色门框
        cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)

    # ---- 状态文字 (左上角, 分行) ----
    cv2.putText(vis, f"FrontCam | P2+P3 | fps={fps:.1f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis, f"P2 ball | n={len(ball_dets or [])}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    if door_result is not None and door_result['detected']:
        door_txt = f"P3 door | {door_result['door_type']} off={door_result['offset']:+.1f}px"
        door_color = (0, 255, 0)
    elif door_result is None:
        door_txt, door_color = "P3 door | --", (0, 0, 255)
    else:
        door_txt, door_color = "P3 door | None", (0, 0, 255)
    cv2.putText(vis, door_txt, (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, door_color, 2)

    _draw_errors(vis, errors, ('p2', 'p3'))
    return vis


def compose_bottom_view(frame, line_result, ball_dets, box_result, fps, errors=None):
    """下视叠加视图: P1 巡线 + P4 球检测 + P5 放球 叠加到同一帧 (单窗口显示).

    Args:
        frame:       原始下视帧 (BGR)
        line_result: P1 检测结果字典 (None=尚无结果)
        ball_dets:   P4 检测结果列表
        box_result:  P5 检测结果字典 (None=尚无结果)
        fps:         显示帧率
        errors:      任务线程错误字典 (可选)
    """
    vis = frame.copy()
    cx_img = vis.shape[1] // 2

    # ---- P1 巡线: 重心 + 偏移箭头 + 拟合方向 ----
    if line_result is not None and line_result['detected']:
        cx, cy = int(line_result['center'][0]), int(line_result['center'][1])
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)              # 红色重心
        cv2.arrowedLine(vis, (cx_img, cy), (cx, cy), (255, 255, 0), 2, tipLength=0.3)
        ang = np.radians(line_result['angle'])
        length = 80
        dx, dy = int(length * np.sin(ang)), int(length * np.cos(ang))
        cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 0, 255), 2)

    # ---- P4 抓球: 外接框 + 中心点 ----
    for det in (ball_dets or []):
        x1, y1, x2, y2 = (int(v) for v in det['bbox'])
        cx, cy = int(det['center'][0]), int(det['center'][1])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(vis, f"{det['label']} {det['score']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---- P5 放球: 框口矩形 + 中心 ----
    if box_result is not None and box_result['detected']:
        x1, y1, x2, y2 = box_result['bbox']
        cx, cy = int(box_result['center'][0]), int(box_result['center'][1])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)

    # ---- 状态文字 (左上角, 分行) ----
    cv2.putText(vis, f"BottomCam | P1+P4+P5 | fps={fps:.1f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if line_result is not None and line_result['detected']:
        line_txt = f"P1 line | off={line_result['offset']:+.1f}px ang={line_result['angle']:+.1f}deg"
        line_color = (255, 255, 0)
    elif line_result is None:
        line_txt, line_color = "P1 line | --", (0, 0, 255)
    else:
        line_txt, line_color = "P1 line | None", (0, 0, 255)
    cv2.putText(vis, line_txt, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
    cv2.putText(vis, f"P4 ball | n={len(ball_dets or [])}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    if box_result is not None and box_result['detected']:
        box_txt = f"P5 box | off={box_result['offset']:+.1f}px"
        box_color = (0, 255, 0)
    elif box_result is None:
        box_txt, box_color = "P5 box | --", (0, 0, 255)
    else:
        box_txt, box_color = "P5 box | None", (0, 0, 255)
    cv2.putText(vis, box_txt, (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    _draw_errors(vis, errors, ('p1', 'p4', 'p5'))
    return vis


# ---- 摄像头 / 汇总辅助 ----

def _open_camera(device, width, height):
    """打开 RDK S100 上的 V4L2 摄像头.

    device 可为摄像头编号 (int, 0=前视 /dev/video0) 或设备节点路径 (str, '/dev/video0');
    设置 640x480 并优先使用 MJPG 编码, 降低 USB 带宽压力.
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device} (请检查设备节点与权限)")
    return cap


def _print_summary(frame_idx, results):
    """控制台打印当前各任务检测概况 (调试 / RDK 无显示环境下使用)."""
    p1 = results.get('p1')
    p1_txt = f"line off={p1['offset']:+.1f}" if p1 is not None and p1['detected'] else "line=--"
    p3 = results.get('p3')
    p3_txt = f"door={p3['door_type']}" if p3 is not None and p3['detected'] else "door=--"
    p5 = results.get('p5')
    p5_txt = f"box off={p5['offset']:+.1f}" if p5 is not None and p5['detected'] else "box=--"
    print(f"[MAIN] frame={frame_idx:4d} | P2:{len(results.get('p2') or [])}球 {p3_txt} "
          f"| P4:{len(results.get('p4') or [])}球 {p1_txt} {p5_txt}")


# ================================================================
# 主程序
# ================================================================
def main():
    """
    RDK S100 双摄像头多线程主程序.

    线程模型 (多线程同时运行):
      采集线程 x2 : 前视(/dev/video0) 与 下视(/dev/video1) 各自独立采集最新帧
      任务线程 x5 : 前视帧交给 P2(撞球)+P3(穿门), 下视帧交给 P1(巡线)+P4(抓球)+P5(放球)
                    每个任务线程独立运行"0.预处理"(带独立开关) + 检测
      主线程      : 汇总结果, 前视双任务叠加一个窗口, 下视三任务叠加一个窗口

    生态说明 (RDK S100):
      - P2/P4 球检测使用 hbm 模型 (BPU 推理), 需要板端 hbm_runtime
      - 摄像头为 V4L2 设备 (/dev/video0 /dev/video1), MJPG 编码降低 USB 带宽压力
      - 无桌面环境下 cv2.imshow 不可用, 自动降级为仅数据处理 (控制台仍可查看概况)
    """
    work_size = (CFG['work_width'], CFG['work_height'])

    # ================================================================
    # 1. 整体初始化: 打开双摄像头 (摄像头0=前视, 摄像头1=下视)
    #    摄像头编号/设备节点取自 main_RDK_config.py 的 front_source/bottom_source
    # ================================================================
    front_dev = CFG['front_source']
    bottom_dev = CFG['bottom_source']
    front_name = f"/dev/video{front_dev}" if isinstance(front_dev, int) else str(front_dev)
    bottom_name = f"/dev/video{bottom_dev}" if isinstance(bottom_dev, int) else str(bottom_dev)
    print(f"[MAIN] 打开前视摄像头 ({front_name} = 摄像头0) ...")
    cap_front = _open_camera(front_dev, *work_size)
    print(f"[MAIN] 打开下视摄像头 ({bottom_name} = 摄像头1) ...")
    cap_bottom = _open_camera(bottom_dev, *work_size)
    print(f"[MAIN] 双摄像头初始化完成: 前视={front_name}, 下视={bottom_name}")

    # ================================================================
    # 2. 构建各任务检测器 (复用各任务已完成的功能代码, 不重写)
    # ================================================================
    # P2/P4 球检测: hbm 模型, 前视/下视各一个实例 (分配不同 BPU 核心避免争抢)
    try:
        # BPU 核心: 默认均用核心 0 (RDK S100 至少 1 核);
        # 若板端为多核, 可在配置中新增 p2_bpu_cores_front / p2_bpu_cores_bottom 分别指定,
        # 让前视/下视球检测各占一核, 避免相互争抢推理资源
        # 目标过滤按任务分别读取配置: P2 用 front_target/p2_*, P4 用 bottom_target/p4_*
        front_ball = create_ball_detector(CFG, CFG.get('p2_bpu_cores_front', [0]),
                                          task_key='p2')
        bottom_ball = create_ball_detector(CFG, CFG.get('p2_bpu_cores_bottom', [0]),
                                           task_key='p4')
    except Exception as exc:
        print(f"[错误] P2/P4 球检测器初始化失败: {exc!r}")
        print("[提示] 请确认在 RDK S100 板端运行 (需安装 hbm_runtime), 并检查模型文件。")
        raise
    # P3 穿门 (前视)
    door_detector = create_door_detector(CFG, work_size)
    # P1 巡线 (下视)
    line_detector = create_line_detector(CFG)
    # P5 放球 (下视)
    box_detector = create_box_detector(CFG, work_size)

    # ================================================================
    # 3. 线程共享存储
    # ================================================================
    front_store = _LatestFrameStore()    # 前视最新帧
    bottom_store = _LatestFrameStore()   # 下视最新帧
    result_store = _ResultStore(['p1', 'p2', 'p3', 'p4', 'p5'])
    stop_event = threading.Event()

    # 各任务线程的 0.预处理 独立开关 + 参数字典
    task_preprocess = {
        'p1': (TASK_PREPROCESS_ENABLE['p1'], build_preprocess_config(CFG, 'p1')),
        'p2': (TASK_PREPROCESS_ENABLE['p2'], build_preprocess_config(CFG, 'p2')),
        'p3': (TASK_PREPROCESS_ENABLE['p3'], build_preprocess_config(CFG, 'p3')),
        'p4': (TASK_PREPROCESS_ENABLE['p4'], build_preprocess_config(CFG, 'p4')),
        'p5': (TASK_PREPROCESS_ENABLE['p5'], build_preprocess_config(CFG, 'p5')),
    }

    # ================================================================
    # 4. 启动采集线程 + 任务线程 (多线程同时运行, 各线程独立)
    # ================================================================
    threads = [
        threading.Thread(target=_capture_worker,
                         args=(cap_front, front_store, stop_event),
                         name='cap-front', daemon=True),
        threading.Thread(target=_capture_worker,
                         args=(cap_bottom, bottom_store, stop_event),
                         name='cap-bottom', daemon=True),
    ]

    # 前视任务: P2 撞球 + P3 穿门
    p2_enable, p2_prep = task_preprocess['p2']
    threads.append(threading.Thread(
        target=_task_worker,
        args=('p2', front_store, front_ball.detect, p2_enable, p2_prep,
              result_store, stop_event),
        name='p2-ball', daemon=True))
    p3_enable, p3_prep = task_preprocess['p3']
    threads.append(threading.Thread(
        target=_task_worker,
        args=('p3', front_store, door_detector.detect, p3_enable, p3_prep,
              result_store, stop_event),
        name='p3-door', daemon=True))

    # 下视任务: P1 巡线 + P4 抓球 + P5 放球
    p1_enable, p1_prep = task_preprocess['p1']
    threads.append(threading.Thread(
        target=_task_worker,
        args=('p1', bottom_store, line_detector.detect, p1_enable, p1_prep,
              result_store, stop_event),
        name='p1-line', daemon=True))
    p4_enable, p4_prep = task_preprocess['p4']
    threads.append(threading.Thread(
        target=_task_worker,
        args=('p4', bottom_store, bottom_ball.detect, p4_enable, p4_prep,
              result_store, stop_event),
        name='p4-ball', daemon=True))
    p5_enable, p5_prep = task_preprocess['p5']
    threads.append(threading.Thread(
        target=_task_worker,
        args=('p5', bottom_store, box_detector.detect, p5_enable, p5_prep,
              result_store, stop_event),
        name='p5-box', daemon=True))

    for t in threads:
        t.start()
    print(f"[MAIN] 已启动 2 个采集线程 + 5 个任务线程 "
          f"(0.预处理开关: P1={'开' if p1_enable else '关'} "
          f"P2={'开' if p2_enable else '关'} P3={'开' if p3_enable else '关'} "
          f"P4={'开' if p4_enable else '关'} P5={'开' if p5_enable else '关'})")

    # ================================================================
    # 5. 主循环: 前视双任务叠加一个窗口, 下视三任务叠加一个窗口
    # ================================================================
    show = bool(CFG.get('show', True))
    if show:
        try:
            cv2.namedWindow(FRONT_WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.namedWindow(BOTTOM_WIN_NAME, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("[警告] 当前环境无法创建显示窗口 (RDK 无桌面时属正常), 仅保留数据处理")
            show = False

    fps, frame_idx = 0.0, 0
    last_err_txt = ''
    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            results, errors = result_store.snapshot()
            front_frame = front_store.get()
            bottom_frame = bottom_store.get()

            # 前视: P2 + P3 叠加
            if front_frame is not None:
                front_vis = compose_front_view(front_frame, results['p2'], results['p3'],
                                               fps, errors)
            # 下视: P1 + P4 + P5 叠加
            if bottom_frame is not None:
                bottom_vis = compose_bottom_view(bottom_frame, results['p1'],
                                                 results['p4'], results['p5'],
                                                 fps, errors)

            if show:
                try:
                    if front_frame is not None:
                        cv2.imshow(FRONT_WIN_NAME, front_vis)
                    if bottom_frame is not None:
                        cv2.imshow(BOTTOM_WIN_NAME, bottom_vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[MAIN] 用户按下 q, 退出")
                        break
                except cv2.error:
                    pass   # 无显示环境忽略 GUI 异常

            # 任务线程异常只在变化时打印, 避免刷屏
            err_txt = ', '.join(f'{k}:{v}' for k, v in errors.items())
            if err_txt and err_txt != last_err_txt:
                print(f"[WARN] 任务线程异常: {err_txt}")
                last_err_txt = err_txt

            # 周期性控制台概况 (RDK 无显示时也能排查)
            if frame_idx % 30 == 0:
                _print_summary(frame_idx, results)

            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            frame_idx += 1

            if not show:
                time.sleep(0.02)   # 无窗口显示时不空转
    except KeyboardInterrupt:
        print("[MAIN] 收到 Ctrl+C, 退出")
    finally:
        # ================================================================
        # 6. 停止所有线程并释放摄像头资源
        # ================================================================
        stop_event.set()
        for t in threads:
            t.join(timeout=3)
        cap_front.release()
        cap_bottom.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("[MAIN] 资源已释放, 程序退出")


if __name__ == "__main__":
    main()
