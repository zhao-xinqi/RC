"""
s100_tasks.py - RDK S100 视觉任务处理管线

职责:
  把 V3.0 各任务检测器 (P1 巡线 / P3 穿门 / P5 放球 / P2P4 球检测) 封装成
  一个"输入一帧 -> 输出各任务结果"的通用管线, 供 s100_receiver.py 调用.
  只复用各任务已有的检测器实现 (不重写算法), 与 main_RDK.py 的组装方式一致.

说明:
  - 各任务内部 0.预处理 默认关闭 (由本管线按任务开关统一执行), 避免重复处理;
  - 球检测 (P2/P4) 使用 hbm 模型 (BPU 推理), 需要板端 hbm_runtime;
    无模型/无运行时环境时自动降级跳过, 不影响其他任务;
  - 路径默认指向 V3.0 完整任务集; 若任务迁移到其他版本, 改 _V3 即可.
"""

# ================================================================
# 库导入
# ================================================================
import importlib.util
from pathlib import Path

import cv2
import numpy as np

# ================================================================
# 路径配置 (默认复用 V3.0 完整任务集)
# ================================================================
_STREAM_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _STREAM_DIR.parent          # 视觉部分/
_V3 = _PROJ_ROOT / 'V3.0'

P1_DIR = _V3 / 'P1 巡线'
P2_DIR = _V3 / 'P2 撞球'
P3_DIR = _V3 / 'P3 穿门'
P5_DIR = _V3 / 'P5 放球'
PREP_PATH = _V3 / '0.预处理' / 'preprocessor.py'

BALL_MODEL = 'best_nashe_320x320_nv12.hbm'   # 相对 P2 撞球目录的模型文件名


# ================================================================
# 通用工具
# ================================================================

def _load_module(module_name: str, path: Path):
    """按文件路径动态加载 Python 模块 (兼容中文/数字目录名).

    各任务文件夹名含数字与中文, 无法用标准 import 语句,
    故使用 importlib.util.spec_from_file_location 按路径加载.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载模块: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 共享 0.预处理 (所有任务线程共用同一函数, 参数各自独立)
_prep_mod = _load_module('preprocessor', PREP_PATH)
preprocess = _prep_mod.preprocess


# ================================================================
# 任务检测器工厂 (复用各任务已有实现)
# ================================================================

def create_line_detector(cfg=None):
    """创建 P1 巡线检测器 (下视).

    cfg: 覆盖配置 dict (可选); 未提供的键使用任务模块自身默认值.
    """
    mod = _load_module('P1_line_detector', P1_DIR / 'line_detector.py')
    return mod.LineDetector(mod.get_config(dict(cfg or {})))


def create_door_detector(cfg=None, work_size=(640, 480)):
    """创建 P3 穿门检测器 (前视).

    cfg: 覆盖配置 dict (可选); 自动补全 camera_center (以图像中心为参考).
    """
    over = dict(cfg or {})
    over.setdefault('camera_center', (work_size[0] / 2.0, work_size[1] / 2.0))
    mod = _load_module('P3_door_detector', P3_DIR / 'door_detector.py')
    return mod.DoorDetector(mod.get_config(over))


def create_box_detector(cfg=None, work_size=(640, 480)):
    """创建 P5 放球检测器 (下视, 识别收集框).

    cfg: 覆盖配置 dict (可选); 自动补全 camera_center.
    """
    over = dict(cfg or {})
    over.setdefault('camera_center', (work_size[0] / 2.0, work_size[1] / 2.0))
    mod = _load_module('P5_box_detector', P5_DIR / 'box_detector.py')
    return mod.BoxDetector(mod.get_config(over))


# ================================================================
# 球检测 (P2 撞球 / P4 抓球, hbm BPU 推理)
# ================================================================

class BallDetector:
    """hbm 球检测适配器 (P2/P4 共用, 复用 P2 撞球/main.py 推理路径).

    对外接口 detect(frame) -> 统一格式结果列表:
        [{'label': str, 'score': float, 'bbox': (x1,y1,x2,y2), 'center': (cx,cy)}, ...]

    初始化需要 RDK 板端 hbm_runtime 与模型文件; 环境不具备时由上层捕获跳过.
    """

    def __init__(self, cfg, model_path=BALL_MODEL, label_file=None):
        p2_mod = _load_module('P2_main', P2_DIR / 'main.py')
        self._p2 = p2_mod
        self._cfg = dict(cfg)
        self._labels = p2_mod.load_labels(label_file)
        # resolve_model_path: 相对路径基于 P2 撞球目录展开
        resolved = p2_mod.resolve_model_path(model_path)
        self._model = p2_mod.create_detector(self._cfg, resolved)

    def detect(self, frame):
        """在单帧上检测目标球, 返回统一格式结果列表."""
        boxes, scores, cls_ids = self._model.predict(frame)
        boxes, scores, cls_ids = self._p2.filter_target_classes(
            boxes, scores, cls_ids, self._cfg, self._labels)
        dets = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            label = self._labels[cls_id] if cls_id < len(self._labels) else str(int(cls_id))
            dets.append({
                'label': label,
                'score': float(score),
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })
        return dets


# ================================================================
# 任务管线: 输入一帧 -> 运行启用的任务 -> 输出结果
# ================================================================

# 各任务 0.预处理 参数 (与 main_RDK.py 的约定一致)
_PREPROCESS_DEFAULT = {
    'enable_resize': False,          # 收到的帧已是工作分辨率, 无需重复缩放
    'enable_color_correct': True,    # 水下颜色校正 (红光补偿)
    'red_boost': 1.2,
    'enable_gaussian': True,
    'gaussian_kernel': 5,
    'enable_clahe': True,
    'clahe_clip': 2.0,
    'clahe_tile': (8, 8),
}


def build_preprocess_config(task_key: str, cfg: dict) -> dict:
    """构建某任务线程的 0.预处理 参数字典.

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


class TaskPipeline:
    """视觉任务管线: 对收到的每帧运行启用的任务检测器.

    用法:
        pipe = TaskPipeline(cfg)
        results = pipe.run(frame)    # {'p1': {...}, 'p3': {...}, ...}
        print(pipe.format_summary(results))

    每个任务独立开关 (cfg[task_key] 为 True/False);
    球检测 (hbm) 在环境不具备时自动降级跳过并打印提示.
    """

    # 任务 -> (内部处理函数, 结果默认 '--')
    _TASKS = {
        'p1': 'line',   # 巡线 (下视)
        'p3': 'door',   # 穿门 (前视)
        'p5': 'box',    # 放球 (下视)
        'p2': 'ball',   # 撞球 (前视, hbm)
        'p4': 'ball',   # 抓球 (下视, hbm, 复用 P2 模型)
    }

    def __init__(self, cfg: dict):
        """
        cfg 关键键:
          work_width / work_height : 工作分辨率 (默认 640x480)
          p1 / p2 / p3 / p4 / p5   : 任务开关 (默认 False)
          p1_enable_preprocess 等  : 各任务 0.预处理 开关 (默认 False)
          p2_model_path            : hbm 模型文件名 (默认 BALL_MODEL)
          p2_label_file            : 类别名文件 (可选)
          p2_score_thres / p2_nms_thres / p2_priority / p2_bpu_cores: 球检测参数
        """
        self.work_size = (cfg.get('work_width', 640), cfg.get('work_height', 480))
        self._cfg = cfg
        self._detectors = {}
        self._errors = {}

        # ---- 构建启用的任务 ----
        for key in ('p1', 'p3', 'p5'):
            if cfg.get(key, False):
                try:
                    self._detectors[key] = self._create_task(key)
                except Exception as exc:
                    self._errors[key] = repr(exc)
                    print(f'[任务] {key.upper()} 初始化失败: {exc!r}, 已跳过')

        # ---- 球检测 (hbm, 环境不具备时跳过) ----
        for key in ('p2', 'p4'):
            if cfg.get(key, False):
                try:
                    self._detectors[key] = self._create_ball_detector(key)
                except Exception as exc:
                    self._errors[key] = repr(exc)
                    print(f'[任务] {key.upper()} 球检测初始化失败: {exc!r}, 已跳过 '
                          f'(需 RDK 板端 hbm_runtime 与模型文件)')

        if not self._detectors:
            print('[任务] 警告: 未启用任何任务 (cfg 中 p1/p2/p3/p4/p5 均未开启)')

    # ---- 构建 ----

    def _create_task(self, key):
        if key == 'p1':
            return create_line_detector(self._cfg)
        if key == 'p3':
            return create_door_detector(self._cfg, self.work_size)
        if key == 'p5':
            return create_box_detector(self._cfg, self.work_size)
        raise ValueError(f'未知任务: {key}')

    def _create_ball_detector(self, key):
        """构建 hbm 球检测器 (P2/P4 各实例化一次, 避免并发调用同一 BPU 模型)."""
        cfg = {
            'score_thres': self._cfg.get('p2_score_thres', 0.25),
            'nms_thres': self._cfg.get('p2_nms_thres', 0.45),
            'priority': self._cfg.get('p2_priority', 0),
            'bpu_cores': list(self._cfg.get('p2_bpu_cores', [0])),
            # P2/P4 各自的目标类别过滤; p4_* 缺省回退 p2_*
            'target_class_ids': list(self._cfg.get(
                f'{key}_target_class_ids', self._cfg.get('p2_target_class_ids', []))),
            'target_class_names': list(self._cfg.get(
                f'{key}_target_class_names', self._cfg.get('p2_target_class_names', []))),
        }
        label_file = self._cfg.get(f'{key}_label_file', self._cfg.get('p2_label_file'))
        model_path = self._cfg.get(f'p2_model_path', BALL_MODEL)
        return BallDetector(cfg, model_path=model_path, label_file=label_file)

    # ---- 运行 ----

    def run(self, frame):
        """对一帧运行所有启用的任务, 返回结果字典 {task_key: result}."""
        results = {}
        for key, det in self._detectors.items():
            try:
                # 可选的 0.预处理 (独立开关, 默认关闭)
                if self._cfg.get(f'{key}_enable_preprocess', False):
                    img = preprocess(frame, build_preprocess_config(key, self._cfg))
                else:
                    img = frame
                results[key] = det.detect(img)
            except Exception as exc:
                self._errors[key] = repr(exc)
                results[key] = None
        return results

    # ---- 结果格式化 ----

    def format_summary(self, results):
        """把任务结果压缩成一行控制台文本 (供周期性打印)."""
        parts = []
        for key in self._detectors:
            r = results.get(key)
            parts.append(f'{key.upper()}: {self._brief(key, r)}')
        return ' | '.join(parts)

    def _brief(self, key, r):
        if r is None:
            return self._errors.get(key, 'ERR')
        if key == 'p1':
            return f'off={r["offset"]:+.1f}' if r.get('detected') else '--'
        if key == 'p3':
            return f'{r.get("door_type", "door")} off={r["offset"]:+.1f}' \
                if r.get('detected') else '--'
        if key == 'p5':
            return f'off={r["offset"]:+.1f}' if r.get('detected') else '--'
        # p2 / p4 球检测: 结果为列表
        return f'n={len(r)}' if r else 'n=0'

    @property
    def task_keys(self):
        return list(self._detectors.keys())


# ================================================================
# 结果绘制 (可选: 供有显示环境时叠加标注)
# ================================================================

def draw_results(frame, results):
    """把任务结果叠加绘制到帧上 (调试/显示用)."""
    vis = frame.copy()
    for key, r in results.items():
        if r is None:
            continue
        if key == 'p1' and r.get('detected'):
            cx, cy = int(r['center'][0]), int(r['center'][1])
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        elif key in ('p3', 'p5') and r.get('detected'):
            x1, y1, x2, y2 = r['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif key in ('p2', 'p4'):
            for det in r:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cx, cy = int(det['center'][0]), int(det['center'][1])
                cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
    return vis
