"""
P2 撞球 - YOLO 目标检测器封装

基于 ultralytics 库 (YOLOv8/v11) 加载模型并推理.
输出统一的检测结果列表, 供 BallDetector 进行目标选择与撞击决策.

依赖: pip install ultralytics
"""

import os
import importlib.util

import numpy as np

# ================================================================
# 动态加载 config (文件夹名含数字/中文, 无法用标准 import)
# ================================================================
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

_CONFIG_PATH = os.path.join(_MODULE_DIR, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P2_config", _CONFIG_PATH)
_cfg_module = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_module)
DEFAULT_CONFIG = _cfg_module.DEFAULT_CONFIG
get_config = _cfg_module.get_config


class YoloDetector:
    """
    YOLO 目标检测器 (ultralytics 封装)

    用法:
        detector = YoloDetector(config)
        detections = detector.detect(frame)
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict, 配置参数字典.
                    未提供时使用 config.py 中的 DEFAULT_CONFIG.
        """
        import copy

        self.config = copy.deepcopy(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(copy.deepcopy(config))

        # 加载模型 (ultralytics)
        self.model = self._load_model()

    # ================================================================
    # 模型加载
    # ================================================================

    def _load_model(self):
        """
        加载 YOLO 模型

        ultralytics 支持:
          - 内置模型名 (如 'yolov8n.pt', 首次使用自动下载)
          - 本地模型文件路径 (.pt 等)

        Returns:
            YOLO 模型对象
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "未安装 ultralytics, 请先执行: pip install ultralytics")

        model_path = self.config['model_path']
        try:
            model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"YOLO 模型加载失败: {model_path}\n"
                f"请确认模型文件存在, 或在 config.py 中修改 model_path.\n"
                f"原始错误: {e}")

        print(f"[*] YOLO 模型已加载: {model_path} (共 {len(model.names)} 个类别)")
        return model

    # ================================================================
    # 推理
    # ================================================================

    def detect(self, frame):
        """
        对一帧图像进行 YOLO 推理

        Args:
            frame: BGR 图像 (numpy ndarray)

        Returns:
            list[dict]: 检测结果列表, 每个元素为:
                {
                    'class_id':   int,               类别 id
                    'class_name': str,               类别名
                    'conf':       float,             置信度
                    'bbox':       [x1, y1, x2, y2],  边界框 (整数像素)
                    'center':     (cx, cy),          边界框中心
                    'width':      float,             框宽
                    'height':     float,             框高
                }
        """
        # ultralytics 需要连续内存数组
        frame = np.ascontiguousarray(frame)

        results = self.model.predict(
            source=frame,
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            imgsz=self.config['imgsz'],
            device=self.config.get('device'),
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        # 单张图像, 取第一个结果
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = results[0].names
        for box in boxes:
            class_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            w = x2 - x1
            h = y2 - y1
            detections.append({
                'class_id': class_id,
                'class_name': names[class_id],   # dict 和 list 均按 id 索引
                'conf': round(conf, 3),
                'bbox': [x1, y1, x2, y2],
                'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                'width': float(w),
                'height': float(h),
            })

        return detections

    # ================================================================
    # 属性
    # ================================================================

    @property
    def class_names(self):
        """所有类别名 (list[str])"""
        names = self.model.names
        if isinstance(names, dict):
            return list(names.values())
        return list(names)
