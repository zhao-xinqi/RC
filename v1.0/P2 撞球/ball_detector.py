"""
P2 撞球 - 撞球检测器

核心功能: 使用前置摄像头 + YOLO 检测目标球,
         输出目标球位置、转向决策与撞击就绪标志, 供控制器转向撞击.

检测流水线:
  原始帧 → (可选预处理) → YOLO 检测 → 目标过滤 → 目标选择 → 转向/撞击决策

输出接口:
  {
      'detected':     bool,      # 是否检测到目标球
      'target':       dict/None, # 选中的目标球信息 (bbox/center/conf/...)
      'balls':        list,      # 所有检测到的目标球
      'offset':       float,     # 目标球中心偏离画面垂直中心线的像素数
      'steer':        str/None,  # 转向决策: 'LEFT' / 'CENTER' / 'RIGHT'
      'distance':     float/None,# 目标球估算距离 (cm)
      'strike_ready': bool,      # 是否足够近且居中, 可执行撞击
      'center':       (x, y),    # 目标球中心坐标
  }
"""

import os
import importlib.util

import cv2
import numpy as np

# ================================================================
# 动态加载模块 (文件夹名含数字/中文, 无法用标准 import)
# ================================================================
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载 0.预处理
_PREP_PATH = os.path.join(_PROJ_ROOT, "0.预处理", "preprocessor.py")
_prep_spec = importlib.util.spec_from_file_location("preprocessor", _PREP_PATH)
_prep_module = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_module)
preprocess = _prep_module.preprocess

# 加载 P2 config
_CONFIG_PATH = os.path.join(_MODULE_DIR, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P2_config", _CONFIG_PATH)
_cfg_module = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_module)
DEFAULT_CONFIG = _cfg_module.DEFAULT_CONFIG
get_config = _cfg_module.get_config

# 加载 YOLO 检测器
_YOLO_PATH = os.path.join(_MODULE_DIR, "yolo_detector.py")
_yolo_spec = importlib.util.spec_from_file_location("P2_yolo_detector", _YOLO_PATH)
_yolo_module = importlib.util.module_from_spec(_yolo_spec)
_yolo_spec.loader.exec_module(_yolo_module)
YoloDetector = _yolo_module.YoloDetector


class BallDetector:
    """
    撞球检测器

    用法:
        detector = BallDetector(config)
        result = detector.detect(frame)

        if result['detected']:
            print(f"转向: {result['steer']}, 偏移: {result['offset']:.1f}px")
    """

    def __init__(self, config=None):
        """
        初始化检测器

        Args:
            config: dict, 配置参数字典.
                    未提供时使用 config.py 中的 DEFAULT_CONFIG.
        """
        import copy

        self.config = copy.deepcopy(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(copy.deepcopy(config))

        # YOLO 目标检测器
        self.yolo = YoloDetector(self.config)

        # 缓存上一帧有效结果, 用于连续帧丢失时的降级策略
        self._last_valid_result = None

    # ================================================================
    # 公开接口
    # ================================================================

    def detect(self, frame):
        """
        检测目标球 (主入口)

        Args:
            frame: BGR 图像 (numpy ndarray)

        Returns:
            dict: {
                'detected':     bool,      是否检测到目标球
                'target':       dict/None, 选中的目标球信息
                'balls':        list,      所有检测到的目标球
                'offset':       float,     目标球中心偏离画面中心的像素数 (正=偏右)
                'steer':        str/None,  转向决策 'LEFT'/'CENTER'/'RIGHT'
                'distance':     float/None 目标球估算距离 (cm)
                'strike_ready': bool,      是否可执行撞击
                'center':       (x, y),    目标球中心坐标
            }
        """
        # ---- 1. (可选) 预处理 ----
        processed = frame
        if self.config.get('enable_preprocess', False):
            processed = preprocess(frame, self.config)

        # ---- 2. YOLO 推理 ----
        detections = self.yolo.detect(processed)

        # ---- 3. 过滤出目标球 ----
        balls = self._filter_targets(detections)

        # ---- 4. 选择目标球 ----
        img_h, img_w = frame.shape[:2]
        target = self._select_target(balls, img_w)

        # ---- 5. 计算转向/撞击决策 ----
        result = self._calc_result(target, balls, img_w)

        # 降级策略: 当前帧未检测到但上一帧有效, 保留上一帧结果供参考
        if not result['detected'] and self._last_valid_result is not None:
            result['last_valid'] = self._last_valid_result

        if result['detected']:
            self._last_valid_result = {
                'offset': result['offset'],
                'steer': result['steer'],
                'distance': result['distance'],
                'center': result['center'],
            }

        return result

    def update_config(self, new_config):
        """
        运行时更新配置 (例如测试工具滑动条回调)

        Args:
            new_config: dict, 需要更新的参数
        """
        self.config.update(new_config)

    def draw_result(self, frame, result):
        """
        在图像上叠加检测结果 (调试/可视化用)

        Args:
            frame:  原始 BGR 图像
            result: detect() 返回的结果字典

        Returns:
            vis: 叠加了标注的图像 (BGR)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx_img = w // 2  # 画面垂直中心线位置

        # ---- 绘制画面中心线 (绿色虚线, 作为参考) ----
        for y in range(0, h, 20):
            cv2.line(vis, (cx_img, y), (cx_img, y + 10), (0, 255, 0), 1)

        # ---- 绘制所有目标球 (青色细框) ----
        for ball in result.get('balls', []):
            x1, y1, x2, y2 = ball['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # ---- 高亮目标球 (红色粗框 + 中心点 + 偏移箭头) ----
        target = result.get('target')
        if target is not None:
            x1, y1, x2, y2 = target['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cx, cy = int(target['center'][0]), int(target['center'][1])
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # 偏移箭头 (从画面中心指向球心)
            cv2.arrowedLine(vis, (cx_img, cy), (cx, cy),
                            (0, 255, 255), 2, tipLength=0.3)

        # ---- 文字信息 ----
        if result.get('detected'):
            cv2.putText(vis, f"Steer: {result['steer']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(vis, f"Offset: {result['offset']:+.1f}px",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            if result.get('distance') is not None:
                cv2.putText(vis, f"Dist: {result['distance']:.0f}cm",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
            if target is not None:
                cv2.putText(vis,
                            f"Conf: {target['conf']:.2f}  {target['class_name']}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
            if result.get('strike_ready'):
                cv2.putText(vis, ">> STRIKE NOW <<",
                            (w // 2 - 90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)
        else:
            cv2.putText(vis, "NO BALL DETECTED",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2)

        return vis

    # ================================================================
    # 内部检测流水线
    # ================================================================

    def _filter_targets(self, detections):
        """
        按配置过滤出目标类别球

        匹配规则: 类别名在 target_class_names 中 或 类别 id 在 target_class_ids 中.
        同时过滤掉框宽过小 (距离过远) 的误检.
        """
        names_set = set(self.config['target_class_names'])
        ids_set = set(self.config['target_class_ids'])
        min_w = self.config['min_detect_width_px']

        balls = []
        for det in detections:
            if det['class_name'] in names_set or det['class_id'] in ids_set:
                if det['width'] >= min_w:
                    balls.append(det)
        return balls

    def _select_target(self, balls, img_w):
        """
        从多个目标球中选择要撞击的目标

        策略 (config['target_strategy']):
          - 'largest': 取框宽最大 (通常最近) 的球, 逐个撞击时用
          - 'center':  取最靠近画面中心的球
        """
        if not balls:
            return None

        strategy = self.config['target_strategy']
        if strategy == 'center':
            return min(balls, key=lambda b: abs(b['center'][0] - img_w / 2.0))
        return max(balls, key=lambda b: b['width'])

    def _calc_result(self, target, balls, img_w):
        """
        根据目标球计算转向/撞击决策

        steer 判定: 球心偏差超过居中区域 (画面宽 × center_zone_ratio) 时需转向
        strike_ready: 居中 且 球框足够宽 (足够近) 时可直接撞击
        """
        image_center_x = img_w / 2.0

        if target is None:
            return {
                'detected': False,
                'target': None,
                'balls': balls,
                'offset': 0.0,
                'steer': None,
                'distance': None,
                'strike_ready': False,
                'center': (0.0, 0.0),
            }

        # ---- 偏移量: 球心到画面中心的水平距离 ----
        offset = target['center'][0] - image_center_x

        # ---- 转向决策 ----
        zone = int(img_w * self.config['center_zone_ratio'])
        if abs(offset) <= zone:
            steer = 'CENTER'
        elif offset > 0:
            steer = 'RIGHT'    # 球在右侧, 需右转
        else:
            steer = 'LEFT'     # 球在左侧, 需左转

        # ---- 距离估算 (针孔成像模型) ----
        distance = None
        if self.config.get('enable_distance', True) and target['width'] > 0:
            focal = self.config['focal_length_px']
            diameter = self.config['ball_real_diameter_cm']
            distance = round(diameter * focal / target['width'], 1)

        # ---- 撞击就绪: 居中 且 足够近 ----
        strike_ready = (
            steer == 'CENTER'
            and target['width'] >= self.config['strike_width_px']
        )

        return {
            'detected': True,
            'target': target,
            'balls': balls,
            'offset': round(float(offset), 2),
            'steer': steer,
            'distance': distance,
            'strike_ready': strike_ready,
            'center': target['center'],
        }
