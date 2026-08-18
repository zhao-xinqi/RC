"""
P3 穿门 - 门检测器

核心功能: 从前视摄像头图像中检测红色 "Π 形" 门 (两竖杆 + 顶部横梁, 内部空心),
         输出门洞中心、门类型 (高门/矮门) 与对齐状态, 供控制器对准穿门.

检测流水线:
  原始帧 → 预处理 → HSV红色分割 → 形态学处理 → 轮廓筛选 → Π形验证 → 门中心/类型/对齐

输出接口:
  {
      'detected':     bool,        是否检测到门
      'door_type':    str/None,    门类型 'high' / 'low'
      'center':       (x, y),      门洞中心像素坐标
      'center_norm':  (dx, dy),    相对画面中心归一化偏移 [-1,1]
      'bbox':         (x1,y1,x2,y2), 门框外接矩形 (像素)
      'offset':       float,       门洞中心偏离画面垂直中心线的像素数 (正=偏右)
      'aligned':      bool,        门洞中心是否与画面中心对齐
      'mask':         ndarray/None 红/Π 二值掩码 (调试用)
  }

与红色物体判别表 (§6) 的区分逻辑:
  - 门 (Π形):   顶部横梁+左右竖杆 有红, 底部无横杆, 内部空心
  - 收集框:     四边闭合矩形 → 底部条红色占比高 → 被底部判据排除
  - 圆环:       空心圆, 宽高比≈1 → 被宽高比/竖杆判据排除
  - 撞球:       实心 → 内部红色占比高 → 被内部判据排除
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

# 加载 P3 config
_CONFIG_PATH = os.path.join(_MODULE_DIR, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P3_config", _CONFIG_PATH)
_cfg_module = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_module)
DEFAULT_CONFIG = _cfg_module.DEFAULT_CONFIG
get_config = _cfg_module.get_config


class DoorDetector:
    """
    门检测器 (红色 Π 形门)

    用法:
        detector = DoorDetector(config)
        result = detector.detect(frame)

        if result['detected']:
            print(f"门类型: {result['door_type']}, "
                  f"偏移: {result['offset']:.1f}px, 对齐: {result['aligned']}")
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

        # 缓存上一帧有效结果, 用于连续帧丢失时的降级策略
        self._last_valid_result = None

    # ================================================================
    # 公开接口
    # ================================================================

    def detect(self, frame):
        """
        检测门 (主入口)

        Args:
            frame: BGR 图像 (numpy ndarray)

        Returns:
            dict: 见模块文档 "输出接口".
        """
        # ---- 1. 预处理 (调用共享模块) ----
        processed = preprocess(frame, self.config)

        # ---- 2. HSV 红色分割 ----
        mask = self._color_segment(processed)

        # ---- 3. 形态学处理 ----
        mask = self._morphology_process(mask)

        # ---- 4. 提取门 (Π形验证) ----
        result = self._extract_door(mask, frame.shape[:2])

        # 降级策略: 当前帧未检测到但上一帧有效, 保留上一帧结果供参考
        if not result['detected'] and self._last_valid_result is not None:
            result['last_valid'] = self._last_valid_result

        if result['detected']:
            self._last_valid_result = {
                'door_type': result['door_type'],
                'center': result['center'],
                'offset': result['offset'],
                'aligned': result['aligned'],
            }

        return result

    def update_config(self, new_config):
        """
        运行时更新配置 (例如测试工具滑动条回调)

        Args:
            new_config: dict, 需要更新的参数
        """
        self.config.update(new_config)

    def draw_result(self, frame, result, show_mask=False):
        """
        在图像上叠加检测结果 (调试/可视化用)

        Args:
            frame:     原始 BGR 图像
            result:    detect() 返回的结果字典
            show_mask: 是否在右侧拼接显示二值掩码

        Returns:
            vis: 叠加了标注的图像 (BGR)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx_img = w // 2  # 画面垂直中心线位置

        if result['detected']:
            x1, y1, x2, y2 = result['bbox']

            # 绘制画面垂直中心线 (绿色虚线, 参考)
            for y in range(0, h, 20):
                cv2.line(vis, (cx_img, y), (cx_img, y + 10), (0, 255, 0), 1)

            # 绘制门框 (红色粗框)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 绘制门洞中心 (青色圆点)
            cx, cy = int(result['center'][0]), int(result['center'][1])
            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

            # 绘制偏移箭头 (画面中心 → 门洞中心)
            cv2.arrowedLine(vis, (cx_img, cy), (cx, cy),
                            (255, 255, 0), 2, tipLength=0.3)

            # 文字信息
            cv2.putText(vis, f"Door: {result['door_type']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(vis, f"Offset: {result['offset']:+.1f}px",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(vis, f"Aligned: {result['aligned']}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
        else:
            # 未检测到门
            cv2.putText(vis, "NO DOOR DETECTED",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2)

        # 可选: 拼接掩码图
        if show_mask and result.get('mask') is not None:
            mask_vis = cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)
            vis = np.hstack([vis, mask_vis])

        return vis

    # ================================================================
    # 内部检测流水线
    # ================================================================

    def _color_segment(self, frame):
        """
        HSV 色彩空间红色分割

        红色在 HSV 中跨 0° 边界, 分两段:
        - 主段:   [0 ~ hsv_upper]        (橙-红)
        - 跨边界: [hsv_lower2 ~ 179]     (红-品)
        两段合并得到完整红色掩码.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv,
                            np.array(self.config['hsv_lower']),
                            np.array(self.config['hsv_upper']))

        if self.config.get('enable_red_wrap', True):
            mask2 = cv2.inRange(hsv,
                                np.array(self.config['hsv_lower2']),
                                np.array(self.config['hsv_upper2']))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1

        return mask

    def _morphology_process(self, mask):
        """
        形态学后处理: 开运算去噪 + 闭运算连接断线

        门框在水下可能因反光/遮挡断裂, 闭运算弥合小断口.
        """
        open_k = self.config['morph_open_kernel']
        close_k = self.config['morph_close_kernel']

        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k, open_k))
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k, close_k))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        return mask

    def _extract_door(self, mask, frame_shape):
        """
        从二值掩码中提取 Π 形门

        策略: 按面积降序遍历轮廓, 逐个做 尺寸过滤 + Π形验证,
              门不一定是画面中最大的红色区域 (可能同时存在其他红色物),
              只要有一个轮廓通过验证即判定为门.
        """
        img_h, img_w = frame_shape
        image_center_x = img_w / 2.0

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)  # 部分版本返回 tuple, 无 sort 方法
        if not contours:
            return self._empty_result()

        contours.sort(key=cv2.contourArea, reverse=True)

        for contour in contours:
            # 面积过小直接跳过 (后续轮廓更小, 可提前结束)
            if cv2.contourArea(contour) < self.config['min_contour_area']:
                break

            x, y, w, h = cv2.boundingRect(contour)

            # 尺寸/宽高比过滤
            if not self._size_ok(w, h, img_w, img_h):
                continue

            # Π 形验证
            if not self._validate_pi_shape(mask, x, y, w, h):
                continue

            # 通过验证 → 计算输出
            return self._build_result(x, y, w, h, img_w, img_h,
                                      image_center_x, mask)

        return self._empty_result()

    def _size_ok(self, w, h, img_w, img_h):
        """
        尺寸与宽高比过滤

        门 70×50cm, 在画面中的外接矩形应处于合理比例范围内.
        """
        cfg = self.config
        w_ratio = w / img_w
        h_ratio = h / img_h
        aspect = h / w

        if not (cfg['min_door_width_ratio'] <= w_ratio <= cfg['max_door_width_ratio']):
            return False
        if not (cfg['min_door_height_ratio'] <= h_ratio <= cfg['max_door_height_ratio']):
            return False
        if not (cfg['min_aspect_ratio'] <= aspect <= cfg['max_aspect_ratio']):
            return False
        return True

    def _validate_pi_shape(self, mask, x, y, w, h):
        """
        Π 形结构验证

        在外接矩形内采样 5 个区域, 统计红色占比:
          - 顶部横梁 (top_strip):     红色占比高 → 有横梁
          - 左竖杆 (left_strip):      红色占比高 → 有左杆
          - 右竖杆 (right_strip):     红色占比高 → 有右杆
          - 底部条 (bottom_strip):    红色占比低 → 无底杆 (区分"收集框"四边闭合矩形)
          - 内部区 (interior):        红色占比低 → 内部空心 (区分"实心球")

        Returns:
            bool: True=符合 Π 形特征
        """
        cfg = self.config
        region = mask[y:y + h, x:x + w]
        region_h, region_w = region.shape
        if region_w < 3 or region_h < 3:
            return False

        # 采样条厚度 (按区域尺寸比例)
        strip_h = max(2, int(region_h * cfg['strip_ratio']))
        strip_w = max(2, int(region_w * cfg['strip_ratio']))

        # 顶部横梁: 顶部整条
        top_strip = region[0:strip_h, :]

        # 左右竖杆: 取中段竖条 (避开顶部横梁区域)
        mid_y1, mid_y2 = int(region_h * 0.3), int(region_h * 0.8)
        left_strip = region[mid_y1:mid_y2, 0:strip_w]
        right_strip = region[mid_y1:mid_y2, region_w - strip_w:]

        # 底部条: 底部整条 (门无底杆, 收集框有)
        bottom_strip = region[region_h - strip_h:, :]

        # 内部空心区: 两杆之间、横梁下方
        interior = region[int(region_h * 0.35):int(region_h * 0.8),
                          int(region_w * 0.2):int(region_w * 0.8)]

        def _red_ratio(roi):
            """区域内红色像素占比"""
            if roi.size == 0:
                return 0.0
            return float(cv2.countNonZero(roi)) / roi.size

        beam_r = _red_ratio(top_strip)
        left_r = _red_ratio(left_strip)
        right_r = _red_ratio(right_strip)
        bottom_r = _red_ratio(bottom_strip)
        interior_r = _red_ratio(interior)

        return (beam_r >= cfg['beam_fill_threshold']
                and left_r >= cfg['pillar_fill_threshold']
                and right_r >= cfg['pillar_fill_threshold']
                and bottom_r <= cfg['bottom_empty_threshold']
                and interior_r <= cfg['interior_empty_threshold'])

    def _build_result(self, x, y, w, h, img_w, img_h, image_center_x, mask):
        """
        计算门洞中心、偏移、归一化偏移、对齐状态与门类型

        门洞中心取外接矩形中心 (Π 形左右对称, 中心即门洞中心).
        """
        cx = x + w / 2.0
        cy = y + h / 2.0

        # 水平偏移: 门洞中心到画面垂直中心线的像素数
        offset = cx - image_center_x

        # 归一化偏移 [-1,1] (正=偏右, 负=偏左; 控制端可直接做偏差)
        center_norm_x = offset / image_center_x if image_center_x else 0.0
        center_norm_y = ((cy - img_h / 2.0) / (img_h / 2.0)) if img_h else 0.0

        # 对齐判定
        aligned = abs(center_norm_x) <= self.config['aligned_ratio']

        # 门类型: 由门顶部 (横梁) 在画面中的归一化位置判定
        door_top_norm = y / img_h if img_h else 1.0
        if door_top_norm < self.config['door_top_norm_threshold']:
            door_type = 'high'   # 横梁偏上 → 高门
        else:
            door_type = 'low'    # 横梁偏下 → 矮门

        return {
            'detected': True,
            'door_type': door_type,
            'center': (round(float(cx), 2), round(float(cy), 2)),
            'center_norm': (round(float(center_norm_x), 3),
                            round(float(center_norm_y), 3)),
            'bbox': (int(x), int(y), int(x + w), int(y + h)),
            'offset': round(float(offset), 2),
            'aligned': bool(aligned),
            'mask': mask,
        }

    @staticmethod
    def _empty_result():
        """返回空检测结果"""
        return {
            'detected': False,
            'door_type': None,
            'center': (0.0, 0.0),
            'center_norm': (0.0, 0.0),
            'bbox': (0, 0, 0, 0),
            'offset': 0.0,
            'aligned': False,
            'mask': None,
        }
