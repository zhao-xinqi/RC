"""
P1 巡线 - 引导线检测器

核心功能: 从下视摄像头图像中检测池底橙红色引导线,
         计算偏移量和角度供机器人控制器使用.

检测流水线:
  原始帧 → 预处理 → HSV颜色分割 → 形态学处理 → 轮廓提取 → 偏移/角度计算

输出接口:
  {
      'offset':   float,   # 引导线中心偏离画面垂直中心线的像素数
      'angle':    float,   # 引导线与垂直方向的夹角 (度)
      'detected': bool,    # 是否成功检测到引导线
      'center':   (x, y),  # 引导线在画面中的重心坐标
  }
"""

import os
import sys
import importlib.util

import cv2
import numpy as np

# ================================================================
# 动态加载模块 (文件夹名含数字/中文, 无法用标准 import)
# ================================================================
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载 0.预处理 (v4.0 中该目录已更名为 pre_process)
_PREP_PATH = os.path.join(_PROJ_ROOT, "pre_process", "preprocessor.py")
_prep_spec = importlib.util.spec_from_file_location("preprocessor", _PREP_PATH)
_prep_module = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_module)
preprocess = _prep_module.preprocess

# 加载 P1 巡线 config
_CONFIG_PATH = os.path.join(_MODULE_DIR, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P1_config", _CONFIG_PATH)
_cfg_module = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_module)
DEFAULT_CONFIG = _cfg_module.DEFAULT_CONFIG # 默认配置
get_config = _cfg_module.get_config


class LineDetector:
    """
    巡线检测器

    用法:
        detector = LineDetector(config)
        result = detector.detect(frame)

        if result['detected']:
            print(f"偏移: {result['offset']:.1f}px, 角度: {result['angle']:.1f}°")
    """

    def __init__(self, config=None):
        """
        初始化检测器

        Args:
            config: dict, 配置参数字典.
                    未提供时使用 config.py 中的 DEFAULT_CONFIG.
                    支持运行时通过 update_config() 更新.
        """
        import copy

        self.config = copy.deepcopy(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(copy.deepcopy(config))

        # 缓存上一帧的检测结果, 用于连续帧丢失时的降级策略
        self._last_valid_result = None

        # 角度消抖: 滑动窗口历史队列
        self._angle_history = []

    # ================================================================
    # 公开接口
    # ================================================================

    def detect(self, frame):
        """
        检测引导线 (主入口)

        Args:
            frame: BGR 图像 (numpy ndarray)

        Returns:
            dict: {
                'offset':   float,   引导线中心偏离垂直中心线的像素数
                                     正值=偏右, 负值=偏左
                'angle':    float,   引导线与垂直方向的夹角 (度)
                                     正值=线向右偏 (机器人需右转对齐)
                                     负值=线向左偏 (机器人需左转对齐)
                'detected': bool,    是否成功检测到引导线
                'center':   (x, y),  引导线重心在画面中的坐标
            }
        """
        # ---- 1. 预处理 (调用共享模块) ----
        processed = preprocess(frame, self.config)

        # ---- 2. HSV 颜色分割 ----
        mask = self._color_segment(processed)

        # ---- 3. 形态学处理 ----
        mask = self._morphology_process(mask)

        # ---- 4. 提取引导线, 计算偏移量和角度 ----
        result = self._extract_line(mask)

        # 降级策略: 如果当前帧未检测到但上一帧有效, 保留上一帧结果供参考
        if not result['detected'] and self._last_valid_result is not None:
            result['last_valid'] = self._last_valid_result

        if result['detected']:
            # ---- 5. 角度消抖 (死区 + 滑动窗口滤波) ----
            result['angle'] = self._debounce_angle(result['angle'])

            self._last_valid_result = {
                'offset': result['offset'],
                'angle': result['angle'],
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

    def draw_result(self, frame, result, show_mask=False, fps=None):
        """
        在图像上叠加检测结果 (调试/可视化用)

        Args:
            frame:     原始 BGR 图像
            result:    detect() 返回的结果字典
            show_mask: 是否在右侧拼接显示二值掩码
            fps:       当前处理帧率 (float, 可选).
                       提供时在画面右上角叠加 FPS 显示, 不提供则忽略

        Returns:
            vis: 叠加了标注的图像 (BGR)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx_img = w // 2  # 图像垂直中心线位置

        if result['detected']:
            cx, cy = result['center']
            cx, cy = int(cx), int(cy)

            # 绘制垂直中心线 (绿色虚线, 作为参考)
            for y in range(0, h, 20):
                cv2.line(vis, (cx_img, y), (cx_img, y + 10),
                         (0, 255, 0), 1)

            # 绘制引导线重心 (红色圆点)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

            # 绘制偏移量箭头
            cv2.arrowedLine(vis, (cx_img, cy), (cx, cy),
                            (255, 255, 0), 2, tipLength=0.3)

            # 绘制拟合的引导线方向
            length = 80
            angle_rad = np.radians(result['angle'])
            dx = int(length * np.sin(angle_rad))
            dy = int(length * np.cos(angle_rad))
            cv2.line(vis,
                     (cx - dx, cy - dy),
                     (cx + dx, cy + dy),
                     (255, 0, 255), 2)

            # 文字信息
            cv2.putText(vis, f"Offset: {result['offset']:+.1f}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(vis, f"Angle:  {result['angle']:+.1f}deg",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
        else:
            # 未检测到引导线
            cv2.putText(vis, "NO LINE DETECTED",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2)

        # 可选: 帧率显示 (右上角, 避免与左上角文字重叠)
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            (tw, _), _ = cv2.getTextSize(
                fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(vis, fps_text,
                        (w - tw - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        # 可选: 拼接掩码图
        if show_mask and 'mask' in result:
            mask_vis = cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)
            vis = np.hstack([vis, mask_vis])

        return vis

    # ================================================================
    # 内部检测流水线
    # ================================================================

    def _color_segment(self, frame):
        """
        HSV 颜色空间橙红色分割

        橙红色在 HSV 空间中的位置跨 0° 边界:
        - 低H段: [0 ~ 25]  (橙-红色)
        - 高H段: [150 ~ 179] (红-品色, 跨边界)

        两段合并得到完整的橙红色掩码.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 主段 (橙-红色)
        lower1 = np.array(self.config['hsv_lower'])
        upper1 = np.array(self.config['hsv_upper'])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        # 跨边界段 (红色)
        if self.config.get('enable_red_wrap', True):
            lower2 = np.array(self.config['hsv_lower2'])
            upper2 = np.array(self.config['hsv_upper2'])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1

        return mask

    def _morphology_process(self, mask):
        """
        形态学后处理: 开运算去噪 + 闭运算连接断线

        引导线在水下可能因反光、遮挡、水质浑浊而断裂,
        闭运算可弥合小断口, 提高轮廓的连续性.
        """
        open_k = self.config['morph_open_kernel']
        close_k = self.config['morph_close_kernel']

        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k, open_k))
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k, close_k))

        # 开运算: 先腐蚀再膨胀 → 去除细小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # 闭运算: 先膨胀再腐蚀 → 连接邻近断裂
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        return mask

    def _extract_line(self, mask):
        """
        从二值掩码中提取引导线, 计算偏移量和角度

        步骤:
          1. (可选) ROI 限制
          2. 查找轮廓, 取最大轮廓
          3. 面积过滤
          4. 计算重心 → offset
          5. 拟合直线 → angle
        """
        h, w = mask.shape
        image_center_x = w / 2.0

        # ---- ROI 限制: 只关注画面中间区域 ----
        if self.config.get('roi_enabled', False):
            roi = self.config['roi_ratio']
            y1, y2 = int(h * roi[0]), int(h * roi[1])
            x1, x2 = int(w * roi[2]), int(w * roi[3])
            # 将 ROI 外区域置零
            roi_mask = np.zeros_like(mask)
            roi_mask[y1:y2, x1:x2] = 255
            mask = cv2.bitwise_and(mask, roi_mask)

        # ---- 查找轮廓 ----
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._empty_result()

        # ---- 取最大轮廓 ----
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.config['min_contour_area']:
            return self._empty_result()

        # ---- 计算重心 (图像矩) ----
        M = cv2.moments(largest)
        if M['m00'] < 1e-6:
            return self._empty_result()

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # ---- 偏移量: 引导线中心到图像垂直中心线的水平距离 ----
        offset = cx - image_center_x

        # ---- 角度: 拟合直线与垂直方向的夹角 ----
        angle = self._calc_vertical_angle(largest)

        return {
            'offset': round(float(offset), 2),
            'angle': round(float(angle), 2),
            'detected': True,
            'center': (round(float(cx), 2), round(float(cy), 2)),
        }

    def _calc_vertical_angle(self, contour):
        """
        计算轮廓拟合直线的方向与垂直方向 (y轴) 的夹角

        使用自实现的总体最小二乘法 (PCA-TLS) 拟合直线, 替代 cv2.fitLine.
        垂直方向向量为 (0, 1), 拟合直线方向为 (vx, vy).
        atan2(vx, vy) = 偏离垂直方向的角度.

        Args:
            contour: OpenCV 轮廓, 形状为 (N, 1, 2)

        Returns:
            float: 角度 (度), 正值=线向右偏, 负值=线向左偏
        """
        # 提取轮廓点坐标, 形状 (N, 2)
        points = contour.reshape(-1, 2).astype(np.float64)

        # 自实现最小二乘拟合, 返回方向向量 (vx, vy)
        vx, vy, _, _ = self._least_squares_fit_line(points)

        angle_rad = np.arctan2(vx, vy)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    @staticmethod
    def _least_squares_fit_line(points):
        """
        基于 PCA 的总体最小二乘 (Total Least Squares) 直线拟合.

        最小化所有点到直线的正交距离平方和, 等价于求协方差矩阵
        最大特征值对应的特征向量 (数据方差最大方向 = 直线方向).
        直线必经过所有点的质心.

        相比 y = kx + b 的普通最小二乘, 本方法:
          - 同时考虑 x/y 方向误差 (图像坐标均有量化噪声)
          - 垂直直线不会出现斜率发散的数值问题
          - 数学上等价于 cv2.fitLine(..., DIST_L2) 的最小二乘模式

        Args:
            points: (N, 2) numpy 数组, 每行一个点 (x, y)

        Returns:
            (vx, vy, x0, y0): 方向向量 (已统一 vy>0) + 直线上点(质心)
        """
        n = points.shape[0]

        # 点数不足时返回默认垂直方向
        if n < 2:
            return 0.0, 1.0, 0.0, 0.0

        # Step 1: 计算质心
        x_mean = np.mean(points[:, 0])
        y_mean = np.mean(points[:, 1])

        # Step 2: 中心化数据
        x_centered = points[:, 0] - x_mean
        y_centered = points[:, 1] - y_mean

        # Step 3: 构建 2x2 协方差矩阵
        sxx = np.sum(x_centered ** 2)
        syy = np.sum(y_centered ** 2)
        sxy = np.sum(x_centered * y_centered)
        cov = np.array([[sxx, sxy],
                        [sxy, syy]])

        # Step 4: 特征值分解 (eigh 针对实对称矩阵优化, 特征值升序)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 最大特征值对应最后一列特征向量 = 直线方向向量
        vx = float(eigvecs[0, 1])
        vy = float(eigvecs[1, 1])

        # 统一方向: 确保 vy > 0, 避免拟合方向反转导致角度跳变
        if vy < 0:
            vx, vy = -vx, -vy

        return vx, vy, float(x_mean), float(y_mean)

    def _debounce_angle(self, angle):
        """
        角度消抖: 小角度死区 + 滑动窗口滤波.

        解决引导线静止时角度仍持续波动的问题. 两级处理:
          1. 死区: |angle| < angle_deadzone 时直接输出 0.0
             (线接近垂直时噪声占比最大, 微小波动无实际控制意义)
          2. 滑动窗口: 取最近 N 帧角度的中值/均值, 抑制突发跳变
             (中值滤波对异常值鲁棒, 推荐默认使用)

        Args:
            angle: 当前帧原始角度 (度)

        Returns:
            float: 消抖后的角度 (度)
        """
        # 总开关
        if not self.config.get('enable_angle_debounce', True):
            return angle

        # ---- 第一级: 小角度死区 ----
        deadzone = float(self.config.get('angle_deadzone', 3.0))
        if abs(angle) < deadzone:
            angle = 0.0

        # ---- 第二级: 滑动窗口滤波 ----
        window = int(self.config.get('angle_smooth_window', 5))
        if window <= 1:
            return round(angle, 2)

        # 入队并保持窗口大小
        self._angle_history.append(angle)
        if len(self._angle_history) > window:
            self._angle_history = self._angle_history[-window:]

        method = self.config.get('angle_smooth_method', 'median')
        if method == 'mean':
            smoothed = float(np.mean(self._angle_history))
        else:
            # 默认中值滤波: 对突发异常值(如单帧轮廓提取错误)鲁棒
            smoothed = float(np.median(self._angle_history))

        return round(smoothed, 2)

    @staticmethod
    def _empty_result():
        """返回空检测结果"""
        return {
            'offset': 0.0,
            'angle': 0.0,
            'detected': False,
            'center': (0.0, 0.0),
        }
