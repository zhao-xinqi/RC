"""
P5 放球 - 红色篮筐 (收集框) 检测器

核心功能: 使用前视摄像头 + HSV 识别红色 "口"形篮筐 (四边闭合矩形边框,
         内部空心), 输出框口中心、框的倾斜朝向与对齐状态, 供控制器对准放球.

检测流水线:
  原始帧 → 预处理 → HSV红色分割 → 形态学处理 → 轮廓筛选 → 闭合矩形验证 → 框口中心/朝向/对齐

输出接口:
  {
      'detected':     bool,         是否检测到篮筐
      'box_type':     str,          篮筐类型 'box' (预留)
      'center':       (x, y),       框口中心像素坐标
      'center_norm':  (dx, dy),     相对画面中心归一化偏移 [-1,1]
      'bbox':         (x1,y1,x2,y2), 篮筐外接矩形 (像素)
      'offset':       float,        框口中心偏离画面垂直中心线的像素数 (正=偏右)
      'angle':        float,        框在画面中的倾斜角 (度, 长轴偏离竖直方向)
      'aspect':       float,        框旋转外接矩形宽高比 (≥1, 朝向参考)
      'aligned':      bool,         框口中心是否与画面中心对齐
      'mask':         ndarray/None  红色二值掩码 (调试用)
  }

与红色物体判别表 (§6) 的区分逻辑:
  - 篮筐/收集框: 四边闭合矩形 → 顶/底/左/右 红色占比均高, 内部空心
  - 门 (Π形):   无底杆 → 底部条红色占比低 → 被底部判据排除
  - 圆环:       空心圆, 边条填充率低/底部不闭合 → 被边条判据排除
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

# 加载 P5 config
_CONFIG_PATH = os.path.join(_MODULE_DIR, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P5_config", _CONFIG_PATH)
_cfg_module = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_module)
DEFAULT_CONFIG = _cfg_module.DEFAULT_CONFIG
get_config = _cfg_module.get_config


class BoxDetector:
    """
    红色篮筐 (收集框) 检测器

    用法:
        detector = BoxDetector(config)
        result = detector.detect(frame)

        if result['detected']:
            print(f"框口偏移: {result['offset']:.1f}px, "
                  f"倾斜: {result['angle']:.1f}°, 对齐: {result['aligned']}")
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
        self._normalize_center_point()

    def _normalize_center_point(self):
        """确保 camera_center 是一个 (x, y) 二元组。"""
        center = self.config.get('camera_center', self.config.get('center_point', (320, 240)))
        if center is None:
            center = (320, 240)
        if isinstance(center, (list, tuple)) and len(center) == 2:
            self.config['camera_center'] = (float(center[0]), float(center[1]))
            self.config['center_point'] = self.config['camera_center']
        else:
            self.config['camera_center'] = (320.0, 240.0)
            self.config['center_point'] = self.config['camera_center']

    def _resolve_camera_center(self, frame_w, frame_h):
        """从配置中解析参考中心点。"""
        cx, cy = self.config.get('camera_center', self.config.get('center_point', (frame_w / 2.0, frame_h / 2.0)))
        return float(cx), float(cy)

    def compute_center_offset(self, center_xy, center_point=None):
        """计算框口中心相对摄像头中心点的偏移。

        约定：
            dx = 框中心x - 摄像头中心x   (右为正)
            dy = 摄像头中心y - 框中心y  (上为正)
        """
        if center_point is None:
            center_point = self.config.get('camera_center', (320, 240))
        cx, cy = center_xy
        ox, oy = center_point
        dx = float(cx) - float(ox)
        dy = float(oy) - float(cy)
        return dx, dy

    def detect_with_center_offset(self, frame):
        """检测篮筐并返回相对摄像头中心点的偏移信息。"""
        result = self.detect(frame)
        if not result['detected']:
            result['center_offset'] = None
            result['camera_center'] = self._resolve_camera_center(frame.shape[1], frame.shape[0])
            return result

        cx, cy = result['center']
        camera_center = self._resolve_camera_center(frame.shape[1], frame.shape[0])
        dx, dy = self.compute_center_offset((cx, cy), camera_center)
        result['camera_center'] = camera_center
        result['center_offset'] = (round(float(dx), 2), round(float(dy), 2))
        return result

    # ================================================================
    # 公开接口
    # ================================================================

    def detect(self, frame):
        """
        检测篮筐 (主入口)

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

        # ---- 4. 提取篮筐 (闭合矩形验证) ----
        result = self._extract_box(mask, frame.shape[:2])

        # 降级策略: 当前帧未检测到但上一帧有效, 保留上一帧结果供参考
        if not result['detected'] and self._last_valid_result is not None:
            result['last_valid'] = self._last_valid_result

        if result['detected']:
            self._last_valid_result = {
                'center': result['center'],
                'offset': result['offset'],
                'angle': result['angle'],
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

    def draw_result(self, frame, result, show_mask=False, fps=None):
        """
        在图像上叠加检测结果 (调试/可视化用)

        新增:
          - 配置中心点参考
          - 门/框中心相对中心点偏移可视化
          - FPS 显示
          - 掩码拼接

        Args:
            frame:     原始 BGR 图像
            result:    detect() 返回的结果字典
            show_mask: 是否在右侧拼接显示二值掩码
            fps:       当前处理帧率 (float, 可选)

        Returns:
            vis: 叠加了标注的图像 (BGR)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx_img = w // 2  # 画面垂直中心线位置
        ref_cx, ref_cy = self._resolve_camera_center(w, h)

        # 画参考中心点及十字线（配置中心）
        cv2.circle(vis, (int(ref_cx), int(ref_cy)), 5, (255, 255, 255), -1)
        cv2.line(vis, (int(ref_cx), 0), (int(ref_cx), h), (255, 255, 255), 1)
        cv2.line(vis, (0, int(ref_cy)), (w, int(ref_cy)), (255, 255, 255), 1)

        if result['detected']:
            x1, y1, x2, y2 = result['bbox']
            cx, cy = int(result['center'][0]), int(result['center'][1])

            # 绘制画面垂直中心线 (绿色虚线, 参考)
            for y in range(0, h, 20):
                cv2.line(vis, (cx_img, y), (cx_img, y + 10), (0, 255, 0), 1)

            # 绘制篮筐框 (红色粗框)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 绘制框口中心 (青色圆点)
            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

            # 绘制偏移箭头：从摄像头中心指向框口中心
            cv2.arrowedLine(vis, (int(ref_cx), int(ref_cy)), (cx, cy),
                            (255, 255, 0), 2, tipLength=0.3)

            # 文字信息
            dx, dy = result.get('center_offset', (0.0, 0.0))
            cv2.putText(vis, f"Offset: {result['offset']:+.1f}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(vis, f"Offset(dx,dy): ({dx:+.1f}, {dy:+.1f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(vis, f"Angle: {result['angle']:+.1f}deg  Aspect: {result['aspect']:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(vis, f"Aligned: {result['aligned']}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
        else:
            # 未检测到篮筐
            cv2.putText(vis, "NO BOX DETECTED",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2)

        # 可选: FPS 显示
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(vis, fps_text,
                        (w - tw - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

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

        篮筐边框在水下可能因反光/遮挡断裂, 闭运算弥合小断口.
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

    def _extract_box(self, mask, frame_shape):
        """
        从二值掩码中提取闭合矩形篮筐

        策略: 按面积降序遍历轮廓, 逐个做 尺寸过滤 + 闭合矩形验证,
              篮筐不一定是画面中最大的红色区域 (可能同时存在其他红色物),
              只要有一个轮廓通过验证即判定为篮筐.
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

            # 闭合矩形验证 (基于旋转外接矩形, 对倾斜框稳健)
            if not self._validate_closed_box(mask, contour):
                continue

            # 通过验证 → 计算输出
            return self._build_result(contour, x, y, w, h, img_w, img_h,
                                      image_center_x, mask)

        return self._empty_result()

    def _size_ok(self, w, h, img_w, img_h):
        """
        尺寸与宽高比过滤

        收集框 30×40cm, 在画面中的外接矩形应处于合理比例范围内.
        """
        cfg = self.config
        w_ratio = w / img_w
        h_ratio = h / img_h
        aspect = h / w

        if not (cfg['min_box_width_ratio'] <= w_ratio <= cfg['max_box_width_ratio']):
            return False
        if not (cfg['min_box_height_ratio'] <= h_ratio <= cfg['max_box_height_ratio']):
            return False
        if not (cfg['min_aspect_ratio'] <= aspect <= cfg['max_aspect_ratio']):
            return False
        return True

    def _validate_closed_box(self, mask, contour):
        """
        闭合矩形结构验证 (与 P3 门的关键区别, 基于旋转外接矩形)

        用 minAreaRect 得到与篮筐贴合的有向矩形, 在其四条边上采样:
          - 上边: 红色占比高 → 有上边框
          - 下边: 红色占比高 → 有下边框 (门为 Π 形无底杆 → 下边空, 被排除)
          - 左边/右边: 红色占比高 → 有左右边框
          - 内部区 (收缩多边形): 红色占比低 → 内部空心 (区分实心球)

        基于旋转矩形做采样, 对倾斜/横放的篮筐同样稳健.

        Returns:
            bool: True=符合闭合矩形特征
        """
        cfg = self.config
        rect = cv2.minAreaRect(contour)
        box = np.round(cv2.boxPoints(rect)).astype(np.int32)   # 旋转矩形 4 角点

        # 角点向中心内缩 4%: 避免采样到形态学侵蚀/抗锯齿的薄弱外边界
        center = box.mean(axis=0)
        inner_box = box + (center - box) * cfg['edge_inset_ratio']

        def _edge_ratio(p1, p2):
            """沿线段均匀采样, 计算红色像素占比"""
            p1, p2 = np.array(p1, float), np.array(p2, float)
            dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            num = max(20, int(dist))   # 每像素约 1 个采样点
            red = total = 0
            for k in range(num + 1):
                t = k / num
                x = int(round(p1[0] + (p2[0] - p1[0]) * t))
                y = int(round(p1[1] + (p2[1] - p1[1]) * t))
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    total += 1
                    if mask[y, x] > 0:
                        red += 1
            return red / max(1, total)

        # 四边红色占比 (采样内缩后的边线)
        ratios = []
        for i in range(4):
            ratios.append(_edge_ratio(inner_box[i], inner_box[(i + 1) % 4]))

        # 四边须闭合
        if min(ratios) < cfg['edge_fill_threshold']:
            return False

        # 内部空心: 角点向中心收缩 40% 形成内部多边形, 统计内部红色占比
        inner = (box + (center - box) * cfg['interior_shrink_ratio']).astype(np.int32)
        inner_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(inner_mask, [inner], 1)
        interior_red = cv2.countNonZero(cv2.bitwise_and(mask, inner_mask))
        interior_total = cv2.countNonZero(inner_mask)
        interior_r = interior_red / max(1, interior_total)

        if interior_r > cfg['interior_empty_threshold']:
            return False

        return True

    def _build_result(self, contour, x, y, w, h,
                      img_w, img_h, image_center_x, mask):
        """
        计算框口中心、偏移、朝向(倾斜角)、归一化偏移与对齐状态

        语义说明:
          - offset: 框口中心相对画面垂直中心线的水平像素偏移，正=偏右
          - center_offset: 框口中心相对配置好的 camera_center 的偏移 (右/上为正)
          - angle: 框的倾斜角
          - aligned: 是否处于画面中心附近
        """
        cx = x + w / 2.0
        cy = y + h / 2.0

        # 水平偏移: 框口中心到画面垂直中心线的像素数
        offset = cx - image_center_x

        # 归一化偏移 [-1,1] (正=偏右, 负=偏左; 控制端可直接做偏差)
        center_norm_x = offset / image_center_x if image_center_x else 0.0
        center_norm_y = ((cy - img_h / 2.0) / (img_h / 2.0)) if img_h else 0.0

        # 对齐判定
        aligned = abs(center_norm_x) <= self.config['aligned_ratio']

        # ---- 朝向: 旋转外接矩形的倾斜角 ----
        angle, aspect = self._calc_orientation(contour)
        camera_center = self._resolve_camera_center(img_w, img_h)
        dx, dy = self.compute_center_offset((cx, cy), camera_center)

        return {
            'detected': True,
            'box_type': 'box',
            'center': (round(float(cx), 2), round(float(cy), 2)),
            'center_norm': (round(float(center_norm_x), 3),
                            round(float(center_norm_y), 3)),
            'bbox': (int(x), int(y), int(x + w), int(y + h)),
            'offset': round(float(offset), 2),
            'angle': angle,
            'aspect': aspect,
            'aligned': bool(aligned),
            'mask': mask,
            'camera_center': camera_center,
            'center_offset': (round(float(dx), 2), round(float(dy), 2)),
        }

    def _calc_orientation(self, contour):
        """
        计算篮筐在画面中的朝向 (倾斜角)

        OpenCV minAreaRect 的角度返回受宽高/版本影响, 语义混乱,
        改用 PCA 主方向分析: 轮廓点方差最大的方向即篮筐长轴.

        角度语义:
          - angle ∈ [-90, 90], 表示长轴相对画面竖直方向的偏角
          - 0   = 长轴竖直 (正对镜头)
          - 正  = 长轴向画面右侧倾斜
          - 负  = 长轴向画面左侧倾斜
          - aspect (≥1) = 长/短轴长度比, 描述框的摆向

        Returns:
            (angle, aspect): 倾斜角(度), 长宽比
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            return 0.0, 0.0

        # PCA 主方向 (最大方差方向 = 长轴方向)
        mean, eigvecs = cv2.PCACompute(pts, mean=None)
        ax, ay = float(eigvecs[0][0]), float(eigvecs[0][1])

        # 消除长轴上下歧义 (统一指下方), 再求相对竖直方向的偏角
        if ay < 0:
            ax, ay = -ax, -ay
        tilt = float(np.degrees(np.arctan2(ax, abs(ay))))

        # 长宽比: 主/次特征值之比的平方根
        centered = pts - mean
        evals = np.linalg.eigvals(np.cov(centered.T))
        evals = np.sort(np.abs(evals))[::-1]
        aspect = float(np.sqrt(evals[0] / max(evals[1], 1e-9)))

        return round(tilt, 2), round(aspect, 2)

    @staticmethod
    def _empty_result():
        """返回空检测结果"""
        return {
            'detected': False,
            'box_type': None,
            'center': (0.0, 0.0),
            'center_norm': (0.0, 0.0),
            'bbox': (0, 0, 0, 0),
            'offset': 0.0,
            'angle': 0.0,
            'aspect': 0.0,
            'aligned': False,
            'mask': None,
            'camera_center': (0.0, 0.0),
            'center_offset': None,
        }

    def run_camera(self, camera_id=0, show_window=True, print_offset=True):
        """实时打开摄像头并进行篮筐检测。

        该函数封装了 OpenCV 中的摄像头读取逻辑，返回检测结果并输出：
            框中心相对 camera_center 的偏移 dx, dy
        约定：右为 x 正方向，上为 y 正方向.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_id}")

        print(f"[BoxDetector] 已打开摄像头 {camera_id}, camera_center={self.config['camera_center']}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[BoxDetector] 读取摄像头帧失败, 退出.")
                    break

                result = self.detect_with_center_offset(frame)

                if print_offset:
                    if result['detected']:
                        dx, dy = result['center_offset']
                        print(
                            f"[BoxDetector] detected={result['box_type']} "
                            f"center={result['center']} "
                            f"camera_center={result['camera_center']} "
                            f"offset(dx,dy)=({dx:+.1f}, {dy:+.1f}) "
                            f"(右/上为正)"
                        )
                    else:
                        print("[BoxDetector] 未检测到篮筐")

                if show_window:
                    vis = self.draw_result(frame, result)
                    cv2.imshow("P5 Box Detection", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()

        return None
