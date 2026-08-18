"""
P3 穿门 - 门检测器 (复用 P5 篮筐的闭合矩形检测)

核心功能: 从前视摄像头图像中检测红色方形门框 (四边闭合矩形, 内部空心),
         输出门洞中心、门类型 (高门/矮门) 与对齐状态, 供控制器对准穿门.

检测说明:
  门框为红色 "口"形 (正方形/矩形边框, 内部空心), 形状与 P5 放球的篮筐一致,
  因此本模块直接复用 P5 放球/box_detector.py 的 BoxDetector 作为检测内核,
  只在其输出上附加门语义 (door_type 高门/矮门).

检测流水线 (由 BoxDetector 完成):
  原始帧 → 预处理 → HSV红色分割 → 形态学处理 → 轮廓筛选 → 闭合矩形验证 → 框中心/朝向/对齐

输出接口:
  {
      'detected':     bool,        是否检测到门
      'door_type':    str/None,    门类型 'high' / 'low'
      'center':       (x, y),      门洞中心像素坐标
      'center_norm':  (dx, dy),    相对画面中心归一化偏移 [-1,1]
      'bbox':         (x1,y1,x2,y2), 门框外接矩形 (像素)
      'offset':       float,       门洞中心偏离画面垂直中心线的像素数 (正=偏右)
      'angle':        float,       门框倾斜角 (度)
      'aspect':       float,       门框长宽比 (≥1)
      'aligned':      bool,        门洞中心是否与画面中心对齐
      'mask':         ndarray/None 红/门二值掩码 (调试用)
      'camera_center': (x, y),     参考中心点
      'center_offset': (dx, dy),   相对参考中心的偏移 (右/上为正)
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

# 加载 P3 config
_CONFIG_PATH = os.path.join(_MODULE_DIR, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P3_config", _CONFIG_PATH)
_cfg_module = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_module)
DEFAULT_CONFIG = _cfg_module.DEFAULT_CONFIG
get_config = _cfg_module.get_config

# 加载 P5 BoxDetector 作为检测内核 (门框与篮筐同为红色闭合矩形, 直接复用)
_P5_BOX_PATH = os.path.join(_PROJ_ROOT, "P5 放球", "box_detector.py")
_box_spec = importlib.util.spec_from_file_location("P5_box_detector", _P5_BOX_PATH)
_box_module = importlib.util.module_from_spec(_box_spec)
_box_spec.loader.exec_module(_box_module)
BoxDetector = _box_module.BoxDetector


class DoorDetector:
    """
    门检测器 (红色方形门框)

    内部持有 P5 的 BoxDetector 作为检测内核, 在其输出上附加门语义.

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

        # 检测内核: 复用 P5 BoxDetector, P3 配置覆盖其默认参数
        self._box = BoxDetector(self.config)

        # 缓存上一帧有效结果, 用于连续帧丢失时的降级策略
        self._last_valid_result = None

    # ================================================================
    # 公开接口
    # ================================================================

    def detect(self, frame):
        """
        检测门 (主入口)

        流程: 调用 P5 BoxDetector 检测闭合矩形, 再映射为门语义.

        Args:
            frame: BGR 图像 (numpy ndarray)

        Returns:
            dict: 见模块文档 "输出接口".
        """
        # ---- 0. 输入检查 ----
        frame = self._ensure_valid_frame(frame)

        # ---- 1. 复用 P5 检测内核 ----
        box_result = self._box.detect(frame)

        # ---- 2. 映射为门语义 (附加 door_type) ----
        result = self._map_to_door(box_result, frame.shape[:2])

        # ---- 3. 降级策略: 当前帧未检测到时保留上一帧有效结果 ----
        if not result['detected'] and self._last_valid_result is not None:
            result['last_valid'] = self._last_valid_result

        # ---- 4. 更新上一帧有效结果 ----
        if result['detected']:
            self._last_valid_result = {
                'door_type': result['door_type'],
                'center': result['center'],
                'offset': result['offset'],
                'aligned': result['aligned'],
            }

        return result

    def detect_with_center_offset(self, frame):
        """检测门并返回相对摄像头中心点的偏移信息。"""
        result = self.detect(frame)
        h, w = frame.shape[:2]
        camera_center = self._box._resolve_camera_center(w, h)
        if not result['detected']:
            result['center_offset'] = None
            result['camera_center'] = camera_center
            return result

        cx, cy = result['center']
        dx, dy = self._box.compute_center_offset((cx, cy), camera_center)
        result['camera_center'] = camera_center
        result['center_offset'] = (round(float(dx), 2), round(float(dy), 2))
        return result

    def compute_center_offset(self, center_xy, center_point=None):
        """计算门中心相对摄像头中心点的偏移 (右/上为正)。"""
        return self._box.compute_center_offset(center_xy, center_point)

    def update_config(self, new_config):
        """
        运行时更新配置 (例如测试工具滑动条回调)

        Args:
            new_config: dict, 需要更新的参数
        """
        self.config.update(new_config)
        self._box.update_config(new_config)

    def draw_result(self, frame, result, show_mask=False, fps=None):
        """
        在图像上叠加检测结果 (复用 P5 的绘制, 附加门类型标注)

        Args:
            frame:     原始 BGR 图像
            result:    detect() 返回的结果字典
            show_mask: 是否在右侧拼接显示二值掩码
            fps:       当前处理帧率 (float, 可选)

        Returns:
            vis: 叠加了标注的图像 (BGR)
        """
        # 构造 BoxDetector 可识别的结果 (键与 P5 一致)
        box_result = {
            'detected': result['detected'],
            'box_type': 'box',
            'center': result['center'],
            'center_norm': result['center_norm'],
            'bbox': result['bbox'],
            'offset': result['offset'],
            'angle': result.get('angle', 0.0),
            'aspect': result.get('aspect', 0.0),
            'aligned': result['aligned'],
            'mask': result.get('mask'),
            'camera_center': result.get('camera_center', (0.0, 0.0)),
            'center_offset': result.get('center_offset'),
        }
        vis = self._box.draw_result(frame, box_result, show_mask=show_mask, fps=fps)

        # 附加门语义标注
        if result['detected']:
            cv2.putText(vis, f"Door: {result['door_type']}",
                        (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
        else:
            # 覆盖 P5 的 "NO BOX DETECTED" 为门语义
            cv2.putText(vis, "NO DOOR DETECTED",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return vis

    def run_camera(self, camera_id=0, show_window=True, print_offset=True):
        """实时打开摄像头并进行门检测 (控制台输出偏移 dx, dy, 右/上为正)。"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_id}")

        print(f"[DoorDetector] 已打开摄像头 {camera_id}, "
              f"camera_center={self.config.get('camera_center')}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[DoorDetector] 读取摄像头帧失败, 退出.")
                    break

                result = self.detect_with_center_offset(frame)

                if print_offset:
                    if result['detected']:
                        dx, dy = result['center_offset']
                        print(
                            f"[DoorDetector] detected={result['door_type']} "
                            f"center={result['center']} "
                            f"camera_center={result['camera_center']} "
                            f"offset(dx,dy)=({dx:+.1f}, {dy:+.1f}) "
                            f"(右/上为正)"
                        )
                    else:
                        print("[DoorDetector] 未检测到门")

                if show_window:
                    vis = self.draw_result(frame, result)
                    cv2.imshow("P3 Door Detection", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()

        return None

    # ================================================================
    # 内部方法
    # ================================================================

    def _map_to_door(self, box_result, frame_shape):
        """
        把 P5 框检测结果映射为门检测语义

        附加字段:
          - door_type: 由门框外接矩形上边在画面中的纵向位置判定 (高门/矮门)
          - angle/aspect: 保留 P5 的门框朝向/长宽比信息
        """
        img_h = frame_shape[0]

        if not box_result['detected']:
            return self._empty_result()

        # 门类型: 门顶(外接矩形上边)在画面中的归一化 y 位置
        top_norm = box_result['bbox'][1] / img_h if img_h else 1.0
        threshold = self.config.get('door_top_norm_threshold', 0.5)
        door_type = 'high' if top_norm < threshold else 'low'

        return {
            'detected': True,
            'door_type': door_type,
            'center': box_result['center'],
            'center_norm': box_result['center_norm'],
            'bbox': box_result['bbox'],
            'offset': box_result['offset'],
            'angle': box_result['angle'],
            'aspect': box_result['aspect'],
            'aligned': box_result['aligned'],
            'mask': box_result['mask'],
            'camera_center': box_result['camera_center'],
            'center_offset': box_result['center_offset'],
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
            'angle': 0.0,
            'aspect': 0.0,
            'aligned': False,
            'mask': None,
            'camera_center': (0.0, 0.0),
            'center_offset': None,
        }

    @staticmethod
    def _ensure_valid_frame(frame):
        """确保输入为有效的 BGR 图像, 否则抛出清晰异常。"""
        if frame is None:
            raise ValueError("DoorDetector.detect() 输入 frame 为空。")
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"DoorDetector.detect() 期望输入为 3 通道 BGR 图像, "
                f"实际 shape={frame.shape}。"
            )
        return frame
