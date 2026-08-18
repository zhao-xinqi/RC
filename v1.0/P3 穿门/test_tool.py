"""
P3 穿门 - 本地调参测试工具

功能:
  1. 支持图片 / 视频文件 / 摄像头三种输入源
  2. 滑动条实时调整 HSV 阈值、形态学核大小、最小面积
  3. 实时显示 原始画面 + 二值掩码 + 检测结果叠加
  4. 控制台输出当前帧的 门类型 / 偏移量 / 对齐状态

用法:
  python test_tool.py                        # 使用默认摄像头 (index 0)
  python test_tool.py --camera 1             # 使用摄像头 1
  python test_tool.py --image path/to/img.jpg  # 测试单张图片 (滑动条调参)
  python test_tool.py --video path/to/vid.mp4  # 测试视频文件
  python test_tool.py --resolution 1280 720  # 指定摄像头分辨率

键盘快捷键:
  q / ESC  - 退出
  s        - 保存当前帧 (含检测叠加) 到文件
  p        - 打印当前参数到控制台
  空格     - 暂停
"""

import os
import sys
import argparse
import importlib.util

import cv2
import numpy as np

# ---- 动态加载门检测器模块 (中文文件夹名, 无法用标准 import) ----
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载 config
_CFG_PATH = os.path.join(_PROJ_ROOT, "P3 穿门", "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P3_config", _CFG_PATH)
_cfg_mod = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)
DEFAULT_CONFIG = _cfg_mod.DEFAULT_CONFIG
get_config = _cfg_mod.get_config

# 加载 door_detector
_P3_PATH = os.path.join(_PROJ_ROOT, "P3 穿门", "door_detector.py")
_dd_spec = importlib.util.spec_from_file_location("door_detector", _P3_PATH)
_dd = importlib.util.module_from_spec(_dd_spec)
_dd_spec.loader.exec_module(_dd)
DoorDetector = _dd.DoorDetector

# ================================================================
# 窗口名称
# ================================================================
WIN_CONTROL = "Controls"
WIN_ORIGINAL = "Original"
WIN_MASK = "Mask"
WIN_RESULT = "Result"


class TestTool:
    """门检测调参测试工具"""

    def __init__(self, source_type, source_path=None, camera_index=0,
                 cam_width=640, cam_height=480):
        """
        Args:
            source_type:  'image' | 'video' | 'camera'
            source_path:  图片/视频文件路径 (camera 时为 None)
            camera_index: 摄像头索引
            cam_width:    摄像头分辨率宽
            cam_height:   摄像头分辨率高
        """
        self.source_type = source_type
        self.source_path = source_path
        self.camera_index = camera_index
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.paused = False
        self.frame_count = 0

        # 初始化检测器
        self.config = get_config()
        self.detector = DoorDetector(self.config)

        # 初始化视频源
        self.cap = None
        self._init_source()

        # 创建窗口和滑动条
        self._create_windows()
        self._create_trackbars()

    # ================================================================
    # 运行
    # ================================================================

    def run(self):
        """主循环"""
        print("\n" + "=" * 55)
        print("  穿门检测测试工具")
        print("  q/ESC: 退出 | s: 截图 | p: 打印参数 | Space: 暂停")
        print("=" * 55 + "\n")

        while True:
            if not self.paused:
                ret, frame = self._read_frame()
                if not ret:
                    print("[!] 无法读取帧, 退出.")
                    break
                self.frame_count += 1

                # 执行检测
                result = self.detector.detect(frame)

                # 绘制显示
                vis_original = frame.copy()
                vis_result = self.detector.draw_result(frame, result)
                vis_mask = result.get('mask')
                if vis_mask is None:
                    vis_mask = np.zeros_like(frame)

                # 每 30 帧打印一次检测结果
                if self.frame_count % 30 == 1:
                    self._print_result(result)

            # 显示
            cv2.imshow(WIN_ORIGINAL, vis_original)
            cv2.imshow(WIN_MASK, vis_mask)
            cv2.imshow(WIN_RESULT, vis_result)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):   # q 或 ESC
                break
            elif key == ord('s'):
                self._save_screenshot(vis_result)
            elif key == ord('p'):
                self._print_params()
            elif key == ord(' '):
                self.paused = not self.paused
                if self.paused:
                    print("[暂停] 按空格继续")
                else:
                    print("[继续]")

        self._cleanup()

    # ================================================================
    # 滑动条回调
    # ================================================================

    def _on_trackbar(self, val):
        """滑动条变化时更新检测器配置"""
        # 读取所有滑动条的当前值
        h_low = cv2.getTrackbarPos("H Low", WIN_CONTROL)
        h_high = cv2.getTrackbarPos("H High", WIN_CONTROL)
        s_low = cv2.getTrackbarPos("S Low", WIN_CONTROL)
        s_high = cv2.getTrackbarPos("S High", WIN_CONTROL)
        v_low = cv2.getTrackbarPos("V Low", WIN_CONTROL)
        v_high = cv2.getTrackbarPos("V High", WIN_CONTROL)
        open_k = cv2.getTrackbarPos("Open Kernel", WIN_CONTROL)
        close_k = cv2.getTrackbarPos("Close Kernel", WIN_CONTROL)
        area_min = cv2.getTrackbarPos("Min Area(x100)", WIN_CONTROL) * 100

        # 更新检测器配置
        self.detector.update_config({
            'hsv_lower': [h_low, s_low, v_low],
            'hsv_upper': [h_high, s_high, v_high],
            'morph_open_kernel': max(1, open_k),
            'morph_close_kernel': max(1, close_k),
            'min_contour_area': area_min,
        })

    # ================================================================
    # 内部方法
    # ================================================================

    def _init_source(self):
        """初始化视频源"""
        if self.source_type == 'image':
            if not os.path.exists(self.source_path):
                raise FileNotFoundError(f"图片不存在: {self.source_path}")
            self._static_frame = cv2.imread(self.source_path)
            if self._static_frame is None:
                raise ValueError(f"无法读取图片: {self.source_path}")
            print(f"[*] 已加载图片: {self.source_path}")
            print(f"    尺寸: {self._static_frame.shape[1]}×{self._static_frame.shape[0]}")

        elif self.source_type == 'camera':
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 index={self.camera_index}")
            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"[*] 已打开摄像头 index={self.camera_index}")
            print(f"    分辨率: {actual_w:.0f}×{actual_h:.0f}")

        elif self.source_type == 'video':
            if not os.path.exists(self.source_path):
                raise FileNotFoundError(f"视频不存在: {self.source_path}")
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频: {self.source_path}")
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_v = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[*] 已打开视频: {self.source_path}")
            print(f"    总帧数: {total}, FPS: {fps_v:.1f}")

    def _read_frame(self):
        """读取一帧"""
        if self.source_type == 'image':
            return True, self._static_frame.copy()
        else:
            ret, frame = self.cap.read()
            return ret, frame

    def _create_windows(self):
        """创建显示窗口"""
        cv2.namedWindow(WIN_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_CONTROL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_ORIGINAL, 480, 360)
        cv2.resizeWindow(WIN_MASK, 480, 360)
        cv2.resizeWindow(WIN_RESULT, 480, 360)
        cv2.resizeWindow(WIN_CONTROL, 500, 420)

    def _create_trackbars(self):
        """创建调参滑动条"""
        cfg = self.config

        # 空函数作为回调占位 (实际读取在 _on_trackbar 中)
        def nothing(x):
            pass

        # HSV 阈值
        cv2.createTrackbar("H Low",  WIN_CONTROL, cfg['hsv_lower'][0],  179, self._on_trackbar)
        cv2.createTrackbar("H High", WIN_CONTROL, cfg['hsv_upper'][0],  179, self._on_trackbar)
        cv2.createTrackbar("S Low",  WIN_CONTROL, cfg['hsv_lower'][1],  255, self._on_trackbar)
        cv2.createTrackbar("S High", WIN_CONTROL, cfg['hsv_upper'][1],  255, self._on_trackbar)
        cv2.createTrackbar("V Low",  WIN_CONTROL, cfg['hsv_lower'][2],  255, self._on_trackbar)
        cv2.createTrackbar("V High", WIN_CONTROL, cfg['hsv_upper'][2],  255, self._on_trackbar)

        # 形态学
        cv2.createTrackbar("Open Kernel",  WIN_CONTROL, cfg['morph_open_kernel'],  15, self._on_trackbar)
        cv2.createTrackbar("Close Kernel", WIN_CONTROL, cfg['morph_close_kernel'], 15, self._on_trackbar)

        # 最小面积 (×100 方便滑动条调节)
        cv2.createTrackbar("Min Area(x100)", WIN_CONTROL,
                           cfg['min_contour_area'] // 100, 200, self._on_trackbar)

    def _print_result(self, result):
        """打印当前帧检测结果"""
        if result['detected']:
            print(f"  [Frame {self.frame_count:4d}]  "
                  f"door={result['door_type']:<4s}  "
                  f"offset={result['offset']:+6.1f}px  "
                  f"aligned={result['aligned']}")
        else:
            print(f"  [Frame {self.frame_count:4d}]  未检测到门")

    def _print_params(self):
        """打印当前参数配置"""
        cfg = self.detector.config
        print("\n--- 当前参数 ---")
        print(f"  HSV Lower: {cfg['hsv_lower']}")
        print(f"  HSV Upper: {cfg['hsv_upper']}")
        print(f"  Open Kernel: {cfg['morph_open_kernel']}")
        print(f"  Close Kernel: {cfg['morph_close_kernel']}")
        print(f"  Min Area: {cfg['min_contour_area']}")
        print(f"  Beam Fill: {cfg['beam_fill_threshold']}")
        print(f"  Pillar Fill: {cfg['pillar_fill_threshold']}")
        print(f"  Bottom Empty: {cfg['bottom_empty_threshold']}")
        print(f"  Interior Empty: {cfg['interior_empty_threshold']}")
        print(f"  Aligned Ratio: {cfg['aligned_ratio']}")
        print(f"  Door Top Norm Threshold: {cfg['door_top_norm_threshold']}")
        print("-----------------\n")

    def _save_screenshot(self, vis):
        """保存截图 (到 ASCII 临时目录, 兼容中文工作目录)"""
        import time
        import tempfile
        out_dir = os.path.join(tempfile.gettempdir(), "p3_screenshots")
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(
            out_dir, f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
        ok = cv2.imwrite(filename, vis)
        if ok:
            print(f"[*] 截图已保存: {filename}")
        else:
            print("[!] 截图保存失败")

    def _cleanup(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


# ================================================================
# 命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="P3 穿门 - 本地调参测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_tool.py                         # 默认摄像头
  python test_tool.py --camera 1              # 指定摄像头
  python test_tool.py --image test.jpg        # 测试图片
  python test_tool.py --video test.mp4        # 测试视频
  python test_tool.py --resolution 1280 720   # 指定摄像头分辨率
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--image', type=str, metavar='PATH',
                       help='测试单张图片 (滑动条调参)')
    group.add_argument('--video', type=str, metavar='PATH',
                       help='测试视频文件')
    parser.add_argument('--camera', type=int, default=0, metavar='N',
                        help='摄像头索引 (默认: 0)')
    parser.add_argument('--resolution', type=int, nargs=2,
                        default=[640, 480], metavar=('W', 'H'),
                        help='摄像头分辨率 (默认: 640 480)')

    args = parser.parse_args()

    # 确定输入源类型
    if args.image:
        source_type, source_path = 'image', args.image
    elif args.video:
        source_type, source_path = 'video', args.video
    else:
        source_type, source_path = 'camera', None

    # 启动测试工具
    tool = TestTool(
        source_type=source_type,
        source_path=source_path,
        camera_index=args.camera,
        cam_width=args.resolution[0],
        cam_height=args.resolution[1],
    )
    tool.run()


if __name__ == '__main__':
    main()
