"""
P2 撞球 - 本地调参测试工具

功能:
  1. 支持图片 / 视频文件 / 摄像头三种输入源
  2. 滑动条实时调整 YOLO 置信度阈值
  3. 实时显示 原始画面 + 检测结果叠加
  4. 控制台输出当前帧的 转向决策 / 偏移量 / 距离

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

# ---- 动态加载撞球检测器模块 (中文文件夹名, 无法用标准 import) ----
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载 config
_CFG_PATH = os.path.join(_PROJ_ROOT, "P2 撞球", "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P2_config", _CFG_PATH)
_cfg_mod = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_mod)
DEFAULT_CONFIG = _cfg_mod.DEFAULT_CONFIG
get_config = _cfg_mod.get_config

# 加载 ball_detector
_P2_PATH = os.path.join(_PROJ_ROOT, "P2 撞球", "ball_detector.py")
_bd_spec = importlib.util.spec_from_file_location("ball_detector", _P2_PATH)
_bd = importlib.util.module_from_spec(_bd_spec)
_bd_spec.loader.exec_module(_bd)
BallDetector = _bd.BallDetector

# ================================================================
# 窗口名称
# ================================================================
WIN_CONTROL = "Controls"
WIN_ORIGINAL = "Original"
WIN_RESULT = "Result"


class TestTool:
    """撞球检测调参测试工具"""

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
        self.detector = BallDetector(self.config)

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
        print("  撞球检测测试工具")
        print("  q/ESC: 退出 | s: 截图 | p: 打印参数 | Space: 暂停")
        print("=" * 55 + "\n")

        fps_start = cv2.getTickCount()
        fps_count = 0
        fps = 0.0

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

                # 统计 FPS
                fps_count += 1
                now = cv2.getTickCount()
                elapsed = (now - fps_start) / cv2.getTickFrequency()
                if elapsed >= 1.0:
                    fps = fps_count / elapsed
                    fps_start = now
                    fps_count = 0

                # 每 30 帧打印一次检测结果
                if self.frame_count % 30 == 1:
                    self._print_result(result, fps)

            # 显示
            cv2.imshow(WIN_ORIGINAL, vis_original)
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
        conf = cv2.getTrackbarPos("Conf(%)", WIN_CONTROL) / 100.0
        self.detector.update_config({'conf_threshold': conf})

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
        cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_CONTROL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_ORIGINAL, 480, 360)
        cv2.resizeWindow(WIN_RESULT, 480, 360)
        cv2.resizeWindow(WIN_CONTROL, 300, 80)

    def _create_trackbars(self):
        """创建调参滑动条"""
        conf = self.config['conf_threshold']
        cv2.createTrackbar("Conf(%)", WIN_CONTROL,
                           int(conf * 100), 100, self._on_trackbar)

    def _print_result(self, result, fps):
        """打印当前帧检测结果"""
        if result['detected']:
            info = (f"  [Frame {self.frame_count:4d}]  "
                    f"steer={result['steer']:<6s}  "
                    f"offset={result['offset']:+6.1f}px  ")
            if result.get('distance') is not None:
                info += f"dist={result['distance']:5.0f}cm  "
            if result.get('strike_ready'):
                info += "| 可撞击!"
            print(info, end='')
        else:
            print(f"  [Frame {self.frame_count:4d}]  未检测到目标球  ", end='')
        print(f"| FPS={fps:.0f}")

    def _print_params(self):
        """打印当前参数配置"""
        cfg = self.detector.config
        print("\n--- 当前参数 ---")
        print(f"  Model Path: {cfg['model_path']}")
        print(f"  Conf Threshold: {cfg['conf_threshold']}")
        print(f"  IOU Threshold: {cfg['iou_threshold']}")
        print(f"  Target Classes: {cfg['target_class_names']}")
        print(f"  Target Strategy: {cfg['target_strategy']}")
        print(f"  Center Zone Ratio: {cfg['center_zone_ratio']}")
        print(f"  Strike Width: {cfg['strike_width_px']}px")
        print(f"  Ball Diameter: {cfg['ball_real_diameter_cm']}cm")
        print("-----------------\n")

    def _save_screenshot(self, vis):
        """保存截图 (到 ASCII 临时目录, 兼容中文工作目录)"""
        import time
        import tempfile
        out_dir = os.path.join(tempfile.gettempdir(), "p2_screenshots")
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
        description="P2 撞球 - 本地调参测试工具",
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
