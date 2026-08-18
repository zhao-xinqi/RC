"""
main.py - 视觉系统主入口

集成所有视觉任务模块 (P1~P5), 提供统一的摄像头采集与任务调度.

模块架构:
  0.预处理    → 通用水下图像预处理 (缩放/颜色校正/去噪/CLAHE)
  P1 巡线     → 池底橙红色引导线检测 (下置摄像头)
  P2 撞球     → 前置摄像头 YOLO 检测目标球, 输出转向/撞击决策
  P3 穿门     → 前置摄像头 HSV+Π形验证 检测高低门, 输出门洞中心
  P4 抓球     → (待实现) 目标球抓取检测
  P5 放球     → 下置摄像头 HSV+闭合矩形验证 识别红色篮筐, 输出框口中心
                (P2/P3 前置任务按 TASK_STAGE 切换)

摄像头规范:
  索引 0 = 前置摄像头 (前视)   → 撞球等前向任务
  索引 1 = 下置摄像头 (下视)   → 巡线等底部任务
  两个摄像头独立初始化, 任一缺失不影响整体程序运行
  两个摄像头的图像均进行预处理, 由各任务模块内部调用共享的 0.预处理 完成

用法:
  python visual_main.py                 # 默认配置 (前置=0, 下置=1)
  python visual_main.py --no-display    # 无窗口模式 (仅打印结果)
"""

import os
import sys
import argparse
import importlib.util

import cv2
import numpy as np



# ================================================================
# 各项参数配置 
# ================================================================
#1.摄像头参数配置
camera_bottom_idx = 1   # 下置摄像头 (下视, 巡线等底部任务)
camera_front_idx = 0    # 前置摄像头 (前视, 撞球等前向任务)
yolo_model_path = "yolov8n.pt"  # YOLO 模型文件路径 (可为内置模型名或本地 .pt 文件)

# 当前任务阶段 (按比赛流程切换, 任务须顺序完成), 决定两路摄像头的任务
#   "P2" → 撞球:  前置 YOLO BallDetector     下置 P1 巡线
#   "P3" → 穿门:  前置 HSV DoorDetector      下置 P1 巡线
#   "P5" → 放球:  前置 仅显示画面            下置 HSV BoxDetector (识别红色篮筐)
#   P3/P5 均纯 HSV, 无需 YOLO 模型
TASK_STAGE = "P5"


#2.任务参数配置（读取P0-P5的配置文件）  
P0_PREPROCESSING_CONFIG = {
    "image_width": 640,           # 采集/预处理宽度
    "image_height": 480,          # 采集/预处理高度
    "enable_resize": True,        # 是否缩放至目标分辨率
    "enable_color_correct": True, # 是否水下颜色校正 (补偿红光衰减)
    "red_boost": 1.2,             # 红色通道增强系数 (>1.0 增强)
    "enable_gaussian": True,      # 是否高斯去噪
    "gaussian_kernel": 5,         # 高斯核大小 (奇数)
    "enable_clahe": True,         # 是否 CLAHE 对比度增强
    "clahe_clip": 2.0,            # CLAHE 对比度限幅
    "clahe_tile": (8, 8),         # CLAHE 网格大小
}

P1_LINE_DETECTION_CONFIG = {
    "hsv_lower": [0, 100, 100],   # 橙红色 HSV 下界 [H, S, V]
    "hsv_upper": [20, 255, 255],  # 橙红色 HSV 上界 [H, S, V]
    "min_contour_area": 100,      # 最小轮廓面积 (像素²), 过滤噪声
}

P2_BALL_DETECTION_CONFIG = {
    "conf_threshold": 0.5,   # 置信度阈值
    "iou_threshold": 0.3,    # NMS 重叠抑制阈值
}

P3_DOOR_DETECTION_CONFIG = {
    # 未提供的参数使用 P3 穿门/config.py 中的 DEFAULT_CONFIG 默认值
    "hsv_lower": [0, 80, 80],      # 红色 HSV 下界 [H, S, V]
    "hsv_upper": [25, 255, 255],   # 红色 HSV 上界 [H, S, V]
    "hsv_lower2": [150, 80, 80],   # 红色跨边界段下界
    "hsv_upper2": [179, 255, 255], # 红色跨边界段上界
    "morph_open_kernel": 3,        # 开运算核 (去噪)
    "morph_close_kernel": 9,       # 闭运算核 (弥合门框断裂)
    "min_contour_area": 800,       # 最小轮廓面积 (像素²)
}

P5_BOX_DETECTION_CONFIG = {
    # 未提供的参数使用 P5 放球/config.py 中的 DEFAULT_CONFIG 默认值
    "hsv_lower": [0, 80, 80],      # 红色 HSV 下界 [H, S, V]
    "hsv_upper": [25, 255, 255],   # 红色 HSV 上界 [H, S, V]
    "hsv_lower2": [150, 80, 80],   # 红色跨边界段下界
    "hsv_upper2": [179, 255, 255], # 红色跨边界段上界
    "morph_open_kernel": 3,        # 开运算核 (去噪)
    "morph_close_kernel": 9,       # 闭运算核 (弥合边框断裂)
    "min_contour_area": 600,       # 最小轮廓面积 (像素²)
}

# ================================================================
# 动态加载模块 (处理中文/数字前缀文件夹名)
# ================================================================

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))  #定义项目根目录

def _load_task_module(task_folder, module_name):
    """
    通用动态模块加载

    Args:
        task_folder: 任务文件夹名 (如 "P1 巡线")
        module_name: 模块文件名 (如 "line_detector.py")

    Returns:
        loaded module
    """
    module_path = os.path.join(_PROJ_ROOT, task_folder, module_name)
    spec = importlib.util.spec_from_file_location(
        module_name.replace('.py', ''), module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================
#辅助函数
# ================================================================

def _open_camera(index, name):
    """
    打开单个摄像头并设置采集分辨率

    处理 Windows 下 USB 摄像头常见的黑屏问题:
      1. 先用默认后端打开并预热 (丢弃前几帧, 等待曝光/白平衡稳定)
      2. 预热后仍为全黑帧时, 自动改用 DSHOW 后端重试
      3. 重试后仍失败则打印提示, 便于排查硬件问题

    Args:
        index: 摄像头索引 (0=前置, 1=下置)
        name:  摄像头名称 (前置/下置, 仅用于日志)

    Returns:
        cv2.VideoCapture 或 None (打开失败时不中断程序)
    """
    # 依次尝试的后端列表: 默认后端 → DSHOW (Windows USB 摄像头更稳定)
    backends = [cv2.CAP_ANY]
    if sys.platform.startswith('win'):
        backends.append(cv2.CAP_DSHOW)

    for i, backend in enumerate(backends):
        if i > 0:
            print(f"[*] {name}摄像头默认后端异常, 改用 DSHOW 后端重试...")
        cap = _try_open_camera(index, backend, name, quiet=(i > 0))
        if cap is not None:
            return cap

    print(f"[!] {name}摄像头 (index={index}) 打开失败")
    return None


def _try_open_camera(index, backend, name, quiet=False):
    """
    用指定后端尝试打开单个摄像头并预热

    Args:
        index:   摄像头索引
        backend: OpenCV 视频后端 (CAP_ANY / CAP_DSHOW)
        name:    摄像头名称 (仅用于日志)
        quiet:   是否抑制黑屏警告 (后端切换后的二次尝试)

    Returns:
        cv2.VideoCapture 或 None (打开失败 / 预热后仍为黑屏)
    """
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, P0_PREPROCESSING_CONFIG['image_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, P0_PREPROCESSING_CONFIG['image_height'])

    # 打开失败或预热后仍为黑屏 → 释放并返回 None
    if not cap.isOpened() or not _camera_warmup(cap, name, quiet=quiet):
        cap.release()
        return None

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[*] {name}摄像头 (index={index}) 已就绪: "
          f"{actual_w:.0f}×{actual_h:.0f}")
    return cap


def _camera_warmup(cap, name, n_frames=5, quiet=False):
    """
    摄像头预热: 读取并丢弃前几帧, 等待相机自动曝光/白平衡稳定

    部分 USB 摄像头刚打开时前几帧为全黑帧, 直接使用会看到黑屏.
    预热期间若始终读到全黑帧, 打印排查提示.

    Args:
        cap:      已打开的 VideoCapture
        name:     摄像头名称 (仅用于日志)
        n_frames: 最多尝试的预热帧数
        quiet:    True 时不打印黑屏警告

    Returns:
        bool: True=已读到有效帧, False=预热后仍为全黑帧
    """
    for _ in range(n_frames):
        ret, frame = cap.read()
        # 亮度均值 > 1.0 视为有效画面 (全黑帧均值约为 0)
        if ret and frame is not None and float(frame.mean()) > 1.0:
            return True

    if not quiet:
        print(f"[!] {name}摄像头预热后仍返回全黑帧, "
              f"请检查镜头遮挡/线缆连接/是否被其他程序占用")
    return False


# ================================================================
# 任务处理模块 (主循环中, 每个摄像头一个独立模块)
# ================================================================

def run_front_task(cap_front, ball_detector):
    """
    摄像头 0 (前置) 任务模块: P2 撞球

    处理单帧:
      1. 读取前置摄像头画面
      2. BallDetector 内部完成 水下预处理 + YOLO 检测目标球
      3. 输出转向/距离/撞击决策, 并叠加显示

    Args:
        cap_front:    前置摄像头 VideoCapture
        ball_detector: P2 撞球检测器 (BallDetector)

    Returns:
        bool: True=继续主循环, False=应退出 (本模块不会主动退出)
    """
    # 摄像头未初始化: 跳过 P2, 不影响主循环
    if cap_front is None:
        return True

    ret, frame = cap_front.read()
    if not ret:
        print("[!] 前置摄像头读取失败, 跳过当前帧")
        return True

    # 检测 (内部完成预处理 + YOLO 推理)
    result = ball_detector.detect(frame)
    vis = ball_detector.draw_result(frame, result)

    # 控制台输出转向/撞击决策
    if result['detected']:
        info = (f"\r  P2 steer={result['steer']:<6s}  "
                f"offset={result['offset']:+6.1f}px  ")
        if result.get('distance') is not None:
            info += f"dist={result['distance']:5.0f}cm"
        if result.get('strike_ready'):
            info += "  <-- 撞击!"
        print(info, end='', flush=True)
    else:
        print("\r  P2 未检测到目标球...", end='', flush=True)

    # 注意: 窗口名必须为纯 ASCII, 否则 Windows 下可能黑屏
    cv2.imshow("Camera 0 - P2 Ball Hitting", vis)
    return True


def run_door_task(cap_front, door_detector):
    """
    摄像头 0 (前置) 任务模块: P3 穿门

    处理单帧:
      1. 读取前置摄像头画面
      2. DoorDetector 内部完成 水下预处理 + HSV红分割 + Π形验证
      3. 输出门类型/偏移/对齐状态, 并叠加显示

    Args:
        cap_front:     前置摄像头 VideoCapture
        door_detector: P3 穿门检测器 (DoorDetector)

    Returns:
        bool: True=继续主循环, False=应退出 (本模块不会主动退出)
    """
    # 摄像头未初始化: 跳过 P3, 不影响主循环
    if cap_front is None:
        return True

    ret, frame = cap_front.read()
    if not ret:
        print("[!] 前置摄像头读取失败, 跳过当前帧")
        return True

    # 检测 (内部完成预处理 + HSV分割 + Π形验证)
    result = door_detector.detect(frame)
    vis = door_detector.draw_result(frame, result)

    # 控制台输出门类型/偏移/对齐状态
    if result['detected']:
        print(f"\r  P3 door={result['door_type']:<4s}  "
              f"offset={result['offset']:+6.1f}px  "
              f"aligned={result['aligned']}  ",
              end='', flush=True)
    else:
        print("\r  P3 未检测到门...", end='', flush=True)

    # 注意: 窗口名必须为纯 ASCII, 否则 Windows 下可能黑屏
    cv2.imshow("Camera 0 - P3 Door Passing", vis)
    return True


def run_box_task(cap_bottom, box_detector):
    """
    摄像头 1 (下置) 任务模块: P5 放球 (识别红色篮筐)

    处理单帧:
      1. 读取下置摄像头画面 (俯视池底/收集框)
      2. BoxDetector 内部完成 水下预处理 + HSV红分割 + 闭合矩形验证
      3. 输出框口偏移/倾斜角/对齐状态, 并叠加显示

    Args:
        cap_bottom:   下置摄像头 VideoCapture
        box_detector: P5 放球检测器 (BoxDetector)

    Returns:
        bool: True=继续主循环, False=应退出 (下置摄像头读取失败)
    """
    # 摄像头未初始化: 跳过 P5, 不影响主循环
    if cap_bottom is None:
        return True

    ret, frame = cap_bottom.read()
    if not ret:
        print("[!] 下置摄像头读取失败, 退出")
        return False

    # 检测 (内部完成预处理 + HSV分割 + 闭合矩形验证)
    result = box_detector.detect(frame)
    vis = box_detector.draw_result(frame, result)

    # 控制台输出框口偏移/倾斜角/对齐状态
    if result['detected']:
        print(f"\r  P5 offset={result['offset']:+6.1f}px  "
              f"angle={result['angle']:+5.1f}°  "
              f"aspect={result['aspect']:.2f}  "
              f"aligned={result['aligned']}  ",
              end='', flush=True)
    else:
        print("\r  P5 未检测到篮筐...", end='', flush=True)

    # 注意: 窗口名必须为纯 ASCII, 否则 Windows 下可能黑屏
    cv2.imshow("Camera 1 - P5 Box Placing", vis)
    return True


def run_bottom_task(cap_bottom, line_detector):
    """
    摄像头 1 (下置) 任务模块: P1 巡线

    处理单帧:
      1. 读取下置摄像头画面
      2. LineDetector 内部完成 水下预处理 + 引导线检测
      3. 输出 offset/angle, 并叠加显示

    Args:
        cap_bottom:   下置摄像头 VideoCapture
        line_detector: P1 巡线检测器 (LineDetector)

    Returns:
        bool: True=继续主循环, False=应退出 (下置摄像头读取失败)
    """
    # 摄像头未初始化: 跳过 P1, 不影响主循环
    if cap_bottom is None:
        return True

    ret, frame = cap_bottom.read()
    if not ret:
        print("[!] 下置摄像头读取失败, 退出")
        return False

    # 检测 (内部完成预处理 + 引导线提取)
    result = line_detector.detect(frame)
    vis = line_detector.draw_result(frame, result)

    if result['detected']:
        print(f"\r  P1 offset={result['offset']:+6.1f}px  "
              f"angle={result['angle']:+5.1f}°  ",
              end='', flush=True)
    else:
        print("\r  P1 未检测到引导线...", end='', flush=True)

    cv2.imshow("Camera 1 - P1 Line Following", vis)
    return True


def run_display_task(cap, window_name):
    """
    摄像头 任务模块: 仅显示画面 (不执行检测)

    用于当前摄像头在某任务阶段无任务时, 保持画面显示.
    例: P5 放球阶段, 前置摄像头 (index 0) 无任务, 仅显示画面.

    Args:
        cap:         摄像头 VideoCapture
        window_name: 显示窗口名 (必须纯 ASCII)

    Returns:
        bool: True=继续主循环, False=应退出 (摄像头读取失败)
    """
    # 摄像头未初始化: 跳过, 不影响主循环
    if cap is None:
        return True

    ret, frame = cap.read()
    if not ret:
        print("[!] 摄像头读取失败, 跳过当前帧")
        return True

    cv2.imshow(window_name, frame)
    return True


# ================================================================
#主程序入口
# ================================================================

def main():
    """
    视觉系统主程序入口

    流程:
      1. 初始化两个摄像头 (索引 0=前置 / 索引 1=下置), 各自独立
      2. 按 TASK_STAGE 加载两路摄像头的任务检测器
      3. 主循环中, 两个摄像头任务各由独立模块处理:

         前置 (index 0): P2 撞球 (run_front_task) / P3 穿门 (run_door_task)
                        P5 阶段无任务 → 仅显示画面 (run_display_task)
         下置 (index 1): P1 巡线 (run_bottom_task) / P5 放球 (run_box_task)

    摄像头分工:
      - 摄像头 1 (下置): P1 巡线 (引导线) / P5 放球 (识别红色篮筐)
      - 摄像头 0 (前置): P2 撞球 (YOLO 目标球) / P3 穿门 (高低门) / 仅显示

    退出: 按 'q' 键结束程序
    """
    # ==========================================================
    # 1. 初始化两个摄像头 (任一缺失不影响整体运行)
    # ==========================================================
    cap_front = _open_camera(camera_front_idx, "前置")
    cap_bottom = _open_camera(camera_bottom_idx, "下置")

    if cap_front is None and cap_bottom is None:
        print("[!] 两个摄像头均打开失败, 程序退出")
        return

    # ==========================================================
    # 2. 按 TASK_STAGE 加载任务检测器 (动态加载中文目录)
    #    前置: P2 撞球 / P3 穿门;  P5 阶段不加载, 仅显示画面
    #    下置: P1 巡线 / P5 放球 (二者互斥, 按阶段二选一)
    # ==========================================================
    front_detector = None
    front_task_name = "仅显示画面"
    if TASK_STAGE == "P3":
        door_mod = _load_task_module("P3 穿门", "door_detector.py")
        front_detector = door_mod.DoorDetector(P3_DOOR_DETECTION_CONFIG)
        front_task_name = "P3 穿门"
    elif TASK_STAGE == "P2":
        ball_mod = _load_task_module("P2 撞球", "ball_detector.py")
        front_detector = ball_mod.BallDetector(P2_BALL_DETECTION_CONFIG)
        front_task_name = "P2 撞球"

    line_detector = None
    box_detector = None
    if TASK_STAGE == "P5":
        box_mod = _load_task_module("P5 放球", "box_detector.py")
        box_detector = box_mod.BoxDetector(P5_BOX_DETECTION_CONFIG)
        bottom_task_name = "P5 放球"
    else:
        line_mod = _load_task_module("P1 巡线", "line_detector.py")
        line_detector = line_mod.LineDetector(P1_LINE_DETECTION_CONFIG)
        bottom_task_name = "P1 巡线"

    print(f"\n[*] 视觉系统启动: "
          f"前置={front_task_name}, 下置={bottom_task_name}, 按 'q' 退出...\n")

    # ==========================================================
    # 3. 主循环: 两路摄像头任务 (各为独立模块)
    # ==========================================================
    while True:
        # 摄像头 0 (前置): P2/P3 阶段执行检测, P5 阶段仅显示
        if front_detector is not None:
            if TASK_STAGE == "P3":
                run_door_task(cap_front, front_detector)
            else:
                run_front_task(cap_front, front_detector)
        else:
            run_display_task(cap_front, "Camera 0 - Front View")

        # 摄像头 1 (下置): P5 阶段执行放球, 其他阶段执行巡线 (读取失败则退出)
        if box_detector is not None:
            if not run_box_task(cap_bottom, box_detector):
                break
        else:
            if not run_bottom_task(cap_bottom, line_detector):
                break

        # 按键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ==========================================================
    # 4. 释放资源
    # ==========================================================
    if cap_front is not None:
        cap_front.release()
    if cap_bottom is not None:
        cap_bottom.release()
    cv2.destroyAllWindows()
    print("\n[*] 视觉系统已退出")


if __name__ == '__main__':
    main()
