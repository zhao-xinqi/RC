'''
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
  '''

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
