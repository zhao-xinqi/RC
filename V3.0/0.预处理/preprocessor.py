"""
通用水下图像预处理函数

适用于水下机器人所有视觉任务的预处理流水线:
  1. 缩放至目标分辨率
  2. 水下颜色校正 (补偿红光衰减)
  3. 高斯去噪
  4. CLAHE 自适应直方图均衡化

各步骤可独立开关, 参数可外部配置.
"""

import cv2
import numpy as np


def preprocess(frame, config=None):
    """
    通用水下图像预处理

    Args:
        frame: 输入图像 (BGR, numpy ndarray)
        config: 配置字典, 支持以下键 (未提供的键使用默认值):

            预处理总控:
            - image_width:   int, 目标宽度, 默认 640
            - image_height:  int, 目标高度, 默认 480
            - enable_resize: bool, 是否缩放, 默认 True

            颜色校正:
            - enable_color_correct: bool, 是否颜色校正, 默认 True
            - red_boost:     float, 红色通道增强系数, 默认 1.2
                             (水下红光衰减严重, 适当增强红色补偿)

            高斯去噪:
            - enable_gaussian:  bool, 是否高斯去噪, 默认 True
            - gaussian_kernel:  int, 高斯核大小 (奇数), 默认 5

            CLAHE:
            - enable_clahe:  bool, 是否 CLAHE 增强, 默认 True
            - clahe_clip:    float, CLAHE 对比度限幅, 默认 2.0
            - clahe_tile:    tuple, CLAHE 网格大小, 默认 (8, 8)

    Returns:
        processed: 预处理后的 BGR 图像 (numpy ndarray)
    """

    # ---- 合并默认配置 ----
    cfg = {
        # 缩放
        'image_width': 640,
        'image_height': 480,
        'enable_resize': True,
        # 颜色校正
        'enable_color_correct': True,
        'red_boost': 1.2,
        # 高斯去噪
        'enable_gaussian': True,
        'gaussian_kernel': 5,
        # CLAHE
        'enable_clahe': True,
        'clahe_clip': 2.0,
        'clahe_tile': (8, 8),
    }

    #修正配置参数
    if config is not None:
        cfg.update(config)

    result = frame.copy()


    # ============================================================
    # 步骤 1: 缩放
    # ============================================================
    if cfg['enable_resize']:
        '''
        该区域内代码可能需要使用，取决于摄像头实际分辨率
        如果摄像头分辨率设置为640*480，则无需缩放，直接使用原图即可
        因此以下补充一个判断条件，如果摄像头分辨率为640*480，则不进行缩放

        h, w = result.shape[:2]
        target_w, target_h = cfg['image_width'], cfg['image_height']

        # 尺寸一致则跳过，避免无意义拷贝
        if w != target_w or h != target_h:
            # 缩小用 INTER_AREA，放大用 INTER_LINEAR
            if target_w < w or target_h < h:
                interp = cv2.INTER_AREA
            else:
                interp = cv2.INTER_LINEAR
        '''
        result = cv2.resize(
            result,
            (cfg['image_width'], cfg['image_height']),
            interpolation=cv2.INTER_LINEAR,   # 线性插值，可能需要变化
        )

    # ============================================================
    # 步骤 2: 水下颜色校正
    # ============================================================
    if cfg['enable_color_correct']:
        result = _underwater_color_correct(result, cfg['red_boost'])

    # ============================================================
    # 步骤 3: 高斯去噪
    # ============================================================
    if cfg['enable_gaussian']:
        ksize = cfg['gaussian_kernel']
        if ksize % 2 == 0:
            ksize += 1  # 确保为奇数
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    # ============================================================
    # 步骤 4: CLAHE 自适应直方图均衡化
    # ============================================================
    if cfg['enable_clahe']:
        result = _apply_clahe(
            result,
            clip_limit=cfg['clahe_clip'],
            tile_grid_size=cfg['clahe_tile'],
        )

    return result


# ================================================================
# 内部辅助函数
# ================================================================

def _underwater_color_correct(frame, red_boost=1.2):
    """
    水下颜色校正

    原理: 水体对红光的吸收远大于蓝绿光, 导致水下图像偏蓝绿.
    本函数使用 "灰度世界假设" 做白平衡, 并额外增强红色通道.

    Args:
        frame:     BGR 图像 (uint8)
        red_boost: 红色通道增强倍数, >1.0 增强, <1.0 抑制

    Returns:
        校正后的 BGR 图像 (uint8)
    """
    # 分离通道并转为 float32 避免溢出
    b, g, r = cv2.split(frame.astype(np.float32))

    # 灰度世界假设: 场景平均反射应为中性灰
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    gray_mean = (mean_b + mean_g + mean_r) / 3.0

    # 按灰度均值缩放各通道
    if mean_b > 1e-6:
        b = b * (gray_mean / mean_b)
    if mean_g > 1e-6:
        g = g * (gray_mean / mean_g)
    if mean_r > 1e-6:
        r = r * (gray_mean / mean_r) * red_boost

    # 裁剪到 [0, 255] 并转回 uint8
    b = np.clip(b, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    r = np.clip(r, 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])


def _apply_clahe(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    自适应直方图均衡化 (CLAHE)

    在 LAB 色彩空间的 L (亮度) 通道上操作, 保持色彩不失真.
    用于应对水下光照不均匀的情况.

    Args:
        frame:         BGR 图像
        clip_limit:    对比度限幅阈值, 越大对比度越强
        tile_grid_size: 网格划分, 越小局部性越强

    Returns:
        增强后的 BGR 图像
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
