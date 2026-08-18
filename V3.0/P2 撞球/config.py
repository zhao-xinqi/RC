"""
P2 撞球 - 可配置参数

所有参数集中管理, 便于后续根据实际水下环境调优.
参数分为三组:
  1. 预处理参数  - 传递给 0.预处理.preprocess (可选, 默认关闭)
  2. YOLO 检测参数 - 模型路径/置信度/NMS/推理调度
  3. 目标决策参数  - 后续撞球控制阶段使用 (目标选择/撞击判定/距离估算)
"""

# ================================================================
# 默认配置
# ================================================================

DEFAULT_CONFIG = {

    # ==========================================
    # 一、预处理参数 (传递给 0.预处理.preprocess)
    # ==========================================

    # 是否在 YOLO 推理前先做水下预处理
    # 若预处理导致 YOLO 检测率下降, 可设为 False 直接使用原始帧
    'enable_preprocess': False,

    # -- 缩放 --
    'image_width': 640,          # 处理分辨率 - 宽
    'image_height': 480,         # 处理分辨率 - 高
    'enable_resize': False,      # 是否缩放 (摄像头直出时通常需要)

    # -- 水下颜色校正 --
    'enable_color_correct': True,  # 是否启用水下颜色校正
    'red_boost': 1.2,              # 红色通道增强系数 (>1.0 增强红色)
                                   # 水下红光衰减严重, 适当增强红色有助于目标球检出

    # -- 高斯去噪 --
    'enable_gaussian': True,   # 是否启用高斯去噪
    'gaussian_kernel': 5,       # 高斯核大小 (奇数), 越大去噪越强但细节越少

    # -- CLAHE 增强 --
    'enable_clahe': True,       # 是否启用 CLAHE (应对水下光照不均)
    'clahe_clip': 2.0,          # 对比度限幅, 越大对比越强
    'clahe_tile': (8, 8),       # 网格大小, 越小局部性越强

    # ==========================================
    # 二、YOLO 检测参数 (detect 任务)
    # ==========================================

    # 模型路径: 当前目录下编译好的 best 球检测模型 (320x320 输入, NV12 格式)
    # 支持相对路径 (相对于本模块目录) 与绝对路径
    'model_path': 'best_nashe_320x320_nv12.hbm',

    'score_thres': 0.25,        # 置信度阈值, 用于过滤检测结果
    'nms_thres': 0.45,          # NMS 非极大值抑制的 IoU 阈值

    # 检测头各尺度特征图下采样倍率 (与模型结构一致, 一般无需修改)
    'strides': [8, 16, 32],
    # 注: 各尺度特征图网格大小 (anchor_sizes) 由 main.py 根据模型实际输入
    #     分辨率自动计算 (320x320 → [40,20,10], 640x640 → [80,40,20]),
    #     无需在此手动配置

    # -- BPU 推理调度 --
    'priority': 0,              # 模型推理优先级 (0~255), 0 最低, 255 最高
    'bpu_cores': [0],           # BPU 核心索引列表 (多核推理可设 [0, 1])

    # 参考中心点：检测框中心相对这个点的偏移会在主程序中打印
    # 以图像中的“右”为 x 正方向，“上”为 y 正方向
    # 默认值为 640x480 图像的中心位置
    'center_point': (320, 240),

    # ==========================================
    # 三、目标决策参数 (后续撞球控制阶段使用)
    # ==========================================

    # 目标球过滤: 只保留指定类别的检测框
    # 可按类别名称 (target_class_names) 或类别 id (target_class_ids) 过滤, 二选一
    # 例如：
    #   只保留 red -> target_class_names=['red']
    #   只保留 blue -> target_class_names=['blue']
    #   只保留第 0 类 -> target_class_ids=[0]
    'target_class_names': ['red'],  # 当前仅保留 red；如需 blue 可改成 ['blue']
    'target_class_ids': [],         # 目标类别 id (优先于名称)

    # 多球时目标选择策略: 'largest'(取最大框, 通常最近) / 'center'(最居中)
    'target_strategy': 'largest',

    # 撞击判定参数
    'center_zone_ratio': 0.12,  # 居中判定: 球心偏差占画面宽度比例
    'strike_width_px': 150,     # 撞击就绪: 球框宽度达到此像素即认为足够近

    # 距离估算 (针孔模型, 需标定焦距)
    'enable_distance': False,        # 是否启用距离估算
    'focal_length_px': 554,         # 相机焦距 (像素, 需标定)
    'ball_real_diameter_cm': 10,    # 目标球实际直径 (cm)
}


# ================================================================
# 辅助函数
# ================================================================

def get_config(overrides=None):
    """
    获取配置副本 (避免修改默认配置的引用)

    Args:
        overrides: dict, 需要覆盖的参数 (可选)

    Returns:
        dict: 配置字典的深拷贝

    Usage:
        cfg = get_config({'score_thres': 0.3})
    """
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides is not None:
        cfg.update(copy.deepcopy(overrides))
    return cfg
