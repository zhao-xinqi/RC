"""
main_bottom_config.py - 下视摄像头功能测试 可配置参数

所有参数集中管理, 便于阅读与调优。
参数分为五组:
  1. 输入源配置    - 视频文件/摄像头编号、工作分辨率
  2. P1 巡线配置   - 对 P1 巡线/config.py 的参数覆盖 (池底橙红色引导线)
  3. P4 抓球配置   - 完全复用 P2 撞球的球检测 (参数与 main_front 的 p2_* 一致)
  4. P5 放球配置   - 对 P5 放球/config.py 的参数覆盖 (红色闭合矩形收集框)
  5. 显示与输出配置 - 检测窗口/结果视频保存

说明: 三个任务的检测器内部都会调用 0.预处理.preprocess() 做水下预处理,
      P1/P5 的预处理参数可在各自的 overrides 字典中调优 (键名与各任务 config.py 一致).
"""

# ================================================================
# 默认配置
# ================================================================

DEFAULT_CONFIG = {

    # ==========================================
    # 一、输入源配置
    # ==========================================
    'source': 'D:\\RC\\test.mp4',    # 下视摄像头视频文件路径或摄像头编号, 例如 0 / bottom.mp4
    'work_width': 640,             # 工作分辨率 - 宽 (与摄像头采集一致)
    'work_height': 480,            # 工作分辨率 - 高

    # ==========================================
    # 二、P1 巡线检测配置
    # (池底橙红色引导线: HSV 分割 + 最小二乘拟合, 输出偏移/角度;
    #  覆盖 P1 巡线/config.py 的参数, 未列出的使用其默认值)
    # ==========================================
    'p1_overrides': {
        # -- 0.预处理 参数 (覆盖 P1 巡线/config.py 默认值, 与 P4/P5 水下默认对齐) --
        'enable_resize': False,          # 帧已统一缩放到工作分辨率, 不重复缩放
        'enable_color_correct': False,    # 水下颜色校正 (红光补偿)
        'red_boost': 1.2,                # 红色通道增强系数 (>1.0 增强红色)
        'enable_gaussian': False,         # 高斯去噪
        'gaussian_kernel': 5,            # 高斯核大小 (奇数)
        'enable_clahe': True,            # CLAHE 增强 (应对水下光照不均)
        'clahe_clip': 2.0,               # CLAHE 对比度限幅
        'clahe_tile': (8, 8),            # CLAHE 网格大小

        # -- 颜色分割参数 --
        'hsv_lower': [0, 80, 80],        # 引导线橙红色 HSV 下界 [H, S, V]
        'hsv_upper': [25, 255, 255],     # 引导线橙红色 HSV 上界 [H, S, V]
        'hsv_lower2': [150, 80, 80],     # 红色跨边界段下界
        'hsv_upper2': [179, 255, 255],   # 红色跨边界段上界
        'enable_red_wrap': True,         # 是否启用红色跨边界检测
        'morph_open_kernel': 3,          # 开运算核 (去噪)
        'morph_close_kernel': 7,         # 闭运算核 (弥合引导线断裂)
        'min_contour_area': 500,         # 最小轮廓面积 (像素²)
        'enable_angle_debounce': True,   # 角度消抖总开关
        'angle_deadzone': 3.0,           # 小角度死区 (度), |angle| 小于此值视为 0
        'angle_smooth_window': 5,        # 滑动窗口大小 (帧)
    },

    # ==========================================
    # 三、P4 抓球检测配置
    # (完全复用 P2 撞球的球检测, 参数与 main_front 的 p2_* 语义一致)
    # ==========================================

    # -- 推理后端与模型 (默认模型位于 P2 撞球 目录) --
    'p4_backend': 'ultralytics',   # 球检测后端: ultralytics(.pt 开发机) / hbm(.hbm RDK 板端)
    'p4_model': None,              # 模型路径; None 时按后端取 P2 撞球 下的默认模型
    'p4_default_model_pt': 'best.pt',               # ultralytics 后端默认模型
    'p4_default_model_hbm': 'best_nashe_320x320_nv12.hbm',  # hbm 后端默认模型
    'p4_label_file': None,         # hbm 后端类别名称文件 (每行一个类别名)

    # -- 推理参数 --
    'p4_conf': 0.25,               # 置信度阈值
    'p4_iou': 0.45,                # NMS 非极大值抑制 IoU 阈值
    'p4_imgsz': 640,               # ultralytics 后端推理分辨率

    # -- 目标过滤 --
    'p4_target_classes': [],       # 只保留的目标类别, 例如 ['red']; 空列表保留全部

    # -- P4 水下预处理 (0.预处理, 复用 P2 逻辑) --
    'p4_enable_preprocess': True,  # 是否在球检测前做水下预处理
    'preprocess_width': None,      # 预处理目标宽; None 时跟随工作分辨率
    'preprocess_height': None,     # 预处理目标高; None 时跟随工作分辨率
    'preprocess_enable_resize': True,          # 是否缩放
    'preprocess_enable_color_correct': True,   # 是否水下颜色校正
    'preprocess_red_boost': 1.2,   # 红色通道增强系数 (>1.0 增强红色, 利于红球检出)
    'preprocess_enable_gaussian': True,        # 是否高斯去噪
    'preprocess_gaussian_kernel': 5,           # 高斯核大小 (奇数)
    'preprocess_enable_clahe': True,           # 是否 CLAHE 增强
    'preprocess_clahe_clip': 2.0,              # CLAHE 对比度限幅
    'preprocess_clahe_tile': (8, 8),           # CLAHE 网格大小

    # ==========================================
    # 四、P5 放球检测配置
    # (池底红色闭合矩形收集框, 与 P3 穿门同为 BoxDetector 内核;
    #  覆盖 P5 放球/config.py 的参数, 未列出的使用其默认值)
    # ==========================================
    'p5_overrides': {
        # -- 0.预处理 参数 (覆盖 P5 放球/config.py 默认值, 与 P4 水下默认对齐) --
        'enable_resize': False,          # 帧已统一缩放到工作分辨率, 不重复缩放
        'enable_color_correct': False,    # 水下颜色校正 (红光补偿)
        'red_boost': 1.2,                # 红色通道增强系数 (>1.0 增强红色)
        'enable_gaussian': False,         # 高斯去噪
        'gaussian_kernel': 5,            # 高斯核大小 (奇数)
        'enable_clahe': True,            # CLAHE 增强
        'clahe_clip': 2.0,               # CLAHE 对比度限幅
        'clahe_tile': (8, 8),            # CLAHE 网格大小

        # -- 颜色分割参数 --
        'hsv_lower': [0, 80, 80],        # 红色 HSV 下界 [H, S, V]
        'hsv_upper': [25, 255, 255],     # 红色 HSV 上界 [H, S, V]
        'hsv_lower2': [150, 80, 80],     # 红色跨边界段下界
        'hsv_upper2': [179, 255, 255],   # 红色跨边界段上界
        'morph_open_kernel': 3,          # 开运算核 (去噪)
        'morph_close_kernel': 9,         # 闭运算核 (弥合边框断裂)
        'min_contour_area': 600,         # 最小轮廓面积 (像素²)
        'min_box_width_ratio': 0.10,     # 框宽最小占比
        'max_box_width_ratio': 0.90,     # 框宽最大占比
        'min_box_height_ratio': 0.10,    # 框高最小占比
        'max_box_height_ratio': 0.95,    # 框高最大占比
        'min_aspect_ratio': 0.4,         # 框宽高比下界
        'max_aspect_ratio': 2.5,         # 框宽高比上界
        'edge_fill_threshold': 0.4,      # 四边红色占比 (闭合矩形验证)
        'interior_empty_threshold': 0.3, # 内部空心红色占比上限
        'edge_inset_ratio': 0.04,        # 边采样角点内缩比例
        'interior_shrink_ratio': 0.6,    # 内部区域角点收缩比例
        'aligned_ratio': 0.15,           # 对齐判定阈值 (归一化)
    },

    # ==========================================
    # 五、显示与输出配置
    # ==========================================
    'show': True,                  # 是否显示检测窗口
    'output': None,                # 结果视频保存路径; None 不保存
}


# ================================================================
# 辅助函数
# ================================================================

def get_config(overrides=None):
    """获取配置副本 (避免修改默认配置的引用)。

    Args:
        overrides: dict, 需要覆盖的参数 (可选)

    Returns:
        dict: 配置字典的深拷贝
    """
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides is not None:
        cfg.update(copy.deepcopy(overrides))
    return cfg
