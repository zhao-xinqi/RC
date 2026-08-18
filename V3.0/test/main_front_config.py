"""
main_front_config.py - 前视摄像头功能测试 可配置参数

所有参数集中管理, 便于阅读与调优。
参数分为四组:
  1. 输入源配置    - 视频文件/摄像头编号、工作分辨率
  2. P2 球检测配置 - 推理后端/模型/置信度/目标过滤/水下预处理
  3. P3 门检测配置 - 对 P3 穿门/config.py 的参数覆盖
  4. 显示与输出配置 - 检测窗口/结果视频保存
"""

# ================================================================
# 默认配置
# ================================================================

DEFAULT_CONFIG = {

    # ==========================================
    # 一、输入源配置
    # ==========================================
    'source': 'D:\\RC\\test.mp4',                 # 视频文件路径或摄像头编号, 例如 0 / test.mp4
    'work_width': 640,             # 工作分辨率 - 宽 (与摄像头采集一致)
    'work_height': 480,            # 工作分辨率 - 高

    # ==========================================
    # 二、P2 撞球检测配置
    # ==========================================

    # -- 推理后端与模型 --
    'p2_backend': 'ultralytics',   # 球检测后端: ultralytics(.pt 开发机) / hbm(.hbm RDK 板端)
    'p2_model': None,              # 模型路径; None 时按后端自动取默认模型 (见 p2_default_model_*)
    'p2_default_model_pt': 'best.pt',               # ultralytics 后端默认模型
    'p2_default_model_hbm': 'best_nashe_320x320_nv12.hbm',  # hbm 后端默认模型
    'p2_label_file': None,         # hbm 后端类别名称文件 (每行一个类别名)

    # -- 推理参数 --
    'p2_conf': 0.25,               # 置信度阈值
    'p2_iou': 0.45,                # NMS 非极大值抑制 IoU 阈值
    'p2_imgsz': 640,               # ultralytics 后端推理分辨率

    # -- 目标过滤 --
    'p2_target_classes': [],       # 只保留的目标类别, 例如 ['red']; 空列表保留全部

    # -- P2 水下预处理 (0.预处理) --
    'p2_enable_preprocess': True,  # 是否在球检测前做水下预处理
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
    # 三、P3 穿门检测配置
    # (门框为红色闭合矩形, 复用 P5 BoxDetector; 覆盖 P3 config.py 的参数,
    #  未列出的使用其默认值)
    # ==========================================
    'p3_overrides': {
        'hsv_lower': [0, 80, 80],        # 红色 HSV 下界 [H, S, V]
        'hsv_upper': [25, 255, 255],     # 红色 HSV 上界 [H, S, V]
        'hsv_lower2': [150, 80, 80],     # 红色跨边界段下界
        'hsv_upper2': [179, 255, 255],   # 红色跨边界段上界
        'morph_open_kernel': 3,          # 开运算核 (去噪)
        'morph_close_kernel': 9,         # 闭运算核 (弥合门框断裂)
        'min_contour_area': 800,         # 最小轮廓面积 (像素²)
        'min_box_width_ratio': 0.15,     # 门宽最小占比
        'max_box_width_ratio': 0.95,     # 门宽最大占比
        'min_box_height_ratio': 0.15,    # 门高最小占比
        'max_box_height_ratio': 0.95,    # 门高最大占比
        'min_aspect_ratio': 0.4,         # 门宽高比下界
        'max_aspect_ratio': 2.5,         # 门宽高比上界
        'edge_fill_threshold': 0.4,      # 四边红色占比 (闭合矩形验证)
        'interior_empty_threshold': 0.3, # 内部空心红色占比上限
        'edge_inset_ratio': 0.04,        # 边采样角点内缩比例 (避开抗锯齿边缘)
        'interior_shrink_ratio': 0.6,    # 内部区域角点收缩比例 (收缩 60% 到中心)
        'aligned_ratio': 0.15,           # 对齐判定阈值 (归一化)
        'door_top_norm_threshold': 0.5,  # 高/矮门判定: 门顶归一化 y 阈值
    },

    # ==========================================
    # 四、显示与输出配置
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
