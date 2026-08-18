"""
main_RDK_config.py - RDK 双摄像头主入口独立配置

职责：
- 统一放置 RDK 主入口所需的全部默认参数；
- main_RDK.py 只负责读取并覆盖配置，不负责写参数逻辑；
- 不依赖 main_front / main_bottom 的测试配置，避免两套配置互相干扰。
"""

# ================================================================
# 默认配置
# ================================================================

DEFAULT_CONFIG = {
    # ==========================================
    # 一、输入源配置
    # ==========================================
    'front_source': 0,                  # 前视摄像头编号或视频文件路径，例如 0 / front.mp4
    'bottom_source': 1,                 # 下视摄像头编号或视频文件路径，例如 1 / bottom.mp4
    'front_target': None,               # 前视(P2 撞球)仅保留类别；例如 'red'；None=不限制(检测全部)
    'bottom_target': None,              # 下视(P4 抓球)仅保留类别；例如 'red'；None=不限制(检测全部)
    'work_width': 640,                  # 工作分辨率 - 宽
    'work_height': 480,                 # 工作分辨率 - 高

    # ==========================================
    # 二、通用检测参数
    # ==========================================
    'conf': 0.25,                       # 检测置信度阈值
    'iou': 0.45,                        # NMS IoU 阈值
    'imgsz': 640,                       # 推理分辨率
    'show': True,                       # 是否显示窗口
    'front_output': None,               # 前视结果视频保存路径
    'bottom_output': None,              # 下视结果视频保存路径

    # ==========================================
    # 三、P2/P4 球检测配置（RDK 板端 HBM）
    # ==========================================
    'p2_model_path': None,              # HBM 模型路径；None 时默认使用 P2 撞球目录下的模型
    'p2_label_file': None,              # P2 类别名称文件（每行一个类别名），按类别名过滤时必填
    'p2_score_thres': 0.25,             # 置信度阈值
    'p2_nms_thres': 0.45,               # NMS IoU 阈值
    'p2_priority': 0,                   # 推理优先级
    'p2_target_class_names': [],        # P2 目标类别名称列表；空列表=不限制(检测全部)
                                        # 例如 ['red'] 只检测红色球；设置后需提供 p2_label_file
    'p2_target_class_ids': [],          # P2 目标类别 id；优先于名称过滤，无需 label 文件
    'p4_label_file': None,              # P4 类别名称文件（每行一个类别名），按类别名过滤时必填
    'p4_target_class_names': [],        # P4 目标类别名称列表；空列表=不限制(检测全部)
                                        # 例如 ['red'] 只检测红色球；设置后需提供 p4_label_file
    'p4_target_class_ids': [],          # P4 目标类别 id；优先于名称过滤，无需 label 文件

    # ==========================================
    # 四、P1 / P3 / P5 任务覆盖参数
    # 说明：
    #   - p1_*: 下视巡线任务参数，控制引导线分割和角度拟合
    #   - p3_*: 前视门框检测参数，控制红色门框的轮廓筛选与矩形验证
    #   - p5_*: 下视收集框参数，控制放球区域的红色矩形框检测
    # ==========================================
    # ---- P1 巡线参数（下视） ----
    'p1_enable_resize': False,                 # 是否在 P1 巡线前先缩放图像，False 表示已由上层统一尺寸
    'p1_enable_color_correct': False,          # 是否启用水下颜色校正，False 可避免巡线前增强偏差
    'p1_red_boost': 1.2,                       # 红通道增强系数，增强调整红色引导线对比度
    'p1_enable_gaussian': False,               # 是否做高斯去噪，False 代表较少平滑以保留线条边缘
    'p1_gaussian_kernel': 5,                   # 高斯核大小，影响边缘去噪强度
    'p1_enable_clahe': True,                   # 是否启用 CLAHE 直方图均衡，提升低对比度水下图像
    'p1_clahe_clip': 2.0,                      # CLAHE 限制对比度扩展程度，过大容易增强噪声
    'p1_clahe_tile': (8, 8),                   # CLAHE 分块大小，影响局部对比增强强度
    'p1_hsv_lower': [0, 80, 80],               # P1 引导线橙红色 HSV 下界，控制低阈值分割范围
    'p1_hsv_upper': [25, 255, 255],            # P1 引导线橙红色 HSV 上界，控制高阈值分割范围
    'p1_hsv_lower2': [150, 80, 80],            # P1 红色跨 H 通道的下界，用于处理 H=180 附近红色分段
    'p1_hsv_upper2': [179, 255, 255],          # P1 红色跨 H 通道的上界
    'p1_enable_red_wrap': True,                 # 是否启用红色跨边界检测，适配红色环绕 H=179/0 处
    'p1_morph_open_kernel': 3,                 # 开运算核大小，用于去除小噪点
    'p1_morph_close_kernel': 7,                # 闭运算核大小，用于连接断裂线条
    'p1_min_contour_area': 500,                # 最小轮廓面积阈值，过滤小噪声区域
    'p1_enable_angle_debounce': True,           # 是否启用角度消抖，减少抖动导致的漂移
    'p1_angle_deadzone': 3.0,                  # 角度死区，绝对值小于该值视为 0，减少小震动
    'p1_angle_smooth_window': 5,               # 角度平滑窗口大小，越大越稳定但越慢响应

    # ---- P3 穿门参数（前视） ----
    'p3_hsv_lower': [0, 80, 80],               # P3 门框红色 HSV 下界，过滤非门框区域
    'p3_hsv_upper': [25, 255, 255],            # P3 门框红色 HSV 上界，控制门框分割阈值
    'p3_hsv_lower2': [150, 80, 80],            # P3 红色跨 H 通道下界，兼容红色在 H=180 附近的分段
    'p3_hsv_upper2': [179, 255, 255],          # P3 红色跨 H 通道上界
    'p3_morph_open_kernel': 3,                 # 开运算核大小，去噪并去掉小斑点
    'p3_morph_close_kernel': 9,                # 闭运算核大小，填补门框断裂区域
    'p3_min_contour_area': 800,                # 最小轮廓面积阈值，过滤噪声和小遮挡块
    'p3_min_box_width_ratio': 0.15,            # 门框最小宽度占图像宽度比例，防止误检极窄区域
    'p3_max_box_width_ratio': 0.95,            # 门框最大宽度占图像宽度比例，过滤超大连通区域
    'p3_min_box_height_ratio': 0.15,           # 门框最小高度占图像高度比例
    'p3_max_box_height_ratio': 0.95,           # 门框最大高度占图像高度比例
    'p3_min_aspect_ratio': 0.4,                # 门框最小长宽比，过滤过于扁或过于高的区域
    'p3_max_aspect_ratio': 2.5,                # 门框最大长宽比，过滤极端比例目标
    'p3_edge_fill_threshold': 0.4,             # 门框边缘红色占比阈值，用于验证边界是否闭合
    'p3_interior_empty_threshold': 0.3,        # 内部空白比例阈值，确保门框内部非全红导致误判
    'p3_edge_inset_ratio': 0.04,               # 边界采样时向内收缩的比例，减少边缘抗锯齿干扰
    'p3_interior_shrink_ratio': 0.6,           # 内部检测采样收缩比例，判断门框中心是否为空
    'p3_aligned_ratio': 0.15,                  # 门框对齐度阈值，控制门框是否接近矩形
    'p3_door_top_norm_threshold': 0.5,         # 门框顶部归一化阈值，用于判断门高/矮的门型分类

    # ---- P5 放球参数（下视） ----
    'p5_enable_resize': False,                  # 是否在 P5 放球前重复缩放，通常为 False 以保持分辨率统一
    'p5_enable_color_correct': False,          # 是否启用水下颜色校正，保持与 P1 一致的低干扰处理
    'p5_red_boost': 1.2,                       # 红色通道增强系数，提升收集框检测的红色对比
    'p5_enable_gaussian': False,               # 是否对框检测前景做高斯去噪，False 可减少误平滑
    'p5_gaussian_kernel': 5,                   # 高斯去噪核大小，影响模糊程度与噪声控制
    'p5_enable_clahe': True,                   # 是否启用 CLAHE 提升低对比水底图像
    'p5_clahe_clip': 2.0,                      # CLAHE 限制对比度扩展，平衡增强与噪声
    'p5_clahe_tile': (8, 8),                   # CLAHE 分块大小，影响局部暗光补偿效果
    'p5_hsv_lower': [0, 80, 80],               # P5 放球框红色 HSV 下界，控制红色区域分割阈值
    'p5_hsv_upper': [25, 255, 255],            # P5 放球框红色 HSV 上界
    'p5_hsv_lower2': [150, 80, 80],            # P5 放球框红色跨 H 通道下界
    'p5_hsv_upper2': [179, 255, 255],          # P5 放球框红色跨 H 通道上界
    'p5_morph_open_kernel': 3,                 # 开运算核大小，去除小噪点
    'p5_morph_close_kernel': 9,                # 闭运算核大小，连接红框边缘并填补断裂
    'p5_min_contour_area': 600,                # 最小轮廓面积阈值，过滤小杂点和噪声
    'p5_min_box_width_ratio': 0.10,            # 放球框最小宽度占图像宽度比例
    'p5_max_box_width_ratio': 0.90,            # 放球框最大宽度占图像宽度比例
    'p5_min_box_height_ratio': 0.10,           # 放球框最小高度占图像高度比例
    'p5_max_box_height_ratio': 0.95,           # 放球框最大高度占图像高度比例
    'p5_min_aspect_ratio': 0.4,                # 放球框最小长宽比，过滤过短或过扁区域
    'p5_max_aspect_ratio': 2.5,                # 放球框最大长宽比，过滤极端比例目标
    'p5_edge_fill_threshold': 0.4,             # 边缘红色填充率阈值，用于判定是否形成闭合框
    'p5_interior_empty_threshold': 0.3,        # 内部空白比例阈值，保证框内不是全红噪声区域
    'p5_edge_inset_ratio': 0.04,               # 框边缘采样内缩比例，减少边界噪声干扰
    'p5_interior_shrink_ratio': 0.6,           # 内部采样区域缩放比例，控制“心形”/中心空洞判断
    'p5_aligned_ratio': 0.15,                  # 框对齐比例阈值，控制矩形框整齐程度

    # ==========================================
    # 五、各任务线程 0.预处理 总开关（独立控制）
    # True = 该任务线程先执行共享的 0.预处理（缩放/颜色校正/高斯/CLAHE）再检测
    # False = 跳过预处理，直接使用原始帧
    # 各任务预处理的分步参数（如 p1_enable_clahe）见上文对应任务小节；
    # 未提供的任务（p2/p3/p4）使用 0.预处理 模块默认值，可按需添加同名键覆盖
    # ==========================================
    'p1_enable_preprocess': False,   # P1 巡线（下视）
    'p2_enable_preprocess': False,   # P2 撞球（前视）
    'p3_enable_preprocess': False,   # P3 穿门（前视）
    'p4_enable_preprocess': False,   # P4 抓球（下视）
    'p5_enable_preprocess': False,   # P5 放球（下视）
}


# ================================================================
# 辅助函数
# ================================================================

def get_config(overrides=None):
    """返回配置副本，保持主函数只作调用。"""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides is not None:
        cfg.update(copy.deepcopy(overrides))
    return cfg

