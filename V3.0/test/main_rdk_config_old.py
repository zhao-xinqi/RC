"""
main_rdk_config.py - RDK 板端双摄像头功能测试 可配置参数

所有参数集中管理, 便于阅读与调优。
参数分为四组:
  1. 输入源配置     - 前视/下视摄像头编号、工作分辨率
  2. P2/P4 球检测   - 复用 P2 撞球/main.py 的 create_detector, 必须使用 hbm 模型
  3. 显示与输出配置  - 检测窗口/前视与下视结果视频保存

说明:
  - P1/P3/P5 的 overrides 不在此重复维护, 由 main_RDK.build_config
    从 main_front_config / main_bottom_config 中取用 (避免多份配置漂移).
  - 球检测固定走 hbm (RDK BPU), 不使用 best.pt; 详见 main_RDK.py 的 RDKBallDetector.
"""

# ================================================================
# 默认配置
# ================================================================

DEFAULT_CONFIG = {

    # ==========================================
    # 一、输入源配置
    # ==========================================
    'front_source': 0,             # 前视摄像头编号 (0/1/...) 或视频文件路径
    'bottom_source': 1,            # 下视摄像头编号 (0/1/...) 或视频文件路径
    'work_width': 640,             # 工作分辨率 - 宽 (与摄像头采集一致)
    'work_height': 480,            # 工作分辨率 - 高

    # ==========================================
    # 二、P2/P4 球检测配置 (hbm, 复用 P2 撞球/main.py)
    # ==========================================

    # -- 模型与标签 (默认模型位于 P2 撞球 目录) --
    'p2_model_path': None,         # hbm 模型路径; None 时取 P2 撞球/best_nashe_320x320_nv12.hbm
    'p2_label_file': None,         # hbm 后端类别名称文件 (每行一个类别名)

    # -- 推理参数 --
    'p2_score_thres': 0.25,        # 置信度阈值
    'p2_nms_thres': 0.45,          # NMS 非极大值抑制 IoU 阈值

    # -- BPU 调度 (前视/下视各一个模型实例, 分配不同核心避免争抢) --
    'p2_priority': 0,              # 推理优先级 (0~255), 0 最低, 255 最高
    'p2_bpu_cores_front': [0],     # 前视球检测使用的 BPU 核心索引
    'p2_bpu_cores_bottom': [1],    # 下视球检测使用的 BPU 核心索引

    # -- 目标过滤 --
    'p2_target_class_names': ['red'],  # 只保留的类别名称, 例如 ['red']; 空列表保留全部
    'p2_target_class_ids': [],         # 目标类别 id (优先于名称), 例如 [0]

    # ==========================================
    # 三、显示与输出配置
    # ==========================================
    'show': True,                  # 是否显示检测窗口 (RDK 无显示时可关闭)
    'front_output': None,          # 前视结果视频保存路径; None 不保存
    'bottom_output': None,         # 下视结果视频保存路径; None 不保存
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
