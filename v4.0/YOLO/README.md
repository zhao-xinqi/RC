# P2 撞球 - 目标球检测（YOLO / HBM 运行时）

从前置摄像头图像中检测**目标球**，输出球框位置与置信度，供机器人控制器后续完成转向对准与撞击决策。是机器人水中巡游项目第二步任务。

检测采用 **RDK 板端 BPU（HBM 运行时）** 方案：使用训练并量化好的 `best_nashe_320x320_nv12.hbm` 模型（320×320 输入、NV12 格式），在 BPU 上完成推理，**当前阶段仅实现 detect（目标检测）**。

## 文件结构

```
P2 撞球/
├── config.py          # 集中管理全部可调参数 (预处理/YOLO检测/目标决策)
├── main.py            # detect 推理入口 (CLI, 单张图像; 可选用 0.预处理)
├── yolo_detect.py     # YOLO DFL 检测封装 (HBM 运行时, 注释已中文化)
├── utils/             # 通用工具包 (file_io/visualize/inspect/preprocess/postprocess)
│   └── py_utils/      #   ← 完全引用当前目录, 未做任何修改
└── README.md          # 本文档
```

## 检测流水线

```
原始帧 → ① (可选)水下预处理 → ② YOLO 检测(BPU) → ③ 后处理 → ④ 结果打印/可视化
```

| 步骤 | 作用 | 说明 |
|------|------|------|
| ① 预处理(可选) | 水下图像增强 | 复用 `0.预处理` 模块（缩放/颜色校正/去噪/CLAHE），由 `config.py` 中 `enable_preprocess` 控制，默认关闭 |
| ② YOLO 检测 | 目标检测 | `yolo_detect.py` 封装 HBM 运行时，输入 BGR 图像，内部转 NV12 后送入 BPU |
| ③ 后处理 | 解码+过滤+NMS | 无锚框 DFL 框解码 → 置信度过滤 → 非极大值抑制（NMS）→ 坐标映射回原图（复用 `utils/py_utils/postprocess.py`） |
| ④ 结果输出 | 打印+可视化 | `visualize.print_detections` 打印，`draw_boxes` 叠加绘制，保存结果图 |

## 参数配置 (config.py)

所有参数集中在 [config.py](config.py) 的 `DEFAULT_CONFIG` 中管理，通过 `get_config()` 获取深拷贝副本（避免修改默认配置）。参数分三组：

### 一、预处理参数（传递给 `0.预处理.preprocess`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_preprocess` | False | 是否在 YOLO 推理前先做水下预处理（默认关闭；开启后自动调用 `0.预处理`，集成方式参考 P1 巡线） |
| `image_width` / `image_height` | 640 / 480 | 处理分辨率 |
| `enable_resize` | False | 是否缩放至目标分辨率 |
| `enable_color_correct` / `red_boost` | False / 1.2 | 水下颜色校正开关 / 红色增强系数 |
| `enable_gaussian` / `gaussian_kernel` | False / 5 | 高斯去噪开关 / 核大小 |
| `enable_clahe` / `clahe_clip` / `clahe_tile` | True / 2.0 / (8,8) | CLAHE 增强开关 / 对比度限幅 / 网格大小 |

### 二、YOLO 检测参数（detect 任务）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `best_nashe_320x320_nv12.hbm` | BPU 量化模型路径（当前目录；支持相对/绝对路径） |
| `score_thres` | 0.25 | 置信度阈值，过滤检测结果 |
| `nms_thres` | 0.45 | NMS 非极大值抑制的 IoU 阈值 |
| `strides` | `[8, 16, 32]` | 检测头各尺度下采样倍率（与模型结构一致，一般不改） |
| `priority` | 0 | BPU 推理优先级（0~255） |
| `bpu_cores` | `[0]` | BPU 核心索引列表（多核可设 `[0, 1]`） |

> **anchor_sizes 说明**：各尺度特征图网格大小由 `main.py` 根据模型**实际输入分辨率**自动计算（320×320 → `[40,20,10]`，640×640 → `[80,40,20]`），无需手动配置。若自行更换模型分辨率，无需改任何参数。

### 三、目标决策参数（后续撞球控制阶段使用）

为后续目标选择与撞击决策预留，当前 detect 阶段尚未使用：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_class_names` / `target_class_ids` | `['ball']` / `[]` | 目标球类别过滤（名称或 id，二选一） |
| `target_strategy` | `largest` | 多球时目标选择：`largest`(最近) / `center`(最居中) |
| `center_zone_ratio` | 0.12 | 居中判定：球心偏差占画面宽度比例 |
| `strike_width_px` | 150 | 撞击就绪：球框宽达到此像素即认为足够近 |
| `enable_distance` / `focal_length_px` / `ball_real_diameter_cm` | True / 554 / 10 | 距离估算（针孔模型，需标定焦距） |

## 使用方式

### 单独运行（单张图像 detect 推理）

```
python main.py --test-img test.jpg                              # 基本用法
python main.py --test-img test.jpg --score-thres 0.3            # 覆盖置信度阈值
python main.py --test-img test.jpg --label-file ball.names      # 指定类别名称文件
python main.py --test-img test.jpg --enable-preprocess          # 强制开启水下预处理
```

命令行参数可覆盖 `config.py` 中的同名默认值；未指定的参数取 `config.py` 配置。

> **注意**：`main.py` 依赖 `hbm_runtime`，需在 RDK 板端运行；`utils` 仅用于推理后处理与可视化，全部复用当前目录 `./utils`。

## 0.预处理 集成说明

与 P1 巡线相同的集成方式（`importlib` 动态加载，处理中文文件夹名）：

```python
# main.py 内（已实现）
_PREP_PATH = os.path.join(_PROJ_ROOT, "0.预处理", "preprocessor.py")
# ... importlib 加载 ...
img = preprocess(img, cfg)   # cfg 即 config.py 配置，仅取其预处理相关键
```

- 默认 `enable_preprocess=False`：检测直接使用原始帧，避免预处理干扰 BPU 量化输入。
- 开启后若发现 YOLO 检测率下降，改回 `False` 即可，不影响其它配置。

## 与主程序集成

当前 `visual_main.py` 通过 `_load_task_module("P2 撞球", ...)` 动态加载本模块。本模块提供两个入口：

- **离线/联调**：直接运行 `main.py`（单张图像 detect）。
- **在线（后续）**：撞球控制阶段将在本模块内新增检测器封装（目标过滤 → 目标选择 → 转向/撞击决策），复用 config 中**三、目标决策参数**，集成方式参考 P1 巡线的 `LineDetector`。

```
python visual_main.py --task P2        # 仅运行撞球任务
python visual_main.py --no-display     # 无窗口模式 (仅打印结果)
```

## 降级策略（规划）

当某一帧未检测到目标球（遮挡、出水、暂时出画面）时：
- 结果中附带 `last_valid` 字段（上一帧有效结果），供控制器做短时保持/平滑
- 连续多帧未检测到时，应让控制器执行搜索逻辑（如旋转寻找），该逻辑在控制端实现
