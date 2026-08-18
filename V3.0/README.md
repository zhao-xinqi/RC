# V3.0 - 视觉算法 (RoboMaster 水下机器人巡游)

本目录为水下机器人巡游任务的视觉算法工程，任务划分与整体集成方式见 [CLAUDE.md](../CLAUDE.md)。

**工具定位**：`main_front.py` / `main_bottom.py` 是**开发机（PC）**调参工具，分别测试前视/下视单路摄像头；
**`main_RDK.py` 是 RDK 板端部署入口**，同时驱动前视 + 下视两个摄像头、各用一个窗口显示，球检测走 hbm 模型。

## 目录结构

```
V3.0/
├── main.py                  # 主程序入口 (按比赛阶段 TASK_STAGE 调度各任务)
├── visual_main.py           # 各任务模块集成入口 (见 CLAUDE.md)
├── main_front.py            # 前视摄像头双任务并发测试 (P2 撞球 + P3 穿门)
├── main_front_config.py     # 前视测试的集中配置 (main_front.py 参数唯一来源)
├── main_bottom.py           # 下视摄像头三任务并发测试 (P1 巡线 + P4 抓球 + P5 放球)
├── main_bottom_config.py    # 下视测试的集中配置 (main_bottom.py 参数唯一来源)
├── main_RDK.py              # RDK 板端双摄像头测试 (前视+下视, 球检测走 hbm)
├── main_rdk_config.py       # RDK 测试的集中配置 (main_RDK.py 参数唯一来源)
├── README.md                # 本文档
├── 0.预处理/                # 水下图增强 (缩放/颜色校正/去噪/CLAHE)
├── P1 巡线/                 # 巡线
├── P2 撞球/                 # 球检测 (YOLO)
├── P3 穿门/                 # 门检测 (红色方形门框, 复用 P5 BoxDetector)
├── P5 放球/                 # 放球 (闭合矩形 BoxDetector, 被 P3 复用)
└── ...
```

> 注：P2、P3、P5 文件夹名含数字与中文，无法用标准 `import` 语句，各模块通过
> `importlib.util.spec_from_file_location` 按路径动态加载（见各模块 `__init__` / 调用方）。

## main_front.py - 前视摄像头双任务并发测试

以前视摄像头视频（或视频文件）为输入源，**多线程 + 帧级同步**同时调用
**P2 撞球**（球检测）与 **P3 穿门**（门检测），拼接显示
「原始画面 | P2 球检测 | P3 门检测」三面板，验证前视摄像头下的识别效果。

### 运行

```bash
python main_front.py --source D:\RC\test.mp4      # 视频文件
python main_front.py --source 0                   # 使用摄像头 0
python main_front.py --source test.mp4 --target red   # 只保留 red 类别
python main_front.py --source test.mp4 --output out.mp4  # 保存结果视频
python main_front.py --source test.mp4 --backend hbm    # RDK 板端 BPU 推理
python main_front.py --source test.mp4 --no-show        # 无窗口 (仅控制台打印)
```

### 线程模型

| 线程 | 职责 |
|------|------|
| 主线程 | 读帧 → 发布 → 回收 → 汇总显示 |
| P2 工作线程 | `BallDetector.detect(frame)` |
| P3 工作线程 | `DoorDetector.detect(frame)` |

两个 `threading.Barrier(3)` 完成**帧级同步**：
- `_barrier_frame`：主线程发布新帧后三方会合，保证 P2/P3 处理的是**同一帧**
- `_barrier_result`：P2/P3 完成检测后与主线程会合，主线程统一汇总

关闭时通过 `Event` + `Barrier.abort()` 打破阻塞，工作线程干净退出。

### 配置

所有参数集中在 `main_front_config.py`，分四组，命令行参数可覆盖同名项：

1. **输入源**：`source`（视频文件/摄像头编号）、`work_width`/`work_height`（工作分辨率 640×480）
2. **P2 球检测**：`p2_backend`（`ultralytics` / `hbm`）、模型路径、`conf`/`iou`/`imgsz`、目标类别过滤、水下预处理开关与参数
3. **P3 门检测**：`p3_overrides` 字典，覆盖 `P3 穿门/config.py` 的参数（HSV 阈值、闭合矩形判别参数等）
4. **显示与输出**：`show`、`output`（结果视频保存路径）

### 依赖

- `opencv-python`、`numpy`
- `ultralytics`（P2 默认 `ultralytics` 后端需要）
- `hbm_runtime`（仅 RDK 板端跑 `hbm` 后端时需要）

### 备注

- 输入帧自动缩放到统一工作分辨率（默认 640×480），保证各面板坐标一致。
- P2 前置可选用 `0.预处理` 水下图增强（`red_boost` 增强红球检出）。
- `hbm` 后端加载 `P2 撞球/yolo_detect.py`，特征图网格按模型输入分辨率自动计算。

## main_bottom.py - 下视摄像头三任务并发测试

以下视摄像头视频（或视频文件）为输入源，**多线程 + 帧级同步**同时调用
**P1 巡线**（引导线检测）、**P4 抓球**（球检测）与 **P5 放球**（收集框检测），
三个任务的检测结果**叠加绘制到同一帧**，通过**单个窗口**（`cv2.imshow`）显示，
验证下视摄像头下的识别效果。

> **P4 完全复用 P2**：P4 抓球直接复用 `main_front.py` 中的 `BallDetector`（P2 球检测适配器），
> 不重复实现；`main_bottom` 通过 `importlib` 加载 `main_front` 复用其适配器与通用工具。
> **P5 复用 BoxDetector 内核**：P5 放球（红色闭合矩形收集框）与 P3 穿门共用 P5 的检测内核。

### 运行

```bash
python main_bottom.py --source D:\RC\bottom.mp4       # 下视视频文件
python main_bottom.py --source 0 --target red         # 使用摄像头 0, 只保留 red 类别
python main_bottom.py --source bottom.mp4 --output out.mp4   # 保存结果视频
python main_bottom.py --source bottom.mp4 --backend hbm      # RDK 板端 BPU 推理
```

### 线程模型

| 线程 | 职责 |
|------|------|
| 主线程 | 读帧 → 发布 → 回收 → 汇总显示 |
| P1 工作线程 | `LineDetector.detect(frame)`（输出 偏移/角度） |
| P4 工作线程 | `BallDetector.detect(frame)`（复用 P2） |
| P5 工作线程 | `BoxDetector.detect(frame)`（收集框） |

两个 `threading.Barrier(4)` 完成**帧级同步**：`_barrier_frame` 保证 P1/P4/P5 处理同一帧，
`_barrier_result` 会合回收结果；关闭时 `Event` + `Barrier.abort()` 干净退出。

### 单窗口叠加显示

三个任务的检测结果绘制在**同一帧**（单窗口，无面板拼接）：

- **P1 巡线**：红色重心点 + 青色偏移箭头 + 洋红拟合方向线
- **P4 抓球**：黄色外接框 + 绿色中心点 + 类别置信度文字
- **P5 放球**：绿色框口矩形 + 中心点 + 绿色偏移箭头

左上角为三任务分行状态文字（偏移/角度/球数），左下角显示各检测线程的异常提示，
白色竖线为画面中心参考线。`--output` 保存的结果视频与工作分辨率同尺寸。

### 配置

所有参数集中在 `main_bottom_config.py`，分五组，命令行参数可覆盖同名项：

1. **输入源**：`source`、`work_width`/`work_height`
2. **P1 巡线**：`p1_overrides` 字典，覆盖 `P1 巡线/config.py` 的参数（**0.预处理**：颜色校正/高斯/CLAHE；HSV 橙红阈值、形态学核、最小面积、角度消抖）
3. **P4 抓球**：`p4_backend`（`ultralytics` / `hbm`）、模型路径、`conf`/`iou`/`imgsz`、目标类别过滤、水下预处理参数（默认模型位于 `P2 撞球/`）
4. **P5 放球**：`p5_overrides` 字典，覆盖 `P5 放球/config.py` 的参数（**0.预处理**：颜色校正/高斯/CLAHE；HSV 阈值、尺寸/宽高比范围、闭合矩形判别等）

> 三个任务的检测器内部都会调用 `0.预处理.preprocess()` 做水下预处理；
> P1/P5 的预处理参数在各自的 `overrides` 字典中集中调优（键名与各任务 `config.py` 一致），
> P4 的预处理参数为平铺的 `preprocess_*` 键。
> 当前默认：**P4 全开**（颜色校正 + 高斯去噪 + CLAHE）；**P1/P5 仅开 CLAHE**
> （颜色校正/高斯在 overrides 中默认关闭，需要时手动开启）。
5. **显示与输出**：`show`、`output`

### 依赖

与 `main_front.py` 相同：`opencv-python`、`numpy`、`ultralytics`（P4 默认后端）、`hbm_runtime`（RDK 板端）。

## main_RDK.py - RDK 板端双摄像头并发测试（部署入口）

**这是机器人在 RDK 板端实际部署运行的入口**。`main_front.py` / `main_bottom.py` 是开发机（PC）调参工具，
`main_RDK` 将二者合一：同时打开**前视 + 下视**两个摄像头，各用一个窗口并发显示检测结果，
球检测走 **hbm 模型**（RDK BPU），不使用 best.pt。

- **前视窗口**（复用 `FrontCamTester`）：原始画面 | P2 球检测 | P3 门检测 三面板拼接
- **下视窗口**（复用 `BottomCamTester`）：P1 线 + P4 球 + P5 框 单窗口叠加

### 任务与窗口对应

| 窗口 | 任务 | 检测器 |
|------|------|--------|
| 前视 | P2 撞球（球检测） | `RDKBallDetector`（hbm，复用 `P2 撞球/main.py`） |
| 前视 | P3 穿门（门检测） | `DoorDetector`（复用 P5 BoxDetector 内核） |
| 下视 | P1 巡线（引导线） | `LineDetector` |
| 下视 | P4 抓球（球检测） | `RDKBallDetector`（hbm，与 P2 相同） |
| 下视 | P5 放球（收集框） | `BoxDetector` |

> 门与收集框同为红色闭合矩形，靠比赛阶段 `TASK_STAGE` 区分，前视测试阶段不会同时出现。

### 球检测：复用 P2 撞球/main.py（hbm 路径）

`RDKBallDetector` 是本次实现的**核心适配器**，把 `P2 撞球/main.py` 的 RDK 推理路径封装成
统一的 `detect(frame)` 接口（输出 `label`/`score`/`bbox`/`center`，与两个 tester 的绘制一致）：

- 通过 **`P2.main.create_detector`** 构造检测器：**按模型实际输入分辨率自动修正 anchor_sizes**
  （320x320 → [40,20,10]），并**设置 BPU 调度参数**（priority / bpu_cores）——这是 RDK 调优后的推理路径
- 实时路径与 **`P2.main.run_live_detect`** 一致：直接 `predict(frame)`，**不额外叠加 0.预处理**
  （`YoloDetect.predict` 内部已完成 NV12 转换 + 缩放）
- 结果经 **`P2.main.filter_target_classes`** 做类别过滤（默认只保留 `red`）

球检测为**前视/下视各一个模型实例**（BPU 核心默认前视 `[0]`、下视 `[1]`），
避免两个线程并发调用同一 BPU 模型。

### 运行

```bash
python main_RDK.py --front-source 0 --bottom-source 1              # 前视摄像头0, 下视摄像头1
python main_RDK.py --front-source 0 --bottom-source 1 --target red # 只保留 red 类别
python main_RDK.py --front-source front.mp4 --bottom-source bottom.mp4 --no-show  # 无窗口, 仅打印
python main_RDK.py --front-source 0 --bottom-source 1 --model best_nashe_320x320_nv12.hbm
python main_RDK.py --front-source 0 --bottom-source 1 --front-output front.mp4 --bottom-output bottom.mp4  # 分别保存结果视频
```

### 双摄像头并发模型

| 线程 | 职责 |
|------|------|
| 前视线程 | `FrontCamTester.run()`（前视窗口: P2 + P3） |
| 下视线程 | `BottomCamTester.run()`（下视窗口: P1 + P4 + P5） |

两路**各自独立**读帧/检测/显示；任一路结束（视频播完 / 按 `q` / 异常）时，
通过 `request_stop()` 协调另一路**同时干净退出**（打破 Barrier + 置停止标志）。
`request_stop()` 为两个 tester 类新增的公共方法，向后兼容。

### 配置

所有参数集中在 `main_rdk_config.py`，分四组，命令行参数可覆盖同名项：

1. **输入源**：`front_source`、`bottom_source`（摄像头编号或视频文件路径）、`work_width`/`work_height`
2. **P2/P4 球检测（hbm）**：`p2_model_path`（默认取 `P2 撞球/best_nashe_320x320_nv12.hbm`）、`p2_label_file`、`p2_score_thres`/`p2_nms_thres`、`p2_priority`、`p2_bpu_cores_front`/`p2_bpu_cores_bottom`、`p2_target_class_names`/`p2_target_class_ids`
3. **P1/P3/P5 overrides**：不在此重复维护，由 `build_config` 从 `main_front_config` / `main_bottom_config` 取用（避免多份配置漂移）
4. **显示与输出**：`show`、`front_output`、`bottom_output`

### RDK 部署说明

- **依赖**：`opencv-python`、`numpy`、`hbm_runtime`（RDK 必需）；`P2 撞球/utils` 为推理工具包（随工程拷贝）
- **本机不可运行**：`yolo_detect.py` 模块顶层 `import hbm_runtime`，故 `main_RDK` 只能在 RDK 板端运行；
  开发机（无 hbm_runtime）仅可做语法/结构检查
- **模型**：`P2 撞球/best_nashe_320x320_nv12.hbm`（320x320 输入、NV12 格式）；更换模型时
  anchor_sizes 由 `create_detector` 按输入分辨率自动计算，无需手动配置
- **摄像头编号**：`0`/`1` 对应板端实际设备；RDK 无显示环境时加 `--no-show`
