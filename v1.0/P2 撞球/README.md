# P2 撞球 - 目标球检测与撞击决策模块

从前置摄像头图像中检测**目标球**，输出目标球位置、中心偏差（offset）、转向决策（左/中/右）、距离估算与撞击就绪标志，供机器人控制器转向并对准球体执行撞击。是机器人水中巡游项目的第二步任务。

## 文件结构

```
P2 撞球/
├── config.py          # 集中管理全部可调参数 (YOLO模型/置信度阈值/转向分区/距离估算)
├── yolo_detector.py   # YOLO 检测器封装 (ultralytics, 模型加载与推理)
├── ball_detector.py   # 检测器核心: BallDetector 类 (目标过滤/选择/撞击决策)
├── test_tool.py       # 本地调参测试工具 (滑动条实时调置信度阈值)
├── __init__.py        # 模块导出 (动态加载, 兼容中文文件夹名)
└── README.md          # 本文档
```

## 检测流水线

```
原始帧 → ① (可选)预处理 → ② YOLO 检测 → ③ 目标过滤 → ④ 目标选择 → ⑤ 转向/撞击决策
```

| 步骤 | 作用 | 说明 |
|------|------|------|
| ① 预处理(可选) | 水下图像增强 | 复用 `0.预处理` 模块，默认开启（主程序规范：前置图像同样预处理）；若预处理导致 YOLO 检测率下降可设 `enable_preprocess=False` |
| ② YOLO 检测 | 目标检测 | 基于 `ultralytics`（YOLOv8/v11），按 `conf/iou/imgsz/device` 参数推理 |
| ③ 目标过滤 | 只保留球 | 按 `target_class_names` / `target_class_ids` 过滤目标类别，剔除过小误检 |
| ④ 目标选择 | 定撞击目标 | 多球时按策略选择：`largest`(最近) 或 `center`(最居中) |
| ⑤ 决策计算 | 转向+撞击判断 | 球心偏移→`steer`；居中且框宽≥`strike_width_px`→`strike_ready`；可选距离估算 |

## 模型准备

撞球检测需一个能识别目标球的 YOLO 模型（`.pt`）：

```bash
pip install ultralytics                 # 安装依赖
yolo train data=ball.yaml model=yolov8n.pt epochs=100   # 训练自己的球模型
```

- 将训练好的模型路径填入 [config.py](config.py) 的 `model_path`（默认 `yolov8n.pt`，ultralytics 内置，首次使用自动下载）
- 将球类别名/编号填入 `target_class_names` / `target_class_ids`，否则检测结果不会被视为目标球

## API 使用

```python
import cv2
from ball_detector import BallDetector   # 或 from P2 撞球 import BallDetector

detector = BallDetector()                # 使用默认配置 (config.py 的 DEFAULT_CONFIG)
# 或: detector = BallDetector({'model_path': 'my_ball.pt', 'conf_threshold': 0.3})

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)      # 主入口

    if result['detected']:
        print(f"转向: {result['steer']}, 偏移: {result['offset']:.1f}px")

    vis = detector.draw_result(frame, result)   # 可视化叠加 (调试用)
    cv2.imshow("result", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 输出结果字典

| 键 | 类型 | 含义 |
|----|------|------|
| `detected` | bool | 是否检测到目标球 |
| `target` | dict/None | 选中的目标球：`bbox`(像素框)、`center`、`conf`、`class_name`、`width`/`height` |
| `balls` | list | 所有检测到的目标球（同 `target` 结构） |
| `offset` | float | 球心偏离画面垂直中心线的像素数，**正值=偏右，负值=偏左** |
| `steer` | str/None | 转向决策：`LEFT` / `CENTER` / `RIGHT`（`None`=未检测到） |
| `distance` | float/None | 目标球估算距离（cm，针孔模型，需标定焦距） |
| `strike_ready` | bool | 是否**居中且足够近**，可执行撞击 |
| `center` | (x, y) | 目标球中心坐标 |
| `last_valid` | dict(可选) | 未检测到时附带上一帧有效结果，供降级策略使用 |

### 主要方法

| 方法 | 说明 |
|------|------|
| `detect(frame)` | 主入口，输入 BGR 帧，返回结果字典 |
| `update_config(dict)` | 运行时更新参数（供滑动条调参、动态调整用） |
| `draw_result(frame, result)` | 叠加绘制中心线/目标框/偏移箭头/转向与撞击提示 |

## 调参工具 test_tool.py

本地调参使用，实时显示原始画面与检测结果，滑动条调整置信度阈值后立即生效：

```
python test_tool.py                          # 使用默认摄像头 (index 0)
python test_tool.py --camera 1               # 指定摄像头
python test_tool.py --image test.jpg         # 测试单张图片
python test_tool.py --video test.mp4         # 测试视频文件
python test_tool.py --resolution 1280 720    # 指定摄像头分辨率
```

**快捷键：** `q/ESC` 退出 ｜ `s` 保存截图 ｜ `p` 打印当前参数 ｜ `空格` 暂停

## 关键参数速查 (详见 config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_preprocess` | True | 是否在 YOLO 推理前先做水下预处理（默认开启） |
| `model_path` | `yolov8n.pt` | YOLO 模型路径（训练好的球模型） |
| `device` | None | 推理设备：`None`=自动 / `'cpu'` / `'0'`(GPU) |
| `conf_threshold` / `iou_threshold` | 0.5 / 0.45 | 置信度 / NMS 阈值 |
| `target_class_names` / `target_class_ids` | `['ball']` / `[]` | 目标球类别（名称或 id） |
| `target_strategy` | `largest` | 多球时目标选择：`largest`(最近) / `center`(最居中) |
| `center_zone_ratio` | 0.12 | 居中判定：球心偏差占画面宽度比例 |
| `strike_width_px` | 150 | 撞击就绪：球框宽达到此像素即认为足够近 |
| `enable_distance` / `focal_length_px` / `ball_real_diameter_cm` | True / 554 / 10 | 距离估算（针孔模型） |

## 降级策略

当某一帧未检测到目标球（遮挡、出水、暂时出画面）时：
- 结果中附带 `last_valid` 字段（上一帧有效结果），供控制器做短时保持/平滑
- 连续多帧未检测到且 `steer` 为 `None` 时，应让控制器执行搜索逻辑（如旋转寻找），该逻辑在控制端实现

## 与主程序集成

`visual_main.py` 通过 `_load_task_module("P2 撞球", "ball_detector.py")` 动态加载本模块，实例化 `BallDetector` 后调用：

```python
system = VisualSystem(...)
system.init_camera()
system.run_ball_hitting()    # 内部: detect → print steer/offset/distance → draw_result 显示
```

运行方式：

```
python visual_main.py --task P2        # 仅运行撞球任务
python visual_main.py --task P1        # 仅运行巡线任务
python visual_main.py --no-display     # 无窗口模式 (仅打印结果)
```

> 注意：本模块内已通过 `importlib` 自行加载 `0.预处理`、`config.py` 与 `yolo_detector.py`，因此无论被主程序还是单独运行都能正常工作。
