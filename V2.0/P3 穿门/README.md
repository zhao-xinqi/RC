# P3 穿门 - 高低门检测模块

从前视摄像头图像中检测红色 **"Π 形"门**（两竖杆 + 顶部横梁，内部空心），输出门洞中心、门类型（高门/矮门）与对齐状态，供机器人控制器对准门洞穿行。是完成撞球任务后的前置导航任务。

## 文件结构

```
P3 穿门/
├── config.py          # 集中管理全部可调参数 (预处理/HSV阈值/Π形判别)
├── door_detector.py   # 检测器核心: DoorDetector 类
├── test_tool.py       # 本地调参测试工具 (滑动条实时调参)
├── __init__.py        # 模块导出 (动态加载, 兼容中文文件夹名)
└── README.md          # 本文档
```

## 检测流水线

```
原始帧 → ① 预处理 → ② HSV红色分割 → ③ 形态学处理 → ④ 轮廓筛选 → ⑤ Π形验证 → ⑥ 门中心/类型/对齐
```

| 步骤 | 作用 | 说明 |
|------|------|------|
| ① 预处理 | 图像增强 | 复用 `0.预处理` 模块（缩放/颜色校正/去噪/CLAHE），参数在 config.py 中传递 |
| ② HSV 红色分割 | 提取红色区域 | 红色跨 H=0° 边界，分两段阈值合并：主段 `[0~25]` + 跨边界段 `[150~179]` |
| ③ 形态学处理 | 去噪、连接断线 | 开运算去细小噪点 + 闭运算弥合门框因反光/遮挡产生的断口 |
| ④ 轮廓筛选 | 尺寸/宽高比过滤 | 按面积降序遍历轮廓，过滤面积过小、宽高比异常的区域 |
| ⑤ Π 形验证 | 结构判别 | 外接矩形内采样 顶部横梁/左右竖杆/底部条/内部区 红色占比，验证"Π"结构 |
| ⑥ 计算 | 门中心 + 类型 | 外接矩形中心 → 门洞中心；由横梁位置判定高门/矮门；水平偏移 → 对齐状态 |

## 与红色物体的区分（§6 判别表）

水下场景中门、收集框、圆环、撞球同为红色，靠形状结构区分：

| 物体 | 判别依据 | 本模块判据 |
|------|----------|-----------|
| **门（Π形）** | 顶部横梁+左右竖杆，无底杆，内部空心 | ✅ 通过：顶/左右杆红色占比高 + 底部无红 + 内部空心 |
| 收集框 | 四边闭合矩形，内部空洞 | ❌ 有底杆 → 底部条红色占比高，被 `bottom_empty_threshold` 排除 |
| 圆环 | 空心圆，宽高比≈1 | ❌ 无横梁/竖杆结构 → 被 Π 形采样判据排除 |
| 撞球 | 实心圆 | ❌ 内部填满 → 内部红色占比高，被 `interior_empty_threshold` 排除 |

## API 使用

### 1) 实时摄像头模式（推荐）

```python
import cv2
from door_detector import DoorDetector, get_config

detector = DoorDetector(get_config())
detector.run_camera(camera_id=0, show_window=True, print_offset=True)
```

### 2) 单帧检测模式

```python
import cv2
from door_detector import DoorDetector, get_config

detector = DoorDetector(get_config())
frame = cv2.imread("door.jpg")
result = detector.detect_with_center_offset(frame)

if result['detected']:
    print(f"门类型: {result['door_type']}")
    print(f"门中心: {result['center']}")
    print(f"camera_center: {result['camera_center']}")
    print(f"偏移(dx,dy): {result['center_offset']}")
```

### 3) 自定义可视化

```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect_with_center_offset(frame)
    vis = detector.draw_result(frame, result)
    cv2.imshow("result", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 输出结果字典

| 键 | 类型 | 含义 |
|----|------|------|
| `detected` | bool | 是否检测到 Π 形门 |
| `door_type` | str/None | 门类型 `'high'`(高门) / `'low'`(矮门) |
| `center` | (x, y) | 门洞中心像素坐标 |
| `center_norm` | (dx, dy) | 相对画面中心归一化偏移 [-1,1]，**控制端可直接做偏差** |
| `bbox` | (x1,y1,x2,y2) | 门框外接矩形（像素） |
| `offset` | float | 门洞中心偏离画面垂直中心线的像素数，**正值=偏右，负值=偏左** |
| `aligned` | bool | 门洞中心是否与画面中心对齐（\|center_norm_x\| ≤ `aligned_ratio`） |
| `camera_center` | (x, y) | 配置的参考中心点，来自 `config.py` 的 `camera_center` |
| `center_offset` | (dx, dy) | 门中心相对 `camera_center` 的偏移，**右/上为正** |
| `mask` | ndarray/None | 红色二值掩码（调试用） |
| `last_valid` | dict(可选) | 未检测到时附带上一帧有效结果，供降级策略使用 |

### 主要方法

| 方法 | 说明 |
|------|------|
| `detect(frame)` | 主入口，输入 BGR 帧，返回结果字典 |
| `update_config(dict)` | 运行时更新参数（供滑动条调参、动态调整用） |
| `draw_result(frame, result, show_mask=False)` | 叠加绘制门框/门洞中心/偏移箭头/文字信息，`show_mask=True` 时右侧拼接二值掩码 |

## 调参工具 test_tool.py

本地调参使用，4 个窗口实时显示，滑动条调整后立即生效：

```
python test_tool.py                          # 使用默认摄像头 (index 0)
python test_tool.py --camera 1               # 指定摄像头
python test_tool.py --image test.jpg         # 测试单张图片
python test_tool.py --video test.mp4         # 测试视频文件
python test_tool.py --resolution 1280 720    # 指定摄像头分辨率
```

**快捷键：** `q/ESC` 退出 ｜ `s` 保存截图 ｜ `p` 打印当前参数 ｜ `空格` 暂停

**调参建议：** 先调 HSV 阈值让掩码只保留红色门框 → 再调开/闭运算核大小消除噪点和断口 → 观察掩码中门是否呈"Π"形、底部和内部是否空心 → 最后按 `p` 打印参数写回 [config.py](config.py) 的 `DEFAULT_CONFIG`。

## 关键参数速查 (详见 config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hsv_lower` / `hsv_upper` | `[0,80,80]` / `[25,255,255]` | 红色主段 HSV 阈值 |
| `hsv_lower2` / `hsv_upper2` | `[150,80,80]` / `[179,255,255]` | 红色跨 180° 边界段 |
| `morph_open_kernel` / `morph_close_kernel` | 3 / 9 | 开/闭运算核大小 |
| `min_contour_area` | 800 | 最小轮廓面积过滤（像素²） |
| `min/max_door_width_ratio` | 0.15 / 0.95 | 门宽占画面宽比例范围 |
| `min/max_door_height_ratio` | 0.15 / 1.0 | 门高占画面高比例范围 |
| `min/max_aspect_ratio` | 0.4 / 2.0 | 外接矩形宽高比范围 |
| `beam_fill_threshold` | 0.4 | 顶部横梁红色占比下限 |
| `pillar_fill_threshold` | 0.3 | 左右竖杆红色占比下限 |
| `bottom_empty_threshold` | 0.5 | 底部条红色占比上限（区分收集框） |
| `interior_empty_threshold` | 0.3 | 内部区红色占比上限（区分实心物） |
| `aligned_ratio` | 0.15 | 对齐判定阈值（归一化） |
| `camera_center` | (320, 240) | 摄像头参考中心点；输出偏移时以它为基准 |
| `door_top_norm_threshold` | 0.5 | 高/矮门判定：横梁归一化 y 位置阈值 |

> 说明：高/矮门当前按"横梁在画面中的纵向位置"判定。若实际高/矮指门自身高度，可改为按外接矩形高度占比判定（见 config.py 注释）。

## 降级策略

当某一帧未检测到门（被遮挡、暂时出画面）时：
- 结果中附带 `last_valid` 字段（上一帧有效结果），供控制器做短时保持/平滑
- 连续多帧未检测到时，应让控制器执行搜索逻辑（如原地旋转找门），该逻辑在控制端实现

## 与主程序集成

`visual_main.py` 通过 `_load_task_module("P3 穿门", "door_detector.py")` 动态加载本模块，实例化 `DoorDetector` 后调用：

```python
door_detector = door_mod.DoorDetector(P3_DOOR_DETECTION_CONFIG)
result = door_detector.detect(frame)
```

运行方式：

```
python visual_main.py --task P3        # 仅运行穿门任务
python visual_main.py --no-display     # 无窗口模式 (仅打印结果)
```

> 注意：本模块内已通过 `importlib` 自行加载 `0.预处理` 与 `config.py`，因此无论被主程序还是单独运行都能正常工作。
