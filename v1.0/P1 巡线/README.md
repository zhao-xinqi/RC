# P1 巡线 - 引导线检测模块

从下视摄像头图像中检测池底**橙红色引导线**，输出引导线相对画面中心线的偏移量（offset）和倾斜角度（angle），供机器人控制器调整航向。是机器人水中巡游项目的第一步任务。

## 文件结构

```
P1 巡线/
├── config.py          # 集中管理全部可调参数 (HSV阈值/形态学/检测参数)
├── line_detector.py   # 检测器核心: LineDetector 类
├── test_tool.py       # 本地调参测试工具 (滑动条实时调参)
├── __init__.py        # 模块导出 (动态加载, 兼容中文文件夹名)
└── README.md          # 本文档
```

## 检测流水线

```
原始帧 → ① 预处理 → ② HSV颜色分割 → ③ 形态学处理 → ④ 轮廓提取 → ⑤ 偏移/角度计算
```

| 步骤 | 作用 | 说明 |
|------|------|------|
| ① 预处理 | 图像增强 | 复用 `0.预处理` 模块（缩放/颜色校正/去噪/CLAHE），参数在 config.py 中传递 |
| ② HSV 颜色分割 | 提取橙红色区域 | 橙红色跨 HSV 的 H=0° 边界，分两段阈值合并：主段 `[0~25]` + 跨边界段 `[150~179]` |
| ③ 形态学处理 | 去噪、连接断线 | 开运算去细小噪点 + 闭运算弥合因反光/遮挡产生的断口 |
| ④ 轮廓提取 | 找最大轮廓 | 取最大轮廓并过滤小于 `min_contour_area` 的噪声区域（可选 ROI 限制） |
| ⑤ 计算 | 偏移量 + 角度 | 图像矩算重心 → offset；`cv2.fitLine` 最小二乘拟合 → angle |

## API 使用

```python
import cv2
from line_detector import LineDetector   # 或 from P1 巡线 import LineDetector

detector = LineDetector()                # 使用默认配置 (config.py 的 DEFAULT_CONFIG)
# 或: detector = LineDetector({'hsv_lower': [0, 100, 100]})   # 自定义配置

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)      # 主入口

    if result['detected']:
        print(f"偏移: {result['offset']:.1f}px, 角度: {result['angle']:.1f}°")

    vis = detector.draw_result(frame, result)   # 可视化叠加 (调试用)
    cv2.imshow("result", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 输出结果字典

| 键 | 类型 | 含义 |
|----|------|------|
| `offset` | float | 引导线重心偏离画面垂直中心线的水平像素数，**正值=偏右，负值=偏左** |
| `angle` | float | 拟合直线与垂直方向（前进方向）的夹角（度），**正值=线向右偏，负值=线向左偏** |
| `detected` | bool | 是否检测到有效引导线 |
| `center` | (x, y) | 引导线重心坐标 |
| `last_valid` | dict(可选) | 未检测到时附带上一帧有效结果，供降级策略使用 |

### 主要方法

| 方法 | 说明 |
|------|------|
| `detect(frame)` | 主入口，输入 BGR 帧，返回结果字典 |
| `update_config(dict)` | 运行时更新参数（供滑动条调参、动态调整用） |
| `draw_result(frame, result, show_mask=False)` | 叠加绘制偏移箭头/角度线/文字信息，`show_mask=True` 时右侧拼接二值掩码 |

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

**调参建议：** 先调 HSV 阈值让掩码只保留引导线 → 再调开/闭运算核大小消除噪点和断口 → 最后调最小面积过滤残留噪点。调试完成后将参数写回 [config.py](config.py) 的 `DEFAULT_CONFIG`。

## 关键参数速查 (详见 config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hsv_lower` / `hsv_upper` | `[0,80,80]` / `[25,255,255]` | 橙红色主段 HSV 阈值 |
| `hsv_lower2` / `hsv_upper2` | `[150,80,80]` / `[179,255,255]` | 红色跨 180° 边界段（偏红时启用） |
| `morph_open_kernel` / `morph_close_kernel` | 3 / 7 | 开/闭运算核大小 |
| `min_contour_area` | 500 | 最小轮廓面积过滤（像素²） |
| `roi_enabled` / `roi_ratio` | False / `[0.2,0.9,0.1,0.9]` | 是否限制检测区域 `[y起, y止, x起, x止]`（比例） |
| `red_boost` | 1.2 | 红色增强（水下红光衰减，传递给预处理） |

## 降级策略

当某一帧未检测到引导线（遮挡、反光、暂时出画面）时：
- 结果中附带 `last_valid` 字段（上一帧有效结果），供控制器做短时保持/平滑
- 连续多帧未检测到时，应让控制器执行搜索逻辑（如原地旋转找线），该逻辑在控制端实现

## 与主程序集成

`visual_main.py` 通过 `_load_task_module("P1 巡线", "line_detector.py")` 动态加载本模块，实例化 `LineDetector` 后调用：

```python
system = VisualSystem(...)
system.init_camera()
system.run_line_following()    # 内部: detect → print offset/angle → draw_result 显示
```

运行方式：

```
python visual_main.py --task P1        # 仅运行巡线任务
python visual_main.py --no-display     # 无窗口模式 (仅打印结果)
```

> 注意：本模块内已通过 `importlib` 自行加载 `0.预处理` 与 `config.py`，因此无论被主程序还是单独运行都能正常工作。
