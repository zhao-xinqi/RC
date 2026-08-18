# P5 放球 - 红色篮筐（收集框）检测模块

从前视摄像头图像中通过 **HSV** 识别红色**篮筐/收集框**（"口"形闭合矩形边框，内部空心），输出框口中心、框的倾斜朝向与对齐状态，供机器人控制器对准框口放球。

> 纯 HSV 方案，不依赖 YOLO：红色在蓝绿水下背景对比度高，四边闭合 + 内部空心的形状判据稳定可靠。

## 文件结构

```
P5 放球/
├── config.py          # 集中管理全部可调参数 (预处理/HSV阈值/闭合矩形判别)
├── box_detector.py    # 检测器核心: BoxDetector 类
├── test_tool.py       # 本地调参测试工具 (滑动条实时调参)
├── __init__.py        # 模块导出 (动态加载, 兼容中文文件夹名)
└── README.md          # 本文档
```

## 检测流水线

```
原始帧 → ① 预处理 → ② HSV红色分割 → ③ 形态学处理 → ④ 轮廓筛选 → ⑤ 闭合矩形验证 → ⑥ 框口中心/朝向/对齐
```

| 步骤 | 作用 | 说明 |
|------|------|------|
| ① 预处理 | 图像增强 | 复用 `0.预处理` 模块（缩放/颜色校正/去噪/CLAHE），参数在 config.py 中传递 |
| ② HSV 红色分割 | 提取红色区域 | 红色跨 H=0° 边界，分两段阈值合并：主段 `[0~25]` + 跨边界段 `[150~179]` |
| ③ 形态学处理 | 去噪、连接断线 | 开运算去细小噪点 + 闭运算弥合边框因反光/遮挡产生的断口 |
| ④ 轮廓筛选 | 尺寸/宽高比过滤 | 按面积降序遍历轮廓，过滤面积过小、宽高比异常的区域 |
| ⑤ 闭合矩形验证 | 结构判别 | 用 `minAreaRect` 有向矩形，四边边线采样 + 内部空心判定（对倾斜框稳健） |
| ⑥ 计算 | 框口中心 + 朝向 | 框中心 → 框口中心；PCA 主方向 → 倾斜角；水平偏移 → 对齐状态 |

## 与红色物体的区分（§6 判别表）

水下场景中篮筐、门、圆环、撞球同为红色，靠形状结构区分：

| 物体 | 判别依据 | 本模块判据 |
|------|----------|-----------|
| **篮筐/收集框** | "口"形闭合矩形，内部空洞 | ✅ 通过：四边(顶/底/左/右)红色占比高 + 内部空心 |
| Π 形（非框/非门） | 顶部横梁+两竖杆，**无底杆** | ❌ 底边空 → 底边红色占比低，被 `edge_fill_threshold` 排除 |
| 圆环 | 空心圆 | ❌ 边线采样只穿过圆弧（填充率~20%）→ 被 `edge_fill_threshold` 排除 |
| 撞球 | 实心圆 | ❌ 内部填满 → 被 `interior_empty_threshold` 排除 |

> 与 P3 门的关系：**门框与篮筐同为红色闭合矩形**，形状上无法互相区分。
> P3 直接复用本模块 `BoxDetector` 作为检测内核，靠比赛阶段（`TASK_STAGE`）区分二者（穿门/放球不同时出现）。

## API 使用

```python
import cv2
from box_detector import BoxDetector   # 或 from P5 放球 import BoxDetector

detector = BoxDetector()               # 使用默认配置 (config.py 的 DEFAULT_CONFIG)
# 或: detector = BoxDetector({'hsv_lower': [0, 80, 80]})   # 自定义配置

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)    # 主入口

    if result['detected']:
        print(f"偏移: {result['offset']:.1f}px, 倾斜: {result['angle']:.1f}°, "
              f"对齐: {result['aligned']}")

    vis = detector.draw_result(frame, result)   # 可视化叠加 (调试用)
    cv2.imshow("result", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 输出结果字典

| 键 | 类型 | 含义 |
|----|------|------|
| `detected` | bool | 是否检测到闭合矩形篮筐 |
| `center` | (x, y) | 框口中心像素坐标 |
| `center_norm` | (dx, dy) | 相对画面中心归一化偏移 [-1,1]，**控制端可直接做偏差** |
| `bbox` | (x1,y1,x2,y2) | 篮筐外接矩形（像素） |
| `offset` | float | 框口中心偏离画面垂直中心线的像素数，**正值=偏右，负值=偏左** |
| `angle` | float | 框长轴相对画面竖直方向的倾斜角（度），**正=长轴偏右，负=偏左** |
| `aspect` | float | 框长/短轴长度比（≥1），描述框的摆向 |
| `aligned` | bool | 框口中心是否与画面中心对齐（\|center_norm_x\| ≤ `aligned_ratio`） |
| `mask` | ndarray/None | 红色二值掩码（调试用） |
| `last_valid` | dict(可选) | 未检测到时附带上一帧有效结果，供降级策略使用 |

### 主要方法

| 方法 | 说明 |
|------|------|
| `detect(frame)` | 主入口，输入 BGR 帧，返回结果字典 |
| `update_config(dict)` | 运行时更新参数（供滑动条调参、动态调整用） |
| `draw_result(frame, result, show_mask=False)` | 叠加绘制框/框口中心/偏移箭头/文字信息，`show_mask=True` 时右侧拼接二值掩码 |

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

**调参建议：** 先调 HSV 阈值让掩码只保留红色边框 → 再调开/闭运算核大小消除噪点和断口 → 调 `Edge Fill`/`Interior` 阈值使篮筐通过而门/球/环被拒 → 最后按 `p` 打印参数写回 [config.py](config.py) 的 `DEFAULT_CONFIG`。

## 关键参数速查 (详见 config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hsv_lower` / `hsv_upper` | `[0,80,80]` / `[25,255,255]` | 红色主段 HSV 阈值 |
| `hsv_lower2` / `hsv_upper2` | `[150,80,80]` / `[179,255,255]` | 红色跨 180° 边界段 |
| `morph_open_kernel` / `morph_close_kernel` | 3 / 9 | 开/闭运算核大小 |
| `min_contour_area` | 600 | 最小轮廓面积过滤（像素²） |
| `min/max_box_width_ratio` | 0.10 / 0.90 | 框宽占画面宽比例范围 |
| `min/max_box_height_ratio` | 0.10 / 0.95 | 框高占画面高比例范围 |
| `min/max_aspect_ratio` | 0.4 / 2.5 | 外接矩形宽高比范围 |
| `edge_fill_threshold` | 0.4 | 四边边线红色占比下限 |
| `interior_empty_threshold` | 0.3 | 内部空心：红色占比上限（区分实心物） |
| `edge_inset_ratio` | 0.04 | 边采样内缩比例（避开形态学侵蚀/抗锯齿的边缘） |
| `interior_shrink_ratio` | 0.6 | 内部区域角点收缩比例 |
| `aligned_ratio` | 0.15 | 对齐判定阈值（归一化） |
| `camera_center` | (320, 240) | 摄像头参考中心点；输出偏移时以它为基准 |

## 降级策略

当某一帧未检测到篮筐（被遮挡、暂时出画面）时：
- 结果中附带 `last_valid` 字段（上一帧有效结果），供控制器做短时保持/平滑
- 连续多帧未检测到时，应让控制器执行搜索逻辑（如原地旋转找框），该逻辑在控制端实现

## 与主程序集成

`visual_main.py` 通过 `_load_task_module("P5 放球", "box_detector.py")` 动态加载本模块，实例化 `BoxDetector` 后调用：

```python
box_detector = box_mod.BoxDetector(P5_BOX_DETECTION_CONFIG)
result = box_detector.detect(frame)
```

运行方式：

```
python visual_main.py --task P5        # 仅运行放球任务
python visual_main.py --no-display     # 无窗口模式 (仅打印结果)
```

> 注意：本模块内已通过 `importlib` 自行加载 `0.预处理` 与 `config.py`，因此无论被主程序还是单独运行都能正常工作。
