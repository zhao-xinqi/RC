# 0.预处理 - 通用水下图像预处理模块

水下图像由于水体对光的选择性吸收和散射，普遍存在**偏蓝绿、低对比度、光照不均、噪点多**等问题。
本模块提供统一的预处理流水线，供 P1~P5 所有视觉任务模块调用，保证各任务入口一致、参数可调。

## 文件结构

```
0.预处理/
├── preprocessor.py    # 预处理核心实现 (preprocess 主函数 + 内部辅助函数)
├── __init__.py        # 模块导出 (对外仅暴露 preprocess)
└── README.md          # 本文档
```

## 处理流水线

```
原始帧 → ① 缩放 → ② 水下颜色校正 → ③ 高斯去噪 → ④ CLAHE 增强 → 输出帧
```

| 步骤 | 作用 | 说明 |
|------|------|------|
| ① 缩放 | 统一分辨率 | 摄像头直出分辨率通常偏高，统一缩放到目标尺寸，降低后续计算量 |
| ② 颜色校正 | 补偿红光衰减 | 基于"灰度世界假设"做白平衡，并对红色通道额外增强（`red_boost`），减轻水下偏蓝绿问题 |
| ③ 高斯去噪 | 平滑噪声 | 消除水下悬浮颗粒、传感器噪声，核大小需为奇数 |
| ④ CLAHE | 自适应对比度增强 | 在 LAB 空间 L 通道上做局部直方图均衡，应对水下光照不均，且不破坏色彩 |

各步骤均可通过配置独立开关（`enable_*`），不需要的环节可以直接关闭以省去计算开销。

## API 使用

```python
import cv2
from preprocessor import preprocess   # 或 from 0.预处理 import preprocess

frame = cv2.imread("test.jpg")

# 使用默认配置
processed = preprocess(frame)

# 自定义配置 (未指定的键使用默认值)
config = {
    'image_width': 320,          # 降低分辨率加速
    'enable_gaussian': False,    # 关闭高斯去噪
    'red_boost': 1.5,            # 增强红色补偿
    'clahe_clip': 3.0,
}
processed = preprocess(frame, config)

cv2.imwrite("processed.jpg", processed)
```

## 配置参数说明

| 配置键 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `image_width` | int | 640 | 目标宽度 |
| `image_height` | int | 480 | 目标高度 |
| `enable_resize` | bool | True | 是否缩放 |
| `enable_color_correct` | bool | True | 是否水下颜色校正 |
| `red_boost` | float | 1.2 | 红色通道增强系数（>1.0 增强，<1.0 抑制） |
| `enable_gaussian` | bool | True | 是否高斯去噪 |
| `gaussian_kernel` | int | 5 | 高斯核大小（偶数会自动 +1 保证奇数） |
| `enable_clahe` | bool | True | 是否 CLAHE 增强 |
| `clahe_clip` | float | 2.0 | 对比度限幅，越大对比度越强 |
| `clahe_tile` | tuple | (8, 8) | CLAHE 网格大小，越小局部性越强 |

> 建议：水下实测后重点调 `red_boost`（红光补偿）和 `clahe_clip`（对比度），这两项对水下画面影响最明显。

## 调用约定

- 输入：BGR 格式的 numpy 数组（`cv2.VideoCapture` 或 `cv2.imread` 的输出即为此格式）
- 输出：BGR 格式，尺寸为 `(image_width, image_height)`
- 内部对输入做 `copy()`，不会修改调用方的原图
- 各任务模块（如 P1 巡线）已在内部调用本模块，**任务调用方通常无需直接使用**，仅在需要单独调试预处理效果时使用

## 与主程序集成

`visual_main.py` 和各任务模块通过 `importlib` 动态加载本文件（文件夹名含数字，无法用标准 `import`），
因此本模块的导出接口保持稳定即可，目录改名或调整不影响其他模块：

```python
# 各任务模块内部的加载方式 (无需手动修改)
_PREP_PATH = os.path.join(_PROJ_ROOT, "0.预处理", "preprocessor.py")
# → 得到 preprocess(frame, config) 函数
```
