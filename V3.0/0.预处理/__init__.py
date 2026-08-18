"""
0.预处理 - 通用水下图像预处理模块

本模块提供统一的水下图像预处理函数, 供所有任务模块调用.
处理流水线: 缩放 → 颜色校正 → 去噪 → CLAHE 增强
"""

from .preprocessor import preprocess

__all__ = ['preprocess']
