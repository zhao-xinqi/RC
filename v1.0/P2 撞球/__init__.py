"""
P2 撞球 模块导出

由于文件夹名含中文, 标准 Python import 不可用.
本模块使用 importlib 动态加载子模块, 确保在任何调用方式下都能正常工作.
"""

import os
import importlib.util

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_submodule(filename, modname):
    path = os.path.join(_MODULE_DIR, filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_yolo_detector = _load_submodule("yolo_detector.py", "P2_yolo_detector")
_ball_detector = _load_submodule("ball_detector.py", "P2_ball_detector")
_config = _load_submodule("config.py", "P2_config")

YoloDetector = _yolo_detector.YoloDetector
BallDetector = _ball_detector.BallDetector
DEFAULT_CONFIG = _config.DEFAULT_CONFIG
get_config = _config.get_config

__all__ = ['BallDetector', 'YoloDetector', 'DEFAULT_CONFIG', 'get_config']
