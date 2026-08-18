"""
main_RDK.py - RDK 板端双摄像头功能测试入口

功能:
  在 RDK 板端同时打开前视 + 下视两个摄像头, 各用一个窗口并发显示检测结果:
    前视窗口 (FrontCamTester) : P2 撞球 (球检测, hbm) + P3 穿门 (门检测)
    下视窗口 (BottomCamTester): P1 巡线 (引导线) + P4 抓球 (球检测, hbm) + P5 放球 (收集框)

关键点:
  - 球检测固定使用 hbm 模型 (RDK BPU), 不使用 best.pt;
    相关代码复用 P2 撞球/main.py 的 create_detector (自动修正 anchor_sizes +
    设置 BPU 调度参数), 即 RDK 调优后的推理路径, 见 RDKBallDetector.
  - P1/P3/P5 的 overrides 复用 main_front_config / main_bottom_config,
    避免多份配置漂移 (由 build_config 合并).
  - 双摄像头并发运行, 任一结束 (视频播完/按 q) 时协调另一路同时退出.

运行示例 (RDK 板端):
  python main_RDK.py --front-source 0 --bottom-source 1
  python main_RDK.py --front-source 0 --bottom-source 1 --target red
  python main_RDK.py --front-source front.mp4 --bottom-source bottom.mp4 --no-show
  python main_RDK.py --front-source 0 --bottom-source 1 --model best_nashe_320x320_nv12.hbm
"""

import argparse
import importlib.util
import threading
import time
from pathlib import Path


# ================================================================
# 常量
# ================================================================
_PROJ_ROOT = Path(__file__).resolve().parent          # V3.0 根目录
_FRONT_PATH = _PROJ_ROOT / "main_front.py"
_BOTTOM_PATH = _PROJ_ROOT / "main_bottom.py"
_RDK_CFG_PATH = _PROJ_ROOT / "main_rdk_config.py"
_P2_DIR = _PROJ_ROOT / "P2 撞球"
_P2_MAIN_PATH = _P2_DIR / "main.py"


# ================================================================
# 通用工具
# ================================================================

def load_module_by_path(module_name: str, path: Path):
    """按文件路径动态加载 Python 模块 (兼容中文/数字目录名)。

    各任务文件夹名含数字与中文, 无法用标准 import 语句,
    故使用 importlib.util.spec_from_file_location 按路径加载。
    """
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 加载 main_front / main_bottom (两者模块顶层均不依赖 hbm_runtime)
_front_mod = load_module_by_path("main_front", _FRONT_PATH)
_bottom_mod = load_module_by_path("main_bottom", _BOTTOM_PATH)

# 复用前视/下视测试器与检测器工厂
FrontCamTester = _front_mod.FrontCamTester
BottomCamTester = _bottom_mod.BottomCamTester
create_door_detector = _front_mod.create_door_detector
create_line_detector = _bottom_mod.create_line_detector
create_box_detector = _bottom_mod.create_box_detector


# 加载本文件同目录的配置模块 (main_rdk_config.py)
_rdk_cfg_mod = load_module_by_path("rdk_main_config_old", _RDK_CFG_PATH)
get_rdk_config = _rdk_cfg_mod.get_config


# ================================================================
# P2/P4 球检测 - hbm 适配器 (复用 P2 撞球/main.py)
# ================================================================

class RDKBallDetector:
    """基于 P2 撞球/main.py 的 hbm 球检测适配器。

    暴露与其它检测器一致的 detect(frame) -> 统一结果列表接口:
        [{'label': str, 'score': float, 'bbox': (x1,y1,x2,y2),
          'center': (cx, cy)}, ...]

    内部推理走 P2.main 的 create_detector:
      - 按模型实际输入分辨率修正各尺度 anchor_sizes (320x320 -> [40,20,10])
      - 设置 BPU 调度参数 (priority / bpu_cores)
    实时路径与 P2.main 的 run_live_detect 一致: 直接 predict(frame),
    不再叠加 0.预处理 (YoloDetect 内部已做 NV12 转换 + 缩放).
    """

    def __init__(self, p2_module, p2_cfg, model_path, bpu_cores, label_file=None):
        self._p2m = p2_module
        # bpu_cores 按实例覆盖 (前视/下视分配不同核心, 避免 BPU 争抢)
        self._cfg = dict(p2_cfg)
        self._cfg["bpu_cores"] = list(bpu_cores)
        self._labels = self._p2m.load_labels(label_file)
        # 构造检测器 (含 anchor_sizes 修正 + BPU 调度设置)
        self._model = self._p2m.create_detector(self._cfg, model_path)

    def detect(self, frame):
        """在单帧上检测目标球, 返回统一格式结果列表。"""
        boxes, scores, cls_ids = self._model.predict(frame)
        boxes, scores, cls_ids = self._p2m.filter_target_classes(
            boxes, scores, cls_ids, self._cfg, self._labels)

        dets = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            label = self._labels[cls_id] if cls_id < len(self._labels) else str(int(cls_id))
            dets.append({
                "label": label,
                "score": float(score),
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })
        return dets


def create_ball_detector(rdk_cfg, bpu_cores):
    """创建 hbm 球检测适配器 (前视/下视各一个实例, 避免两线程并发调用同一模型)。

    Args:
        rdk_cfg:  main_rdk_config 配置
        bpu_cores: 本实例使用的 BPU 核心索引列表
    """
    p2m = load_module_by_path("P2_main", _P2_MAIN_PATH)   # 惰性加载 (需 hbm_runtime)
    cfg = p2m.get_config()                    # 以 P2 config.py 为基准
    cfg["score_thres"] = rdk_cfg["p2_score_thres"]
    cfg["nms_thres"] = rdk_cfg["p2_nms_thres"]
    cfg["priority"] = rdk_cfg["p2_priority"]
    cfg["target_class_names"] = rdk_cfg["p2_target_class_names"]
    cfg["target_class_ids"] = rdk_cfg["p2_target_class_ids"]

    model_path = rdk_cfg["p2_model_path"] or "best_nashe_320x320_nv12.hbm"
    model_path = p2m.resolve_model_path(model_path)   # 相对 P2 目录展开

    return RDKBallDetector(p2m, cfg, model_path, bpu_cores,
                           label_file=rdk_cfg["p2_label_file"])


# ================================================================
# 命令行入口
# ================================================================

def parse_args():
    """解析命令行参数; 未指定的参数使用 main_rdk_config.py 中的默认值。"""
    parser = argparse.ArgumentParser(
        description="RDK 板端双摄像头并发测试 (前视: P2 撞球+P3 穿门 | 下视: P1 巡线+P4 抓球+P5 放球)")
    parser.add_argument("--front-source", type=str, default=None,
                        help="前视摄像头编号或视频文件路径, 例如 0 / front.mp4")
    parser.add_argument("--bottom-source", type=str, default=None,
                        help="下视摄像头编号或视频文件路径, 例如 1 / bottom.mp4")
    parser.add_argument("--model", type=str, default=None,
                        help="hbm 球检测模型路径 (默认取 P2 撞球/best_nashe_320x320_nv12.hbm)")
    parser.add_argument("--label-file", type=str, default=None,
                        help="hbm 后端类别名称文件 (每行一个类别名)")
    parser.add_argument("--target", type=str, nargs="+", default=None,
                        help="只保留的目标类别, 例如 --target red; 不指定则保留全部")
    parser.add_argument("--conf", type=float, default=None, help="球检测置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="球检测 NMS IoU 阈值")
    parser.add_argument("--width", type=int, default=None, help="工作分辨率宽")
    parser.add_argument("--height", type=int, default=None, help="工作分辨率高")
    parser.add_argument("--front-output", type=str, default=None,
                        help="前视结果视频保存路径 (可选)")
    parser.add_argument("--bottom-output", type=str, default=None,
                        help="下视结果视频保存路径 (可选)")
    parser.add_argument("--show", action="store_true", default=None, help="显示检测窗口")
    parser.add_argument("--no-show", action="store_false", dest="show", default=None,
                        help="不显示窗口 (RDK 无显示时可关闭)")
    return parser.parse_args()


def build_config(opt):
    """合并配置: main_rdk_config 为基准, 命令行参数覆盖同名项。"""
    cfg = get_rdk_config()
    # 输入源
    if opt.front_source is not None:
        cfg["front_source"] = opt.front_source
    if opt.bottom_source is not None:
        cfg["bottom_source"] = opt.bottom_source
    if opt.width is not None:
        cfg["work_width"] = opt.width
    if opt.height is not None:
        cfg["work_height"] = opt.height
    # P2/P4 球检测 (hbm)
    if opt.model is not None:
        cfg["p2_model_path"] = opt.model
    if opt.label_file is not None:
        cfg["p2_label_file"] = opt.label_file
    if opt.target is not None:
        cfg["p2_target_class_names"] = opt.target
    if opt.conf is not None:
        cfg["p2_score_thres"] = opt.conf
    if opt.iou is not None:
        cfg["p2_nms_thres"] = opt.iou
    # 显示与输出
    if opt.show is not None:
        cfg["show"] = opt.show
    if opt.front_output is not None:
        cfg["front_output"] = opt.front_output
    if opt.bottom_output is not None:
        cfg["bottom_output"] = opt.bottom_output

    # 各任务 overrides 复用前视/下视配置 (避免多份配置漂移)
    cfg["p1_overrides"] = _bottom_mod.get_config()["p1_overrides"]
    cfg["p3_overrides"] = _front_mod.get_config()["p3_overrides"]
    cfg["p5_overrides"] = _bottom_mod.get_config()["p5_overrides"]
    return cfg


def _run_safe(name, tester, errors):
    """在线程中运行单个测试器; 异常记录到 errors 列表, 不中断另一路。"""
    try:
        tester.run()
    except Exception as exc:
        errors.append(f"{name}: {exc!r}")
        print(f"[MAIN] {name} 摄像头异常退出: {exc!r}")


def _run_cameras(front, bottom):
    """并发运行前视/下视两个测试器, 各用一个窗口。

    当任一路结束 (视频播完 / 用户按 q / 异常) 时, 通过 request_stop()
    协调另一路同时干净退出。
    """
    errors = []
    ft = threading.Thread(target=_run_safe, args=("前视", front, errors),
                          name="front-cam", daemon=True)
    bt = threading.Thread(target=_run_safe, args=("下视", bottom, errors),
                          name="bottom-cam", daemon=True)
    ft.start()
    bt.start()

    while ft.is_alive() and bt.is_alive():
        time.sleep(0.1)

    # 先结束的一路触发另一路协调停止
    if not ft.is_alive():
        bottom.request_stop()
    if not bt.is_alive():
        front.request_stop()

    ft.join()
    bt.join()

    if errors:
        raise RuntimeError(f"双摄像头运行异常: {errors[0]}")


def main():
    opt = parse_args()
    cfg = build_config(opt)
    work_size = (cfg["work_width"], cfg["work_height"])

    # ---- 球检测 (hbm, 复用 P2 撞球/main.py; 前视/下视各一个实例) ----
    print(f"[P2/P4] 创建 hbm 球检测器 (前视 cores={cfg['p2_bpu_cores_front']}, "
          f"下视 cores={cfg['p2_bpu_cores_bottom']}), 目标类别: "
          f"{cfg['p2_target_class_names'] or '全部'}")
    front_ball = create_ball_detector(cfg, cfg["p2_bpu_cores_front"])
    bottom_ball = create_ball_detector(cfg, cfg["p2_bpu_cores_bottom"])

    # ---- 前视测试器: P2 撞球 + P3 穿门 ----
    door_detector = create_door_detector(work_size, cfg["p3_overrides"])
    front = FrontCamTester(
        source=cfg["front_source"],
        ball_detector=front_ball,
        door_detector=door_detector,
        work_size=work_size,
        show=cfg["show"],
        output=cfg["front_output"],
    )
    print(f"[FRONT] 前视测试器就绪, 源: {cfg['front_source']} (P2 撞球 + P3 穿门)")

    # ---- 下视测试器: P1 巡线 + P4 抓球 + P5 放球 ----
    line_detector = create_line_detector(cfg["p1_overrides"])
    box_detector = create_box_detector(work_size, cfg["p5_overrides"])
    bottom = BottomCamTester(
        source=cfg["bottom_source"],
        line_detector=line_detector,
        ball_detector=bottom_ball,
        box_detector=box_detector,
        work_size=work_size,
        show=cfg["show"],
        output=cfg["bottom_output"],
    )
    print(f"[BOTTOM] 下视测试器就绪, 源: {cfg['bottom_source']} (P1 巡线 + P4 抓球 + P5 放球)")

    # ---- 双摄像头并发运行 (两个窗口) ----
    _run_cameras(front, bottom)


if __name__ == "__main__":
    main()
