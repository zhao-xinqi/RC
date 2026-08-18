"""P2 撞球 - YOLO 目标检测（detect）推理入口脚本。

本脚本基于 main_old.py 精简改写，任务仅为 detect（目标检测），
默认使用当前目录下编译好的 best 球检测模型（best_nashe_320x320_nv12.hbm）
在 RDK 板端 BPU（HBM 运行时）上完成单张图像的检测推理。

所有参数集中管理在 config.py 中；命令行参数可覆盖 config 中的默认值。
后续如需在推理前进行水下预处理，打开 config 中的 enable_preprocess 即可，
此时会自动调用共享的 0.预处理 模块（集成方式参考 P1 巡线）。

运行环境：
    - RDK 板端，需安装 hbm_runtime
    - 输入图像为 BGR 格式
    - utils 工具包完全引用当前目录下的 ./utils（未做任何修改）

用法示例：
    python main.py --test-img test.jpg
    python main.py --test-img test.jpg --score-thres 0.3 --nms-thres 0.5
    python main.py --test-img test.jpg --label-file ball.names --img-save-path out.jpg
"""

import argparse
import importlib.util
import os
import sys
import cv2

# 将本文件所在目录加入 sys.path，确保可引用当前目录下的 utils 工具包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_detect import YoloDetect, YoloDetectConfig

# 完全引用当前目录下的 utils 工具包
import utils.py_utils.file_io as file_io
import utils.py_utils.visualize as visualize
import utils.py_utils.inspect as inspect


# ================================================================
# 动态加载模块 (文件夹名含数字/中文, 无法用标准 import)
# ================================================================
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # V2.0 根目录
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 本模块目录

# 加载本模块 config.py (唯一命名, 避免与其它任务模块的 config 冲突)
_CONFIG_PATH = os.path.join(_MODULE_DIR, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("P2_config", _CONFIG_PATH)
_cfg_module = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg_module)
DEFAULT_CONFIG = _cfg_module.DEFAULT_CONFIG   # 默认配置
get_config = _cfg_module.get_config           # 获取配置副本

# 加载 0.预处理 (可选; 模块缺失时跳过, enable_preprocess 自动失效)
_PREP_PATH = os.path.join(_PROJ_ROOT, "0.预处理", "preprocessor.py")
preprocess = None
if os.path.exists(_PREP_PATH):
    _prep_spec = importlib.util.spec_from_file_location("P2_preprocessor", _PREP_PATH)
    _prep_module = importlib.util.module_from_spec(_prep_spec)
    _prep_spec.loader.exec_module(_prep_module)
    preprocess = _prep_module.preprocess


def parse_args() -> argparse.Namespace:
    """解析命令行参数；未指定的参数使用 config.py 中的默认值。"""
    parser = argparse.ArgumentParser(
        description="P2 撞球 - YOLO detect 推理 (基于 HBM 运行时)")
    parser.add_argument('--model-path', type=str, default=None,
                        help='BPU 量化模型 *.hbm 路径 (默认取 config.py 的 model_path)')
    parser.add_argument('--priority', type=int, default=None,
                        help='模型推理优先级 (0~255), 0 最低, 255 最高')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=None,
                        help='BPU 核心索引列表, 例如 --bpu-cores 0 1')
    parser.add_argument('--test-img', type=str, default=None,
                        help='输入测试图像路径；不指定时默认调用摄像头实时识别')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='摄像头设备编号 (默认 0)')
    parser.add_argument('--label-file', type=str, default=None,
                        help='类别名称文件路径 (每行一个类别名); '
                             '不提供时按类别 id 显示')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='检测结果图像保存路径 (默认: result.jpg)')
    parser.add_argument('--score-thres', type=float, default=None,
                        help='置信度阈值 (默认取 config.py 的 score_thres)')
    parser.add_argument('--nms-thres', type=float, default=None,
                        help='NMS 非极大值抑制的 IoU 阈值 (默认取 config.py 的 nms_thres)')
    parser.add_argument('--enable-preprocess', action='store_true', default=None,
                        help='强制开启 0.预处理 水下预处理 (覆盖 config.py)')
    parser.add_argument('--no-display', action='store_true', default=False,
                        help='实时识别时不显示窗口，只打印检测信息')
    return parser.parse_args()


def build_config(opt: argparse.Namespace) -> dict:
    """合并配置：以 config.py 为基准，命令行参数覆盖同名项。"""
    cfg = get_config()  # config.py 默认配置的深拷贝
    if opt.model_path is not None:
        cfg['model_path'] = opt.model_path
    if opt.score_thres is not None:
        cfg['score_thres'] = opt.score_thres
    if opt.nms_thres is not None:
        cfg['nms_thres'] = opt.nms_thres
    if opt.priority is not None:
        cfg['priority'] = opt.priority
    if opt.bpu_cores is not None:
        cfg['bpu_cores'] = opt.bpu_cores
    if opt.enable_preprocess is not None:
        cfg['enable_preprocess'] = opt.enable_preprocess
    return cfg


def resolve_model_path(model_path: str) -> str:
    """解析模型路径：相对路径基于本模块目录展开。"""
    if not os.path.isabs(model_path):
        model_path = os.path.join(_MODULE_DIR, model_path)
    return model_path


def load_labels(label_file: str) -> list:
    """加载类别名称列表；未指定或文件不存在时返回空列表。"""
    if not label_file or not os.path.exists(label_file):
        return []
    return file_io.load_class_names(label_file)


def create_detector(cfg: dict, model_path: str) -> YoloDetect:
    """根据配置构造并加载 YOLO detect 检测器。

    说明：特征图网格大小（anchor_sizes）需与模型输入分辨率匹配，
    本函数根据加载模型的实际输入高度自动计算：
        640x640 输入 → [80, 40, 20]
        320x320 输入 → [40, 20, 10]
    """
    config = YoloDetectConfig(
        model_path=model_path,
        score_thres=cfg['score_thres'],
        nms_thres=cfg['nms_thres'],
        strides=cfg['strides'],
    )
    model = YoloDetect(config)

    # 根据模型实际输入分辨率修正各尺度特征图网格大小
    model.cfg.anchor_sizes = [model.input_h // s for s in model.cfg.strides]

    # 配置推理调度参数 (优先级与 BPU 核心)
    model.set_scheduling_params(priority=cfg['priority'], bpu_cores=cfg['bpu_cores'])
    return model


def compute_center_offset(box: tuple, center_point: tuple) -> tuple:
    """计算检测框中心相对参考中心点的偏移量。

    约定：x 轴正方向为向右，y 轴正方向为向上。
    因此相对中心点的偏移为：
        dx = box_center_x - center_x
        dy = center_y - box_center_y
    """
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    center_x, center_y = center_point
    dx = cx - center_x
    dy = center_y - cy
    return dx, dy


def print_detection_offsets(boxes, scores, cls_ids, center_point: tuple, labels=None) -> None:
    """打印每个检测框的中心相对中心点偏移，并按右/上为正输出。"""
    if boxes is None or len(boxes) == 0:
        print(f"未检测到目标，参考中心点为: {center_point}")
        return

    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, cls_ids)):
        dx, dy = compute_center_offset(tuple(box.tolist()), center_point)
        label = labels[cls_id] if labels and cls_id < len(labels) else str(int(cls_id))
        print(
            f"[检测 {i}] 类别={label} 置信度={score:.3f} "
            f"box_center=({(box[0]+box[2])/2:.1f}, {(box[1]+box[3])/2:.1f}) "
            f"相对中心点偏移: dx={dx:.1f}, dy={dy:.1f} "
            f"(右/上为正)"
        )


def run_detect(cfg: dict, opt: argparse.Namespace) -> None:
    """执行单张图像的 YOLO detect 推理主流程。"""
    # 1. 解析并检查模型路径
    model_path = resolve_model_path(cfg['model_path'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 2. 加载输入图像 (BGR)
    img = file_io.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]
    print(f"输入图像: {opt.test_img}, 尺寸: {img_w}x{img_h}")

    # 3. (可选) 水下预处理: 调用共享的 0.预处理 模块
    if cfg['enable_preprocess']:
        if preprocess is None:
            print("[警告] 已开启 enable_preprocess, 但未找到 0.预处理 模块, 跳过预处理")
        else:
            print("正在进行水下预处理 (0.预处理) ...")
            img = preprocess(img, cfg)

    # 4. 加载类别名称 (可选)
    labels = load_labels(opt.label_file)

    # 5. 构造检测器并打印模型信息
    model = create_detector(cfg, model_path)
    inspect.print_model_info(model.model)

    # 6. 推理 (内部完成 预处理 -> BPU 推理 -> 后处理)
    boxes, scores, cls_ids = model.predict(img)

    # 7. 打印并可视化检测结果
    visualize.print_detections(boxes, scores, cls_ids, labels)
    print_detection_offsets(boxes, scores, cls_ids, cfg.get('center_point', (img_w / 2, img_h / 2)), labels)
    result_img = visualize.draw_boxes(
        img, boxes, cls_ids, scores, labels, visualize.rdk_colors)

    # 8. 保存结果图像
    cv2.imwrite(opt.img_save_path, result_img)
    print(f"[完成] 检测结果已保存至: {opt.img_save_path}")


def run_live_detect(cfg: dict, camera_id: int = 0, show_window: bool = True) -> None:
    """使用 OpenCV 调用摄像头进行实时检测，并打印检测框相对中心点的偏移。"""
    # 1. 解析并检查模型路径
    model_path = resolve_model_path(cfg['model_path'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"正在打开摄像头: {camera_id} ...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {camera_id}")

    labels = load_labels(None)
    model = create_detector(cfg, model_path)

    print(f"参考中心点: {cfg.get('center_point', (320, 240))} (x 右正, y 上正)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头帧失败，退出 ...")
            break

        # 这里不对图像做额外缩放，直接喂给模型，保持原始帧内容
        boxes, scores, cls_ids = model.predict(frame)
        print_detection_offsets(boxes, scores, cls_ids, cfg.get('center_point', (frame.shape[1] / 2, frame.shape[0] / 2)), labels)

        if show_window:
            result_img = visualize.draw_boxes(
                frame, boxes, cls_ids, scores, labels, visualize.rdk_colors)
            cv2.imshow('P2 撞球实时检测', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def main() -> None:
    """程序入口。支持单张图像模式和实时摄像头模式。"""
    opt = parse_args()
    cfg = build_config(opt)

    if opt.test_img is not None:
        run_detect(cfg, opt)
    else:
        run_live_detect(cfg, camera_id=opt.camera_id, show_window=not opt.no_display)


if __name__ == "__main__":
    main()
