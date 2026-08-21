"""P2 撞球 - 开发机测试版（.pt 模型）

说明：
- 仅用于本地开发机/PC 端测试，不改动原有 detect 和 main 主体代码。
- 适配 Ultralytics YOLO 的 .pt 模型。
- 先调用 0.预处理 对视频帧做水下增强，再用 best.pt 识别。
- 允许只保留 red 或 blue。

运行示例:
    python main_dev_pt.py --model best.pt --source test.mp4 --target red
    python main_dev_pt.py --model best.pt --source 0 --target blue
"""

import argparse
import importlib.util
import os
from pathlib import Path

import cv2


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def load_preprocessor():
    """动态加载 0.预处理/preprocessor.py，兼容中文目录。"""
    root_dir = Path(__file__).resolve().parent.parent
    prep_path = root_dir / "0.预处理" / "preprocessor.py"
    spec = importlib.util.spec_from_file_location("preprocessor_dev", str(prep_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载预处理模块: {prep_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="P2 撞球开发机测试版（.pt 模型 + 水下预处理）")
    parser.add_argument("--model", type=str, default="best.pt", help="YOLO .pt 模型路径")
    parser.add_argument("--source", type=str, default="0", help="视频文件路径或摄像头编号，例如 0 / test.mp4")
    parser.add_argument("--target", type=str, default="red", choices=["red", "blue"], help="仅保留 red 或 blue")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--output", type=str, default="dev_pt_result.mp4", help="输出视频保存路径")
    parser.add_argument("--show", action="store_true", default=True, help="显示检测窗口")
    parser.add_argument("--no-show", action="store_false", dest="show", help="不显示窗口")
    return parser.parse_args()


def build_preprocess_config():
    return {
        "image_width": 640,
        "image_height": 480,
        "enable_resize": True,
        "enable_color_correct": True,
        "red_boost": 1.2,
        "enable_gaussian": True,
        "gaussian_kernel": 5,
        "enable_clahe": True,
        "clahe_clip": 2.0,
        "clahe_tile": (8, 8),
    }


def draw_target_boxes(frame, result, target_name: str):
    """只画出指定类别的检测框。"""
    if result is None or result.boxes is None:
        return frame

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        if _normalize_name(label) != target_name:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return frame


def main():
    opt = parse_args()
    target_name = _normalize_name(opt.target)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请先执行: pip install ultralytics") from exc

    model_path = opt.model
    if not os.path.isabs(model_path):
        model_path = str(Path(__file__).resolve().parent / model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    preprocess = load_preprocessor()
    proc_cfg = build_preprocess_config()
    model = YOLO(model_path)

    source = opt.source
    try:
        source_int = int(source)
    except ValueError:
        source_int = None

    if source_int is not None:
        cap = cv2.VideoCapture(source_int)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频或相机源: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(Path(__file__).resolve().parent / opt.output)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer = None
        print(f"[WARN] 无法创建输出视频文件: {out_path}")

    frame_idx = 0
    print(f"[PT-DEV] 开始处理视频; 目标类别={target_name}; 预处理已启用")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess(frame, proc_cfg)
        results = model(processed, conf=opt.conf, iou=opt.iou, imgsz=opt.imgsz, verbose=False)
        result = results[0]

        names = result.names if hasattr(result, "names") else model.names
        kept = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                if _normalize_name(label) == target_name:
                    kept.append((label, float(box.conf.item()), box.xyxy[0].tolist()))

        vis = draw_target_boxes(frame.copy(), result, target_name)
        cv2.putText(
            vis,
            f"Target={target_name}  detections={len(kept)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if writer is not None:
            writer.write(vis)

        if opt.show:
            cv2.imshow("P2 PT Dev - water-preprocessed + YOLO", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"[Frame {frame_idx}] 过滤后保留的目标: {kept}")
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if opt.show:
        cv2.destroyAllWindows()

    print(f"[PT-DEV] 处理完成; 输出视频已保存到: {out_path}")


if __name__ == "__main__":
    main()
