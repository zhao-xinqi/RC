#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vision.py - 摄像头 + 板端 YOLO(.hbm) 简单测试

功能:
  1. 用 OpenCV 打开摄像头 (默认 /dev/video0)
  2. 用 hobot_dnn(pyeasy_dnn) 加载板端 .hbm 模型做目标检测 (YOLOv8 / Nash 架构)
  3. 绘制检测框 + 类别/置信度标注, 实时显示或保存标注帧

用法:
  python3 test_vision.py               # 有显示环境: 弹窗实时显示 (q/ESC 退出, s 保存)
  python3 test_vision.py --once        # 无显示环境: 只处理 1 帧, 打印检测结果并保存 out_detect.jpg
  python3 test_vision.py --frames 300  # 无显示环境: 连续处理 300 帧, 每 30 帧保存一张快照
"""
import os
import sys
import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn

# ================================================================
# 可调参数
# ================================================================
MODEL_PATH = "/home/root/RC/best_nashe_320x320_nv12.hbm"
CAMERA_INDEX = 1
INPUT_SIZE = 320                 # 模型输入边长
CONF_THRESH = 0.25               # 置信度阈值 (sigmoid 置信度)
NMS_THRESH = 0.45                # NMS 阈值
STRIDES = [8, 16, 32]            # 三个检测尺度
CLASS_NAMES = ["class0", "class1"]  # 类别名, 按实际模型类别修改
CAM_W, CAM_H = 640, 480          # 摄像头分辨率
USE_GUI = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ================================================================
# 预处理
# ================================================================

def bgr2nv12(image):
    """BGR 图像 -> NV12 一维数据 (地瓜 BPU 模型输入格式)"""
    h, w = image.shape[:2]
    area = h * w
    i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(area * 3 // 2)
    y = i420[:area]
    uv = i420[area:].reshape(2, area // 4).transpose(1, 0).reshape(area // 2)
    return np.concatenate([y, uv]).astype(np.uint8)


def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ================================================================
# 解码 (YOLOv8: 每尺度输出 类别分[H,W,C] + DFL框分支[H,W,4,16])
# ================================================================

def valid_buffer(tensor):
    """裁剪 BPU 输出 buffer 的尾部对齐 padding, 只保留有效 shape 数据"""
    h, w, c = tensor.properties.shape[1], tensor.properties.shape[2], tensor.properties.shape[3]
    return tensor.buffer[:h * w * c].reshape(h, w, c)


def decode(model, nv12, ori_shape):
    """
    推理 + 解码, 返回检测列表, 每项 [x1,y1,x2,y2,conf,cls_id] (原图像素)

    输出语义(与 D-Robotics 官方 ultralytics_yolo 后处理一致):
      偶数索引 = 类别分支 (原始 logit), 奇数索引 = DFL 框分支 (4x16)
      conf = sigmoid(最大类logit), 不是 softmax (softmax 对少类别模型会普遍偏高)
    """
    outputs = model.forward(nv12)          # 6 个输出
    dets = []
    conf_thresh_raw = -np.log(1.0 / CONF_THRESH - 1.0)   # sigmoid 阈值对应的 raw 阈值
    for s, stride in enumerate(STRIDES):
        cls = valid_buffer(outputs[2 * s]).reshape(-1, len(CLASS_NAMES))
        box = valid_buffer(outputs[2 * s + 1]).reshape(-1, 4, 16)
        # 类别: 最大原始 logit, sigmoid 得到置信度
        raw = cls.max(axis=1)
        keep = raw >= conf_thresh_raw
        if not keep.any():
            continue
        conf = 1.0 / (1.0 + np.exp(-raw[keep]))     # sigmoid
        cid = cls[keep].argmax(axis=1)
        # 框: DFL 解码 -> [l,t,r,b] 距网格中心的距离 (输入像素)
        dist = (softmax(box[keep].reshape(-1, 4, 16), axis=2) * np.arange(16)).sum(axis=2)
        # 网格中心坐标 (YOLOv8 为 anchor-free)
        grid = INPUT_SIZE // stride
        xx, yy = np.meshgrid(np.arange(grid), np.arange(grid))
        cx = (xx.ravel() + 0.5) * stride
        cy = (yy.ravel() + 0.5) * stride
        cx, cy = cx[keep], cy[keep]
        # 映射回原图尺寸
        sx, sy = ori_shape[1] / INPUT_SIZE, ori_shape[0] / INPUT_SIZE
        x1 = (cx - dist[:, 0]) * sx
        y1 = (cy - dist[:, 1]) * sy
        x2 = (cx + dist[:, 2]) * sx
        y2 = (cy + dist[:, 3]) * sy
        for i in range(len(conf)):
            dets.append([x1[i], y1[i], x2[i], y2[i], conf[i], cid[i]])
    return dets


def nms(dets, thresh=NMS_THRESH):
    """经典 NMS (基于 IoU), 返回过滤后的列表"""
    if not dets:
        return []
    boxes = np.array(dets, dtype=np.float32)
    x1, y1, x2, y2, score = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = score.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < thresh]
    return [dets[k] for k in keep]


def draw(frame, dets):
    """在帧上绘制检测框与标注"""
    for x1, y1, x2, y2, conf, cid in dets:
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = "{} {:.2f}".format(CLASS_NAMES[int(cid)], conf)
        cv2.putText(frame, label, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ================================================================
# 主流程
# ================================================================

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_once = True
        total_frames = 1
    elif len(sys.argv) > 1 and sys.argv[1] == "--frames":
        run_once = False
        total_frames = int(sys.argv[2])
    else:
        run_once = False
        total_frames = 0          # 0 = 不限, 直到用户退出

    print("[*] 加载模型:", MODEL_PATH)
    model = dnn.load(MODEL_PATH)[0]
    print("[*] 模型输入:", model.inputs[0].properties.shape,
          " 输出数:", len(model.outputs))

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("[!] 无法打开摄像头 /dev/video%d" % CAMERA_INDEX)
        return

    if USE_GUI and not run_once:
        print("[*] GUI 模式: q/ESC 退出, s 保存截图")
    else:
        print("[*] 无显示模式: 每 30 帧保存一张标注快照")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] 读帧失败")
            break

        # 检测
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        dets = nms(decode(model, bgr2nv12(resized), frame.shape[:2]))
        draw(frame, dets)

        # 显示 / 保存
        if USE_GUI and not run_once:
            cv2.imshow("test_vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                cv2.imwrite("out_detect.jpg", frame)
                print("[*] 已保存 out_detect.jpg")
        else:
            if dets:
                for d in dets:
                    print("  det: cls=%s conf=%.2f bbox=[%.0f,%.0f,%.0f,%.0f]"
                          % (CLASS_NAMES[int(d[5])], d[4], d[0], d[1], d[2], d[3]))
            if frame_idx % 30 == 0:
                name = "out_%04d.jpg" % (frame_idx // 30)
                cv2.imwrite(name, frame)
                print("[*] 已保存 %s (帧 %d)" % (name, frame_idx))

        frame_idx += 1
        if run_once or (total_frames and frame_idx >= total_frames):
            break

    if not run_once:
        cv2.imwrite("out_detect.jpg", frame)
        print("[*] 已保存最终帧 out_detect.jpg")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
