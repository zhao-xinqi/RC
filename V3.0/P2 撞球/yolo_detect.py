# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa: E501
# flake8: noqa: E402

"""基于 DFL 回归的 YOLO 目标检测推理封装与流水线工具。

本模块基于 HBM 运行时实现了一个轻量级的 YOLO 检测封装，支持基于
DFL（分布焦点损失）回归的检测模型，包括 YOLOv5u、YOLOv8、YOLO11
与 YOLO12，它们共享相同的无锚框（anchor-free）框解码逻辑。

主要特性：
    - YoloDetectConfig 数据类：用于配置模型参数。
    - YoloDetect 类：提供 pre_process、forward、post_process、predict
      以及 __call__ 方法。
    - 通过 DFL 回归输出进行无锚框框解码。
    - 按类别分别进行非极大值抑制（NMS）。

典型用法：
    >>> from yolo_detect import YoloDetect, YoloDetectConfig
    >>> cfg = YoloDetectConfig(model_path="/path/to/yolo11n_detect.hbm")
    >>> model = YoloDetect(cfg)
    >>> boxes, scores, cls_ids = model(img)

说明：
    - 运行环境需安装 hbm_runtime。
    - 输入图像默认期望为 BGR 格式。
    - 检测头采用无锚框的 DFL 回归：每个检测尺度输出一对 分类输出 与
      框分布输出（每个预测包含 4 * reg 个回归值）。
"""

import os
import sys
import hbm_runtime
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# 将工程根目录加入 sys.path，以便导入工具模块。
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.preprocess as pre_utils
import utils.py_utils.postprocess as post_utils


@dataclass
class YoloDetectConfig:
    """YoloDetect 模型初始化配置。

    该数据类保存模型路径以及 YOLO 检测流水线中预处理、推理、后处理所需的
    全部运行时参数，适用于所有基于 DFL 的 YOLO 检测模型（v5u、v8、v11、v12）。

    属性说明：
        model_path: 编译好的 YOLO 检测模型 `.hbm` 路径。
        classes_num: 检测类别数量，默认 80（COCO）。
        resize_type: 预处理时的图像缩放方式。
            - 0: 直接拉伸缩放。
            - 1: 保持宽高比的 letterbox 填充。
        score_thres: 过滤检测结果的置信度阈值。
        nms_thres: 非极大值抑制（NMS）使用的 IoU 阈值。
        reg: 每个框边 DFL 回归的离散分箱数量，默认 16。
        strides: 各检测尺度的特征图下采样倍率。
        anchor_sizes: 各检测尺度的特征图网格尺寸（像素）。
    """
    model_path: str
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.45
    reg: int = 16
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: list = field(default_factory=lambda: [80, 40, 20])


class YoloDetect:
    """基于 HB_HBMRuntime 的 YOLO DFL 检测封装。

    该类为 DFL 型 YOLO 检测模型（v5u、v8、v11、v12）提供统一的推理流水线，
    包括输入预处理、模型执行以及后处理步骤（无锚框 DFL 框解码、置信度过滤、
    非极大值抑制 NMS 等）。

    属性说明：
        model: 已加载的 HBM 运行时模型实例。
        model_name: 加载的第一个模型名称。
        input_names: 输入张量名称列表。
        output_names: 输出张量名称列表。
        input_shapes: 输入张量形状字典。
        input_h: 模型输入高度（像素）。
        input_w: 模型输入宽度（像素）。
        weights_static: DFL 离散位置权重，用于计算框的期望偏移。
        cfg: 模型配置对象。

    说明：
        所有支持的 YOLO 检测变体（v5u、v8、v11、v12）共享相同的无锚框
        DFL 检测头结构：每个检测尺度输出一对 分类输出 与 框分布输出。
    """

    def __init__(self, config: YoloDetectConfig):
        """根据给定配置初始化 YoloDetect 模型。

        Args:
            config: 包含模型路径、预处理参数与后处理参数的配置对象，
                各字段语义与约束见 `YoloDetectConfig` 数据类定义。
        """
        # 加载模型并提取元数据
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        # 模型输入分辨率 (H, W)，从输入张量形状推断
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        # DFL 权重：形状 (1, 1, reg)，用于计算期望框偏移
        self.weights_static = np.arange(config.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

        # 保存配置
        self.cfg = config

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """配置推理调度参数。

        Args:
            priority: 推理优先级，取值范围 [0, 255]。
            bpu_cores: 用于推理的 BPU 核心索引列表。

        Returns:
            None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self,
                    img: np.ndarray,
                    image_format: Optional[str] = "BGR"
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        """将输入图像预处理为模型所需的张量格式。

        输入图像会按照配置的缩放策略调整尺寸，并从 BGR 格式转换为
        NV12（Y 与 UV 两个平面）。

        Args:
            img: 输入图像数组。
            image_format: 输入图像格式，目前仅支持 `"BGR"`。

        Returns:
            嵌套输入张量字典，形式为：
            `{model_name: {input_name: tensor}}`。

        Raises:
            ValueError: 若提供了不支持的图像格式。
        """
        if image_format == "BGR":
            resize_img = pre_utils.resized_image(
                img, self.input_w, self.input_h, self.cfg.resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        else:
            raise ValueError(f"不支持的图像格式: {image_format}")

        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """执行模型推理。

        Args:
            input_tensor: 由 `pre_process()` 生成的预处理输入张量字典。

        Returns:
            运行时返回的原始输出张量字典。
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将模型原始输出转换为最终检测结果。

        该步骤包括无锚框 DFL 框解码、置信度过滤、非极大值抑制（NMS）
        以及将坐标缩放回原始图像分辨率。

        Args:
            outputs: 推理得到的原始输出张量（由 `forward()` 返回）。
            ori_img_w: 原始输入图像宽度。
            ori_img_h: 原始输入图像高度。
            score_thres: 置信度阈值覆盖值。若为 `None`，使用配置中的值。
            nms_thres: NMS 的 IoU 阈值覆盖值。若为 `None`，使用配置中的值。

        Returns:
            元组，包含：
                - boxes: 形状为 `(N, 4)` 的边界框，坐标为原始图像坐标，
                  格式为 `[x1, y1, x2, y2]`。
                - scores: 形状为 `(N,)` 的置信度分数。
                - cls_ids: 形状为 `(N,)` 的类别索引。
        """
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres

        # 计算原始 logit 过滤所需的逆 sigmoid 阈值
        conf_thres_raw = -np.log(1.0 / score_thres - 1)

        # 第一步：解码各检测尺度的 分类输出 与 框分布输出
        model_outputs = outputs[self.model_name]
        all_boxes = []
        all_scores = []
        all_ids = []
        for i, (stride, anchor_size) in enumerate(
                zip(self.cfg.strides, self.cfg.anchor_sizes)):
            cls_key = self.output_names[2 * i]      # 分类 logits 输出
            box_key = self.output_names[2 * i + 1]  # DFL 框分布输出

            # 在 sigmoid 之前先按原始 logit 阈值过滤
            scores, ids, valid_indices = post_utils.filter_classification(
                model_outputs[cls_key], conf_thres_raw)

            # 对有效预测解码 DFL 边界框
            dbboxes = post_utils.decode_boxes(
                model_outputs[box_key], valid_indices,
                anchor_size, stride, self.weights_static)

            all_boxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)

        # 第二步：拼接所有检测尺度的结果
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        cls_ids = np.concatenate(all_ids, axis=0)

        # 第三步：非极大值抑制
        keep = post_utils.NMS(boxes, scores, cls_ids, nms_thres)

        # 第四步：将框坐标缩放回原始图像尺寸
        xyxy = post_utils.scale_coords_back(
            boxes[keep], ori_img_w, ori_img_h,
            self.input_w, self.input_h, self.cfg.resize_type)

        return xyxy, scores[keep], cls_ids[keep]

    def predict(self,
                img: np.ndarray,
                image_format: str = "BGR",
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在单张图像上运行完整的检测流水线。

        该方法内部依次完成预处理、推理与后处理。

        Args:
            img: 输入图像数组。
            image_format: 输入图像格式，目前支持 `"BGR"`。
            score_thres: 置信度阈值覆盖值。
            nms_thres: NMS 的 IoU 阈值覆盖值。

        Returns:
            元组，包含：
                - boxes: 形状为 `(N, 4)` 的边界框。
                - scores: 形状为 `(N,)` 的置信度分数。
                - cls_ids: 形状为 `(N,)` 的类别索引。
        """
        ori_img_h, ori_img_w = img.shape[:2]

        # 1) 预处理
        input_tensor = self.pre_process(img, image_format)

        # 2) 推理
        outputs = self.forward(input_tensor)

        # 3) 后处理
        boxes, scores, cls_ids = self.post_process(
            outputs, ori_img_w, ori_img_h, score_thres, nms_thres)

        return boxes, scores, cls_ids

    def __call__(self,
                 img: np.ndarray,
                 image_format: str = "BGR",
                 score_thres: Optional[float] = None,
                 nms_thres: Optional[float] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """检测流水线的可调用接口。

        该方法与调用 `predict()` 功能完全等价。

        Args:
            img: 输入图像数组。
            image_format: 输入图像格式。
            score_thres: 置信度阈值覆盖值。
            nms_thres: NMS 的 IoU 阈值覆盖值。

        Returns:
            与 `predict()` 相同的返回值。
        """
        return self.predict(img, image_format, score_thres, nms_thres)
