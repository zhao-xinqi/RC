"""
前视摄像头主程序 (v4.0 / camera_front)

功能:
  以 ~200fps 高速采集前视摄像头画面, 对每帧执行 水下预处理 + YOLO 目标检测
  (撞球任务), 节流显示标注帧. 控制台逐帧输出目标中心点及相对预设 POINT 的
  偏移 (右/上为正); 采集/处理/已处理等统计写入 ./camera_front/result/*.log,
  不占用控制台.

模块复用 (不重写, 全部引用已有实现):
  - 预处理          : v4.0/pre_process/preprocessor.py 的 preprocess(frame, config)
  - YOLO 模型/后处理 : v4.0/YOLO/yolo_detect.py  (hbm, RDK 板端 BPU)
                      或 v4.0/YOLO/best.pt       (ultralytics, 开发机测试)

高帧率方案 (参考 test/README.md):
  采集线程独占 cap.read() -> 有界队列(满丢最旧) -> N个检测worker(独立模型)
  -> 发布最新标注帧 -> 主循环按固定节拍节流显示 (~60fps).
  标注帧同时经后台线程写入 ./camera_front/result/*.mp4 (保存不占推理时间).

关键点:
  - 相机参数固定: MSMF + MJPG + 640x480 + CAP_PROP_FPS=1000 (链路实测 ~204fps)
  - 主循环必须 time.sleep(0.001) 让出 GIL, 否则死循环饿死采集线程 (掉到 ~35fps)
  - 收尾用 stop_evt + get(timeout), 不用队列哨兵 (满队列会阻塞挂死)

运行示例:
  python main_front.py --device 1                          # camera 1 (索引由配置/命令行指定)
  python main_front.py --device 0 --backend ultralytics    # 开发机 (默认 worker8 x 批4)
  python main_front.py --no-show                           # 无窗口模式 (只打印 fps/概况)
  python main_front.py --no-save                           # 不保存检测视频
  python main_front.py --no-log                            # 不写统计 log
  检测视频默认保存到 ./camera_front/result/front_detect_<时间戳>.mp4
  统计 log 默认保存到 ./camera_front/result/front_log_<时间戳>.log
"""

#====================
#库导入
#====================
import argparse
import importlib.util
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np


#====================
#参数配置
#====================

#-----------------------------------------------------------
# 1. 摄像头参数配置 (200fps 高帧率, 依据 test/README.md)
#-----------------------------------------------------------
CAMERA_INDEX = 0        # 前视摄像头索引 (0/1, 按实际设备在配置中修改)
FORMAT = 'MJPG'         # 必须 MJPG 才能上 200fps (YUYV 顶死 USB2.0 只有 60fps)
IMAGE_WIDTH = 640       # 实测 "最高分辨率 + 最高帧率" 的最优组合
IMAGE_HEIGHT = 480
FPS_TARGET = 1000       # 请求超高帧率, 逼相机协商出上限 (协商出 400, 链路实测 ~204)
POINT = (320, 240)      # 预设参考点 (画面中心), 控制台输出的"目标相对位置"以此为准

#-----------------------------------------------------------
# 2. 预处理参数配置 (传给 v4.0/pre_process/preprocessor.py)
#-----------------------------------------------------------
ENABLE_PREPROCESS = False   # 是否在 YOLO 前先做水下预处理; 关闭可提升吞吐
PREPROCESS_CFG = {
    'enable_resize': False,            # 摄像头已直出 640x480, 无需重复缩放
    'enable_color_correct': True,      # 水下颜色校正 (红光补偿, 利于红球检出)
    'red_boost': 1.2,                  # 红色通道增强系数
    'enable_gaussian': True,           # 高斯去噪
    'gaussian_kernel': 5,              # 高斯核大小 (奇数)
    'enable_clahe': True,              # CLAHE 自适应直方图均衡 (应对光照不均)
    'clahe_clip': 2.0,                 # CLAHE 对比度限幅
    'clahe_tile': (8, 8),              # CLAHE 网格大小
}

#-----------------------------------------------------------
# 3. YOLO 检测参数配置 (模型/后处理取自 v4.0/YOLO)
#-----------------------------------------------------------
BACKEND = 'hbm'         # 推理后端: 'hbm'(RDK板端BPU) / 'ultralytics'(开发机测试)
# 模型文件名 (相对 v4.0/YOLO 目录; 支持绝对路径)
#   hbm 后端        -> best_nashe_320x320_nv12.hbm
#   ultralytics后端 -> best.pt (不传 --model 时自动切换)
MODEL_PATH = 'best_nashe_320x320_nv12.hbm'
LABEL_FILE = None       # 类别名称文件 (每行一个类名, 可选); 无则显示类别 id
SCORE_THRES = 0.25      # 置信度阈值, 过滤检测结果
NMS_THRES = 0.45        # NMS 非极大值抑制的 IoU 阈值
STRIDES = [8, 16, 32]   # 检测头各尺度下采样倍率 (与模型结构一致, 一般不改)
PRIORITY = 0            # BPU 推理优先级 (0~255)
BPU_CORES = [0]         # BPU 核心索引列表 (多核可设 [0, 1])
TARGET_CLASS_NAMES = [] # 只保留的类别名 (如 ['red']); 空列表保留全部

#-----------------------------------------------------------
# 4. 线程/显示参数配置 (高帧率管线, 依据 test/README.md)
#-----------------------------------------------------------
WORKERS = 8             # 检测 worker 线程数 (各自独立模型实例)
                        #   开发机 ultralytics 建议 8 (test_yolo.py 最优组合);
                        #   RDK 板端 hbm 建议 1 (BPU 共享, 多实例无吞吐增益, 可改回 1)
BATCH_SIZE = 4          # 每 worker 机会式批处理最大帧数 (ultralytics 建议 4; hbm 逐帧处理)
LOG_INTERVAL = 1.0      # 统计写入 log 文件的间隔(秒)
SHOW_INTERVAL = 1.0 / 60  # 显示节流: 主循环每 SHOW_INTERVAL 秒显示最新标注帧
GET_TIMEOUT = 0.2       # worker 取帧超时(秒): 空队列时回去检查 stop_evt, 保证能退出
SHOW = True             # 是否显示检测窗口
WIN_NAME = 'FrontCam | YOLO @200fps'
SAVE_VIDEO = True       # 是否将检测标注帧保存为视频
SAVE_DIR = 'result'     # 保存目录 (相对本模块 camera_front)
# 视频的"播放帧率"(VideoWriter 创建时固定, 只是播放元数据):
# 帧是按实际处理速率逐帧写入的 (~200fps), 想 1 秒真实时间 = 1 秒视频,
# 就把 SAVE_FPS 设为实际处理帧率 (管线设计 ~200). 低于实际 → 慢动作, 高于实际 → 快进.
# 结束时保存线程会打印实际写盘帧率, 据此微调即可.
SAVE_FPS = 200
ENABLE_LOG = True       # 是否把 采集/处理/已处理 统计写入 result 目录下的 log 文件

#-----------------------------------------------------------
# 5. 路径配置 (模块复用定位)
#-----------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # v4.0 根目录
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 本模块目录 (camera_front)
_YOLO_DIR = os.path.join(_PROJ_ROOT, 'YOLO')                              # YOLO 模块目录
_PREP_PATH = os.path.join(_PROJ_ROOT, 'pre_process', 'preprocessor.py')   # 预处理文件
_SAVE_DIR = os.path.join(_MODULE_DIR, SAVE_DIR)                           # 结果保存目录


#====================
#辅助函数
#====================

#-----------------------------------------------------------
# 动态加载模块 (文件夹名含中文/数字, 无法用标准 import)
#-----------------------------------------------------------
def load_module_by_path(module_name, path):
    """按文件路径动态加载 Python 模块 (兼容中文/数字目录名)."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载模块: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_preprocess():
    """加载 v4.0/pre_process 的通用水下预处理函数 preprocess(frame, config)."""
    if not os.path.exists(_PREP_PATH):
        raise FileNotFoundError(f'预处理模块不存在: {_PREP_PATH}')
    return load_module_by_path('v4_preprocessor', _PREP_PATH).preprocess


def load_yolo_module():
    """加载 v4.0/YOLO/yolo_detect.py (hbm 检测封装).

    该模块内部会 import utils.py_utils, 因此先把 v4.0/YOLO 加入 sys.path,
    确保 utils 包解析到 v4.0/YOLO/utils (而不是其它同名目录).
    """
    if _YOLO_DIR not in sys.path:
        sys.path.insert(0, _YOLO_DIR)
    return load_module_by_path('v4_yolo_detect', os.path.join(_YOLO_DIR, 'yolo_detect.py'))


def load_labels(label_file):
    """读取类别名称文件 (每行一个类别名), 未配置/不存在时返回空列表."""
    if not label_file or not os.path.exists(label_file):
        return []
    with open(label_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def normalize_name(name):
    """标准化类别名, 兼容 大小写/空格/下划线/短横线 差异."""
    return str(name).strip().lower().replace('_', ' ').replace('-', ' ')


def resolve_model_path(model_path):
    """解析模型/标签路径: 相对路径基于 v4.0/YOLO 目录展开."""
    if not os.path.isabs(model_path):
        model_path = os.path.join(_YOLO_DIR, model_path)
    return model_path


#-----------------------------------------------------------
# 摄像头 (MSMF + MJPG + 640x480 + 高帧率请求)
#-----------------------------------------------------------
def fourcc_str(v):
    """把 fourcc 整数解码成可读字符串, 非法值返回 '-'.

    MSMF 下 fourcc 读回常不可靠 (MJPG 可能读回 '-'), 仅作参考.
    """
    try:
        if not v:
            return '-'
        s = ''.join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
        return s if s.isprintable() else '-'
    except Exception:
        return '-'


def open_camera(idx, w, h):
    """按指定索引打开相机 (MSMF + MJPG + 640x480 + 高帧率请求).

    Args:
        idx: 相机索引, 由 参数配置 CAMERA_INDEX / 命令行 --device 指定
        w, h: 目标分辨率

    Returns:
        cap: 打开的 VideoCapture; 打不开时返回 None (不做自动回退).
    真实帧率信实测 (perf_counter), 不信 cap.get(FPS).
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FORMAT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    return cap


#-----------------------------------------------------------
# 采集线程: 独占 cap.read(), 每帧推入有界分发队列
#-----------------------------------------------------------
def dispatch(q, frame):
    """有界队列入队: 满则丢最旧帧, 保证处理延迟不堆积 (实时关键).

    采集 > 处理吞吐时, 队列始终保持容量上限, 每来一帧丢一帧最旧的,
    内存占用与延迟恒定, 不会越积越多.
    """
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put(frame)


class Grabber:
    """采集线程体: 独占 read(), 每读到一帧就推入分发队列 + 累计帧数.

    采集链路不受 YOLO 处理/显示速度影响 (稳定 ~204fps);
    队列满时丢最旧帧 (dispatch 兜底), 丢帧只发生在这里.
    """

    def __init__(self, cap, out_q, stop_evt):
        self.cap = cap
        self.lock = threading.Lock()
        self.out_q = out_q
        self.stop_evt = stop_evt
        self.count = 0      # 累计成功读取帧数

    def run(self):
        """采集循环, 在独立线程中运行 (daemon)."""
        while not self.stop_evt.is_set():
            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.stop_evt.is_set():
                break        # 退出前最后读的一帧不再入队
            dispatch(self.out_q, frame)
            with self.lock:
                self.count += 1


#-----------------------------------------------------------
# YOLO 检测器封装 (复用 v4.0/YOLO, 支持 hbm / ultralytics)
#-----------------------------------------------------------
class Detector:
    """前视 YOLO 检测器封装.

    对上层提供统一接口:
      detect(frames, enable_preprocess, prep_cfg)
        -> [(annotated_bgr, detections), ...]
          - annotated_bgr: 画好检测框/标签的 BGR 图 (供显示)
          - detections:    结构化结果 [{'label','score','bbox','center'}, ...] (供控制)

    支持两种后端:
      - 'hbm':         RDK 板端, 复用 v4.0/YOLO/yolo_detect.py (BPU 推理)
      - 'ultralytics': 开发机, 复用 v4.0/YOLO/best.pt
    """

    def __init__(self, backend, model_path, labels, preprocess_fn):
        self.backend = backend
        self.labels = labels or []
        self.preprocess = preprocess_fn          # 通用水下预处理函数
        # 目标类别过滤 (名称), 空集合表示保留全部
        self.targets = {normalize_name(n) for n in TARGET_CLASS_NAMES} if TARGET_CLASS_NAMES else set()
        if self.targets and not self.labels:
            print('[警告] 配置了 TARGET_CLASS_NAMES 但未提供 LABEL_FILE, '
                  '无法按类别名过滤, 将保留全部检测结果')

        if backend == 'hbm':
            self._init_hbm(model_path)
        elif backend == 'ultralytics':
            self._init_ultralytics(model_path)
        else:
            raise ValueError(f'未知的检测后端: {backend}')

    # ---------- 后端初始化 ----------

    def _init_hbm(self, model_path):
        """加载 hbm 模型 (v4.0/YOLO/yolo_detect.py, 需 RDK 板端 hbm_runtime)."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        try:
            yolo_mod = load_yolo_module()
        except ImportError as exc:
            raise RuntimeError(
                'hbm 后端需要 RDK 板端环境 (hbm_runtime); 开发机请改用 --backend ultralytics'
            ) from exc
        cfg = yolo_mod.YoloDetectConfig(
            model_path=model_path,
            score_thres=SCORE_THRES,
            nms_thres=NMS_THRES,
            strides=STRIDES,
        )
        self.model = yolo_mod.YoloDetect(cfg)
        # 特征图网格大小按模型实际输入分辨率自动计算 (320x320 -> [40,20,10])
        self.model.cfg.anchor_sizes = [self.model.input_h // s for s in self.model.cfg.strides]
        self.model.set_scheduling_params(priority=PRIORITY, bpu_cores=BPU_CORES)

    def _init_ultralytics(self, model_path):
        """加载 ultralytics .pt 模型 (开发机测试)."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('未安装 ultralytics, 请先执行: pip install ultralytics') from exc
        self.model = YOLO(model_path)

    # ---------- 统一检测接口 ----------

    def detect(self, frames, enable_preprocess, prep_cfg):
        """对一帧(或一批帧)执行 预处理 + YOLO, 返回 [(标注帧, 检测列表), ...].

        Args:
            frames: 单帧 numpy 或帧列表
            enable_preprocess: 是否先做水下预处理
            prep_cfg: 预处理参数字典 (传给 preprocess)
        """
        if isinstance(frames, np.ndarray):
            frames = [frames]
        if enable_preprocess:
            frames = [self.preprocess(f, prep_cfg) for f in frames]

        if self.backend == 'ultralytics':
            return self._detect_ultralytics(frames)
        return self._detect_hbm(frames)

    def _detect_hbm(self, frames):
        """hbm 后端: 逐帧 predict (BPU 单帧推理), 统一格式化."""
        results = []
        for frame in frames:
            boxes, scores, cls_ids = self.model.predict(frame)
            dets = self._format_detections(boxes, scores, cls_ids)
            results.append((draw_detections(frame, dets), dets))
        return results

    def _detect_ultralytics(self, frames):
        """ultralytics 后端: 批量 predict (一次前向吃掉一批, 榨满 GPU)."""
        preds = self.model.predict(frames, conf=SCORE_THRES, iou=NMS_THRES, verbose=False)
        results = []
        for frame, pred in zip(frames, preds):
            dets = []
            if pred.boxes is not None:
                names = pred.names
                for box in pred.boxes:
                    cls_id = int(box.cls.item())
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    if not self._is_target(label):
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    dets.append(self._make_detection(label, float(box.conf.item()), x1, y1, x2, y2))
            results.append((draw_detections(frame, dets), dets))
        return results

    def _format_detections(self, boxes, scores, cls_ids):
        """把 hbm 后处理的 (boxes, scores, cls_ids) 转为统一格式."""
        dets = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            cls_id = int(cls_id)
            label = self.labels[cls_id] if self.labels and cls_id < len(self.labels) else str(cls_id)
            if not self._is_target(label):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            dets.append(self._make_detection(label, float(score), x1, y1, x2, y2))
        return dets

    @staticmethod
    def _make_detection(label, score, x1, y1, x2, y2):
        """构造统一格式的单个检测结果字典."""
        return {
            'label': label,
            'score': float(score),
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
        }

    def _is_target(self, label):
        """按配置的目标类别过滤; 未配置时保留全部."""
        if not self.targets:
            return True
        return normalize_name(label) in self.targets

    def warmup(self):
        """首次推理预热 (CUDA/BPU 上下文与算子初始化很慢, 不预热会拖慢启动)."""
        black = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.detect([black], enable_preprocess=False, prep_cfg=None)


def draw_detections(frame, detections):
    """在帧上绘制检测结果 (外接框 + 中心点 + 类别置信度)."""
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = det['center']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)   # 黄色外接框
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)              # 绿色中心点
        cv2.putText(vis, f"{det['label']} {det['score']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


#-----------------------------------------------------------
# 线程共享状态
#-----------------------------------------------------------
class Stats:
    """多 worker 共享的处理统计 (锁保护)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0    # 已跑过 YOLO 的帧数
        self.det_frames = 0   # 检出目标的帧数

    def add(self, n_det):
        """记录一帧处理完成, n_det 为该帧检测框数量."""
        with self.lock:
            self.processed += 1
            if n_det:
                self.det_frames += 1

    def snapshot(self):
        """返回 (processed, det_frames)."""
        with self.lock:
            return self.processed, self.det_frames


class LatestResult:
    """所有 worker 共享的最新标注结果 (标注帧 + 检测列表).

    主循环按固定时间节拍取最新一张显示, 显示帧率因此只由主循环节拍决定 (~60fps),
    不受"只显示单个 worker 自己的帧"拖累.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.annotated = None   # 最新标注帧 (BGR), 尚未发布时为 None
        self.detections = []    # 最新检测列表
        self.count = 0          # 已发布帧计数 (主循环据此感知"新帧", 逐帧输出位置)

    def publish(self, annotated, detections):
        """发布一帧最新标注结果 (任意 worker 调用, 不阻塞)."""
        with self.lock:
            self.annotated = annotated
            self.detections = detections
            self.count += 1

    def latest(self):
        """取当前最新标注结果 (标注帧, 检测列表, 发布计数)."""
        with self.lock:
            return self.annotated, self.detections, self.count


#-----------------------------------------------------------
# 后台视频保存 (结果写入 ./camera_front/result, 不占用推理时间)
#-----------------------------------------------------------
class VideoSaver:
    """后台保存线程: 把标注帧写入 result 目录下的 mp4 视频.

    worker 只把标注帧丢进有界队列 (满丢最旧, 复用 dispatch), 保存线程在
    独立线程中逐帧写盘, 不拖累采集/推理.
    输出帧率由 SAVE_FPS 决定: 实际处理帧率高于它则视频慢动作回放 (便于
    观察), 低于它则快进.
    收尾顺序: 主线程先 stop_evt 停 worker, 再 join 本线程排空队列.
    """

    def __init__(self, out_dir, width, height, fps=60, filename=None):
        os.makedirs(out_dir, exist_ok=True)      # 目录不存在则自动创建
        if filename is None:
            filename = f'front_detect_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        self.path = os.path.join(out_dir, filename)
        self.q = queue.Queue(maxsize=128)        # 有界队列, 满丢最旧, 防写盘不及积压
        self.writer = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not self.writer.isOpened():
            self.writer = None                   # 编码器不可用则禁用保存
            print(f'[警告] 无法创建输出视频: {self.path} (保存已禁用)')
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.written = 0       # 已写盘帧数
        self.t_start = None    # 第一帧写入时刻 (计时真实写盘速率)

    def _run(self):
        """写盘循环 (独立线程): 收到 None 哨兵即退出."""
        while True:
            frame = self.q.get()
            if frame is None:
                break
            if self.t_start is None:
                self.t_start = time.perf_counter()
            self.writer.write(frame)
            self.written += 1
            self.q.task_done()

    def put(self, frame):
        """把一帧标注图排入保存队列, 立即返回 (满则丢最旧, 不阻塞 worker)."""
        if self.writer is not None:
            dispatch(self.q, frame)

    def join(self):
        """等队列全部写完再退出保存线程 (主线程收尾时调用).

        结束时报真实写盘帧率: 与 SAVE_FPS 比较可判断视频是慢放还是快进.
        """
        if self.writer is None:
            return
        self.q.join()                            # 排空已入队帧
        self.q.put(None)                         # 哨兵退出写盘线程
        self.thread.join()
        self.writer.release()
        real_fps = (self.written / (time.perf_counter() - self.t_start)
                    if self.written and self.t_start else 0.0)
        print(f'[保存] 检测视频已写入: {self.path}')
        print(f'[保存] 共 {self.written} 帧, 实际写盘 {real_fps:.1f} fps '
              f'(视频播放帧率 {SAVE_FPS}: 写盘 < 播放 → 慢动作, > → 快进)')


#-----------------------------------------------------------
# 统计日志 (采集/处理/已处理 写入 result, 控制台留给每帧目标位置)
#-----------------------------------------------------------
class LogWriter:
    """把采集/处理等统计写入 result 目录下的 log 文件.

    控制台只保留"每帧目标中心 + 相对 POINT 偏移"的实时输出;
    周期性统计 (采集/处理/已处理/队列) 与结束汇总全部落盘, 便于事后回放.
    """

    def __init__(self, out_dir, filename=None):
        os.makedirs(out_dir, exist_ok=True)      # 目录不存在则自动创建
        if filename is None:
            filename = f'front_log_{time.strftime("%Y%m%d_%H%M%S")}.log'
        self.path = os.path.join(out_dir, filename)
        self.lock = threading.Lock()

    def write(self, msg):
        """追加一行带时间戳的日志 (线程安全)."""
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        with self.lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


#-----------------------------------------------------------
# 检测 worker (机会式小批量)
#-----------------------------------------------------------
def make_worker(detector, dispatch_q, latest, stats, stop_evt, enable_preprocess, batch, saver=None):
    """构造一个检测 worker 的线程函数 (闭包绑定各自模型).

    机会式小批量: 取 1 帧立即处理 (保持低延迟), 推理前顺手把队列里已有的帧
    最多带走 batch-1 个, 一次前向吃掉一批 (ultralytics 后端吃满 GPU).
    负载轻 = 单帧低延迟; 负载重 = 自动凑批吃满算力.
    每处理完一帧就把结果 publish 到 latest, 供主循环节流显示 (imshow 非线程安全);
    同时把标注帧丢给后台 saver (若有) 写盘保存.
    退出: get 带超时感知 stop_evt, 不依赖哨兵 (哨兵在队列满时会被卡死).
    """
    def work():
        while not stop_evt.is_set():
            try:
                f = dispatch_q.get(timeout=GET_TIMEOUT)
            except queue.Empty:
                continue        # 队列暂时空: 回去检查 stop_evt, 支持被强制停止
            frames = [f]
            for _ in range(batch - 1):
                try:
                    frames.append(dispatch_q.get_nowait())
                except queue.Empty:
                    break       # 队列空了, 就只处理已拿到的
            try:
                results = detector.detect(frames, enable_preprocess, PREPROCESS_CFG)
                for annotated, dets in results:
                    stats.add(len(dets))
                    latest.publish(annotated, dets)
                    if saver is not None:        # 后台保存 (可选)
                        saver.put(annotated)
            except Exception as exc:
                print(f'[worker] 检测异常: {exc!r}')
    return work


def format_det_offset(det, point):
    """格式化单个检测: 中心点坐标 + 相对预设点 point 的偏移 (右/上为正)."""
    cx, cy = det['center']
    px, py = point
    dx, dy = cx - px, py - cy
    return (f"  {det['label']} conf={det['score']:.3f} center=({cx},{cy}) "
            f"相对POINT({int(px)},{int(py)}) dx={dx:+.1f}, dy={dy:+.1f} (右/上为正)")


#====================
#主程序
#====================
def parse_args():
    """解析命令行参数; 未指定的参数使用 参数配置 中的默认值."""
    parser = argparse.ArgumentParser(
        description='前视摄像头 200fps 采集 + YOLO 检测 (复用 v4.0/YOLO + v4.0/pre_process)')
    parser.add_argument('--device', type=int, default=CAMERA_INDEX,
                        help='相机索引, 默认%d (取 参数配置 CAMERA_INDEX)' % CAMERA_INDEX)
    parser.add_argument('--backend', type=str, default=BACKEND,
                        choices=['hbm', 'ultralytics'],
                        help='检测后端: hbm(RDK板端BPU) / ultralytics(开发机)')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='模型文件名 (相对 v4.0/YOLO; ultralytics 不传时自动用 best.pt)')
    parser.add_argument('--label-file', type=str, default=LABEL_FILE,
                        help='类别名称文件 (每行一个类名)')
    parser.add_argument('--workers', type=int, default=None,
                        help='检测 worker 线程数 (默认取 参数配置 WORKERS=%d)' % WORKERS)
    parser.add_argument('--batch', type=int, default=None,
                        help='每 worker 机会式批处理最大帧数 (默认取 参数配置 BATCH_SIZE=%d)' % BATCH_SIZE)
    parser.add_argument('--no-preprocess', action='store_true', default=None,
                        help='关闭水下预处理 (提升吞吐)')
    parser.add_argument('--no-show', action='store_true', default=None,
                        help='无窗口模式 (只打印 fps/检测概况)')
    parser.add_argument('--no-save', action='store_true', default=None,
                        help=f'不保存检测视频 (默认保存到 ./camera_front/{SAVE_DIR})')
    parser.add_argument('--no-log', action='store_true', default=None,
                        help=f'不写采集/处理统计 log (默认写入 ./camera_front/{SAVE_DIR})')
    return parser.parse_args()


def main():
    """前视摄像头 200fps 采集 + YOLO 检测主流程.

    线程模型:
      采集线程   : 独占 cap.read(), 每帧推入有界队列 (稳定 ~204fps)
      N个worker  : 各自独立模型实例, 机会式小批量检测, 发布最新标注结果
      主线程     : 按固定节拍显示最新标注帧 + 每秒打印 采集/处理 fps
    """
    opt = parse_args()

    # ---- 1. 加载共享模块: 预处理 + YOLO 检测器 ----
    preprocess_fn = load_preprocess()                            # v4.0/pre_process
    # 类别名文件 (可选): 仅显式提供时才解析路径, 否则为 None 跳过
    label_path = resolve_model_path(opt.label_file) if opt.label_file else None
    labels = load_labels(label_path)
    model_path = resolve_model_path(opt.model)
    # ultralytics 后端不传 --model 时, 自动切换到开发机 .pt 模型
    if opt.backend == 'ultralytics' and opt.model == MODEL_PATH:
        model_path = resolve_model_path('best.pt')

    # worker/批大小: 默认取 参数配置, 命令行可覆盖
    # (开发机 ultralytics 建议 8x4 才能追平 200fps 采集; RDK hbm 建议 1x1)
    workers = opt.workers or WORKERS
    batch = opt.batch or BATCH_SIZE
    enable_preprocess = ENABLE_PREPROCESS and not bool(opt.no_preprocess)
    show = SHOW and not bool(opt.no_show)
    save_video = SAVE_VIDEO and not bool(opt.no_save)
    log_enabled = ENABLE_LOG and not bool(opt.no_log)

    # ---- 2. 打开相机 (索引由 配置 CAMERA_INDEX / 命令行 --device 指定) ----
    cap = open_camera(opt.device, IMAGE_WIDTH, IMAGE_HEIGHT)
    if cap is None:
        print(f'[错误] 无法打开 camera {opt.device} (MJPG {IMAGE_WIDTH}x{IMAGE_HEIGHT}), 请检查设备')
        return 1
    nw, nh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfps = cap.get(cv2.CAP_PROP_FPS)
    nfourcc = fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    print('== camera %d 请求 %s %dx%d -> 协商 %dx%d @%.0ffps (fourcc=%s) =='
          % (opt.device, FORMAT, IMAGE_WIDTH, IMAGE_HEIGHT, nw, nh, nfps, nfourcc))

    # ---- 3. 后台输出: 检测视频 + 统计 log (均写入 ./camera_front/result) ----
    saver = None
    if save_video:
        saver = VideoSaver(_SAVE_DIR, nw, nh, SAVE_FPS)
        print(f'[*] 检测结果将保存到: {saver.path}')
    logger = None
    if log_enabled:
        logger = LogWriter(_SAVE_DIR)
        print(f'[*] 采集/处理统计将写入: {logger.path}')

    # ---- 4. 加载检测器 (每个 worker 独立实例, 避免多线程共享同一模型的状态竞态) ----
    detectors = [Detector(opt.backend, model_path, labels, preprocess_fn) for _ in range(workers)]
    print(f'[*] 预热模型 ({opt.backend}) x{workers} ...')
    for det in detectors:
        det.warmup()
    print(f'[*] 后端={opt.backend} worker x{workers} 批{batch} | '
          f'预处理: {"开" if enable_preprocess else "关"} | 显示: {"开" if show else "关"}'
          f' | 保存: {"开" if save_video else "关"}')

    # ---- 5. 分发队列 + 采集线程 (每读到一帧直接入队) ----
    stop_evt = threading.Event()
    dispatch_q = queue.Queue(maxsize=workers * batch * 2)   # 满则丢最旧, 容量供 worker 补批
    g = Grabber(cap, dispatch_q, stop_evt)
    threading.Thread(target=g.run, daemon=True).start()

    # ---- 6. 启动 N 个检测 worker (标注帧同时进 latest 显示 + saver 保存) ----
    stats = Stats()
    latest = LatestResult()
    worker_threads = []
    for det in detectors:
        t = threading.Thread(
            target=make_worker(det, dispatch_q, latest, stats, stop_evt,
                               enable_preprocess, batch, saver),
            daemon=True)
        t.start()
        worker_threads.append(t)

    # ---- 7. 主循环: 让出 GIL + 逐帧输出目标位置 + 节流显示 + 统计写 log ----
    t0 = time.perf_counter()
    t_last = t0
    last_cap = 0
    last_show = 0.0
    last_pub = 0          # 上一帧已输出到控制台的发布序号 (LatestResult.count)
    print('[*] 开始运行 (Esc 或 Ctrl+C 退出); 每帧目标位置输出到控制台'
          + (f', 统计写入 {logger.path}' if logger is not None else '') + ' ...')

    try:
        while not stop_evt.is_set():
            time.sleep(0.001)          # 关键: 让出 GIL, 否则死循环饿死采集线程 (~35fps)
            with g.lock:
                cur_cap = g.count

            # 逐帧: 每发布一帧, 控制台输出该帧目标中心点 + 相对预设 POINT 的偏移
            annotated, dets, pub_count = latest.latest()
            if pub_count != last_pub:
                last_pub = pub_count
                if dets:                               # 无目标帧不刷屏
                    print(f'[{time.perf_counter() - t0:7.3f}s] 第{pub_count}帧')
                    for d in dets:
                        print(format_det_offset(d, POINT))

            # 节流显示: 主循环按固定时间节拍显示"最新标注帧" (~60fps 跟手)
            if show:
                now = time.perf_counter()
                if now - last_show >= SHOW_INTERVAL:
                    if annotated is not None:
                        cv2.imshow(WIN_NAME, annotated)
                        if cv2.waitKey(1) & 0xFF == 27:   # Esc 提前退出
                            stop_evt.set()
                    last_show = now

            # 统计写入 log 文件 (采集/处理/已处理, 不再打印到控制台)
            now = time.perf_counter()
            if now - t_last >= LOG_INTERVAL:
                processed, det_frames = stats.snapshot()
                cap_fps = (cur_cap - last_cap) / (now - t_last)
                proc_fps = processed / (now - t0)
                if logger is not None:
                    logger.write('采集 %6.1f fps | 处理 %6.1f fps | 已处理 %d | 检出目标 %d帧 | 队列 %d'
                                 % (cap_fps, proc_fps, processed, det_frames, dispatch_q.qsize()))
                last_cap = cur_cap
                t_last = now
    except KeyboardInterrupt:
        print('\n[*] 收到 Ctrl+C, 退出 ...')

    # ---- 8. 收尾: stop_evt 通知 worker 自行退出, 排空保存队列, 释放资源 ----
    # 不用哨兵: 若队列在收尾时是满的, put(None) 会永久阻塞; worker 用超时 get 感知退出
    stop_evt.set()
    for t in worker_threads:
        t.join(timeout=5)
    if saver is not None:
        saver.join()         # 等已入队标注帧全部写盘后再退出
    cap.release()
    if show:
        cv2.destroyAllWindows()

    # 结束汇总写入 log 文件 (控制台只保留运行结束提示)
    elapsed = time.perf_counter() - t0
    with g.lock:
        final_cap = g.count
    processed, det_frames = stats.snapshot()
    summary = ['== 汇总 (%.2fs) ==' % elapsed,
               '  采集帧率   : %.1f fps (累计 %d 帧)' % (final_cap / elapsed if elapsed else 0, final_cap),
               '  处理帧率   : %.1f fps (共处理 %d 帧, %d worker x 批%d)'
               % (processed / elapsed if elapsed else 0, processed, workers, batch),
               '  检出目标帧 : %d / %d' % (det_frames, processed)]
    if saver is not None:
        summary.append('  检测视频   : %s' % saver.path)
    if logger is not None:
        for line in summary:
            logger.write(line)
        print('[*] 运行结束, 采集/处理统计与汇总已写入: %s' % logger.path)
    else:
        print('\n'.join(summary))
    return 0


if __name__ == '__main__':
    main()
