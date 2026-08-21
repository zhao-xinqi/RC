# 三线程场景下逐阶段计时 (定位 85fps 瓶颈)
import sys, time, threading
import cv2

sys.path.insert(0, r"d:\RC\S100\视觉部分\streaming")
import frame_protocol as fp
from dev_capture_send import _LatestFrameStore, _Stats, _encode

VIDEO = r"C:\Users\Zhao-Xinqi\AppData\Local\Temp\test_frames.mp4"

rx = fp.FrameReceiver(proto="tcp", host="127.0.0.1", port=8936).start()
tx = fp.FrameSender(proto="tcp", host="127.0.0.1", port=8936)
tx.connect(timeout=2)
deadline = time.time() + 3
while not rx.stats()["connected"] and time.time() < deadline:
    time.sleep(0.01)

store = _LatestFrameStore()
stats = _Stats()
stop = threading.Event()
SLEEP_MS = {}   # 实际 sleep 时长统计

def cap_worker():
    cap = cv2.VideoCapture(VIDEO)
    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.002)
            continue
        store.put(frame)
    cap.release()

def send_worker():
    last_gen = 0
    acc_sleep = acc_take = acc_enc = acc_send = 0.0
    iters = 0
    while not stop.is_set():
        t = time.perf_counter(); time.sleep(0.0005); acc_sleep += time.perf_counter() - t
        t = time.perf_counter(); gen, frame = store.take_new(last_gen); acc_take += time.perf_counter() - t
        if frame is None:
            continue
        last_gen = gen
        t = time.perf_counter(); payload, (w, h) = _encode(frame, 1.0, 70); acc_enc += time.perf_counter() - t
        t = time.perf_counter(); tx.send(gen, w, h, payload); acc_send += time.perf_counter() - t
        iters += 1
        if iters >= 200:
            break
    print("前200次迭代均耗: sleep=%.3fms take=%.3fms encode=%.3fms send=%.3fms"
          % (acc_sleep / iters * 1000, acc_take / iters * 1000,
             acc_enc / iters * 1000, acc_send / iters * 1000))

t_cap = threading.Thread(target=cap_worker, daemon=True)
t_send = threading.Thread(target=send_worker, daemon=True)
t_cap.start(); t_send.start()

# 主线程 (与真实运行一致)
t0 = time.monotonic(); last_stats = t0
while time.monotonic() - t0 < 1.5:
    time.sleep(0.001)
    now = time.monotonic()
    if now - last_stats >= 1.0:
        last_stats = now
stop.set()
t_cap.join(timeout=2); t_send.join(timeout=2)
rx.stop(); tx.close()
