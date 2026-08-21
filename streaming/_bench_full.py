# 复刻真实运行的三线程结构, 逐项打开以定位 72fps 瓶颈
import sys, time, threading
import cv2

sys.path.insert(0, r"d:\RC\S100\视觉部分\streaming")
import frame_protocol as fp
from dev_capture_send import _LatestFrameStore, _Stats, _encode

VIDEO = r"C:\Users\Zhao-Xinqi\AppData\Local\Temp\test_frames.mp4"

rx = fp.FrameReceiver(proto="tcp", host="127.0.0.1", port=8935).start()
tx = fp.FrameSender(proto="tcp", host="127.0.0.1", port=8935)
tx.connect(timeout=2)
deadline = time.time() + 3
while not rx.stats()["connected"] and time.time() < deadline:
    time.sleep(0.01)

store = _LatestFrameStore()
stats = _Stats()
stop = threading.Event()

def cap_worker():
    cap = cv2.VideoCapture(VIDEO)
    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.002)
            continue
        store.put(frame)
        with stats.lock:
            stats.cap_count += 1
    cap.release()

def send_worker():
    last_gen = 0
    while not stop.is_set():
        gen, frame = store.wait_new(last_gen, timeout=0.2)
        if frame is None:
            continue
        with stats.lock:
            stats.drop_count += max(gen - last_gen - 1, 0)
        last_gen = gen
        payload, (w, h) = _encode(frame, 1.0, 70)
        if payload is None:
            continue
        if tx.send(gen, w, h, payload):
            with stats.lock:
                stats.send_count += 1
                stats.send_bytes += len(payload)

t_cap = threading.Thread(target=cap_worker, daemon=True)
t_send = threading.Thread(target=send_worker, daemon=True)
t_cap.start(); t_send.start()

# 主线程: 模拟真实 main (sleep 0.001 + 每秒统计打印)
t0 = time.monotonic()
last_stats = t0; last_cap = last_send = 0
while time.monotonic() - t0 < 3.0:
    time.sleep(0.001)
    now = time.monotonic()
    if now - last_stats >= 1.0:
        with stats.lock:
            cap_c, send_c = stats.cap_count, stats.send_count
        print("窗口: 采集 %5.1f fps | 发送 %5.1f fps" % (
            (cap_c - last_cap) / (now - last_stats), (send_c - last_send) / (now - last_stats)))
        last_cap, last_send, last_stats = cap_c, send_c, now
stop.set()
t_cap.join(timeout=2); t_send.join(timeout=2)
rx.stop(); tx.close()
