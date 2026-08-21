# TCP 回环吞吐基准: 找出发送端瓶颈
import time
import threading

import frame_protocol as fp

N = 500
PAYLOAD = b"\x00" * 10500     # 模拟 10.5KB JPEG

rx = fp.FrameReceiver(proto="tcp", host="127.0.0.1", port=8933).start()
tx = fp.FrameSender(proto="tcp", host="127.0.0.1", port=8933)
tx.connect(timeout=2)
deadline = time.time() + 3
while not rx.stats()["connected"] and time.time() < deadline:
    time.sleep(0.01)

# 纯发送速率 (不跑接收端处理循环)
t0 = time.perf_counter()
for i in range(N):
    assert tx.send(i, 640, 480, PAYLOAD)
t_send = time.perf_counter() - t0
print("send %d frames in %.3fs = %.0f fps (发送端 sendall 速率)" % (N, t_send, N / t_send))

# 接收端 drain 速率 (后台线程已在收; 统计收到的帧数)
time.sleep(0.5)
st = rx.stats()
print("receiver drained: recv=%d drop=%d (后台线程消费速率)" % (st["recv"], st["drop"]))
rx.stop()
tx.close()
