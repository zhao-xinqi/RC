# 临时协议自测脚本 (回环验证 TCP/UDP 收发与重组)
import time

import frame_protocol as fp

for proto in ("tcp", "udp"):
    rx = fp.FrameReceiver(proto=proto, host="127.0.0.1", port=8932).start()
    tx = fp.FrameSender(proto=proto, host="127.0.0.1", port=8932)
    assert tx.connect(timeout=2), proto + " connect failed"
    if proto == "tcp":
        deadline = time.time() + 3
        while not rx.stats()["connected"] and time.time() < deadline:
            time.sleep(0.02)
        assert rx.stats()["connected"], proto + " not connected"

    payload = bytes(range(256)) * 100          # 25.6KB JPEG 模拟负载
    assert tx.send(frame_id=1, width=640, height=480, payload=payload), proto + " send failed"
    got = None
    deadline = time.time() + 3
    while got is None and time.time() < deadline:
        got = rx.latest()
        time.sleep(0.01)
    assert got is not None, proto + " no frame received"
    assert got.frame_id == 1 and got.payload == payload, proto + " payload mismatch"
    n = len(got.payload)
    d = rx.stats()["drop"]
    print("%s FrameSender->FrameReceiver loopback OK, payload=%dB drop=%d" % (proto.upper(), n, d))
    tx.close()
    rx.stop()
print("ALL OK")
