"""
frame_protocol.py - 帧传输协议 (TCP / UDP)

职责:
  定义"开发机采集发送 -> S100 接收处理"的图像帧线协议, 提供收发两端复用工具.
  负载为 JPEG 编码 (发送端 cv2.imencode 生成, 接收端 cv2.imdecode 还原 BGR 帧).

支持两种传输方式 (命令行 --proto 选择):
  - tcp : 长度前缀帧. 可靠有序, 适合有线/千兆链路;
          网络饱和时压力反馈到发送端 (sendall 阻塞), 由发送端主动丢最旧帧.
  - udp : 分片数据报. 每帧按 UDP_CHUNK 拆成多个 MTU 内数据报, 接收端按 frame_id
          重组; 适合带宽紧张或可容忍少量丢帧的无线链路 (丢包只丢帧, 不重传).

============================================================================
线格式 (TCP):  [固定 18 字节包头][payload_len 字节 JPEG 数据]

  包头 (小端):
    magic      4B  b'RCV1'          魔数, 校验链路对齐
    version    1B  协议版本 (=1)
    codec      1B  负载编码 (1=JPEG)
    width      2B  图像宽度
    height     2B  图像高度
    frame_id   4B  帧序号 (发送端读帧计数, 单调递增)
    payload    4B  负载字节数

============================================================================
线格式 (UDP 分片):  每帧拆成 N 个数据报, 每个数据报:

  [固定 22 字节分片头][≤UDP_CHUNK 字节数据]

  分片头 (小端):
    magic      4B  b'RCV1'
    version    1B  协议版本 (=1)
    codec      1B  负载编码 (1=JPEG)
    width      2B  图像宽度
    height     2B  图像高度
    frame_id   4B  帧序号
    total_len  4B  整帧负载字节数
    frag_idx   2B  分片序号 (0 起)
    frag_cnt   2B  该帧分片总数

  接收端按 frame_id 重组; 超过 FRAME_TIMEOUT 未收全的残帧定时丢弃.
============================================================================
"""

import socket
import struct
import threading
import time

# ================================================================
# 常量
# ================================================================
MAGIC = b'RCV1'                    # 协议魔数
VERSION = 1                        # 协议版本
CODEC_JPEG = 1                     # 负载编码: JPEG

FRAME_MAX_PAYLOAD = 4 * 1024 * 1024  # 单帧负载上限 4MB (640x480 JPEG 远小于此)
UDP_CHUNK = 1200                   # UDP 每数据报最大负载字节 (22+1200 < MTU 1500)
FRAME_TIMEOUT = 0.5                # UDP 残帧重组超时 (秒)

# TCP 头: magic(4)+ver(1)+codec(1)+w(2)+h(2)+frame_id(4)+payload(4) = 18B
_TCP_HEADER = struct.Struct('<4sBBHHII')
# UDP 分片头: TCP 头 + total(4)+frag_idx(2)+frag_cnt(2) = 22B
_UDP_HEADER = struct.Struct('<4sBBHHIIHH')

FRAME_HEADER_SIZE = _TCP_HEADER.size   # 18
UDP_HEADER_SIZE = _UDP_HEADER.size     # 22


# ================================================================
# TCP: 打包 / 解析 / 收发
# ================================================================

def pack_tcp_frame(frame_id: int, width: int, height: int, payload: bytes,
                   codec: int = CODEC_JPEG) -> bytes:
    """把一帧打包为 TCP 线格式 (头 + 负载)."""
    header = _TCP_HEADER.pack(MAGIC, VERSION, codec, width, height, frame_id, len(payload))
    return header + payload


def parse_tcp_frame(buf: bytes) -> dict | None:
    """解析 TCP 包头; 魔数/负载长度非法时返回 None (链路不同步, 交由上层重置连接)."""
    if len(buf) < FRAME_HEADER_SIZE:
        return None
    magic, version, codec, width, height, frame_id, payload_len = _TCP_HEADER.unpack(buf)
    if magic != MAGIC:
        return None
    if not (0 < payload_len <= FRAME_MAX_PAYLOAD):
        return None
    return {'version': version, 'codec': codec, 'width': width,
            'height': height, 'frame_id': frame_id, 'payload_len': payload_len}


def recv_exact(sock, n: int) -> bytes | None:
    """从 socket 精确读取 n 字节; 连接断开/对端关闭时返回 None."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:      # 非阻塞检查: 继续等
            continue
        except OSError:
            return None
        if not chunk:               # 对端关闭
            return None
        buf.extend(chunk)
    return bytes(buf)


# ================================================================
# UDP: 分片 / 重组
# ================================================================

def make_udp_frames(frame_id: int, width: int, height: int, payload: bytes,
                    codec: int = CODEC_JPEG) -> list[bytes]:
    """把一帧拆成 UDP 分片列表 (每个元素为一个可发送的数据报)."""
    total = len(payload)
    frag_cnt = (total + UDP_CHUNK - 1) // UDP_CHUNK
    datagrams = []
    for idx in range(frag_cnt):
        chunk = payload[idx * UDP_CHUNK: (idx + 1) * UDP_CHUNK]
        header = _UDP_HEADER.pack(MAGIC, VERSION, codec, width, height,
                                  frame_id, total, idx, frag_cnt)
        datagrams.append(header + chunk)
    return datagrams


class UdpReassembler:
    """UDP 分片重组器: 按 frame_id 收齐分片后输出完整负载.

    说明:
      - 乱序分片按 (frame_id, frag_idx) 落位, 无需排序
      - 超过 FRAME_TIMEOUT 未收全的残帧定时清理丢弃
    """

    def __init__(self, timeout: float = FRAME_TIMEOUT):
        self._timeout = timeout
        self._partial = {}      # frame_id -> {'buf','got','frag_cnt','last'}
        self._lock = threading.Lock()

    def feed(self, datagram: bytes):
        """喂入一个数据报; 收齐整帧时返回 (frame_id, width, height, payload, codec),
        否则返回 None."""
        if len(datagram) < UDP_HEADER_SIZE:
            return None
        magic, version, codec, width, height, frame_id, total, frag_idx, frag_cnt = \
            _UDP_HEADER.unpack(datagram[:UDP_HEADER_SIZE])
        if magic != MAGIC:
            return None
        if not (0 < total <= FRAME_MAX_PAYLOAD) or frag_idx >= frag_cnt:
            return None

        data = datagram[UDP_HEADER_SIZE:]
        now = time.monotonic()
        with self._lock:
            cur = self._partial.get(frame_id)
            if cur is None:
                cur = {'buf': bytearray(total), 'got': 0, 'frag_cnt': frag_cnt, 'last': now}
                self._partial[frame_id] = cur
            cur['last'] = now

            # 分片写入 (越界保护, 损坏包不越界)
            start = frag_idx * UDP_CHUNK
            if start < total:
                end = min(start + len(data), total)
                cur['buf'][start:end] = data[:end - start]
                cur['got'] += 1

            if cur['got'] >= cur['frag_cnt']:
                payload = bytes(cur['buf'])
                del self._partial[frame_id]
                return (frame_id, width, height, payload, codec)
        return None

    def cleanup(self):
        """丢弃超过重组超时的残帧 (在接收循环空闲时调用)."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, v in self._partial.items() if now - v['last'] > self._timeout]
            for k in stale:
                del self._partial[k]


# ================================================================
# Frame 命名元组
# ================================================================
from collections import namedtuple
Frame = namedtuple('Frame', ['frame_id', 'width', 'height', 'payload', 'codec'])


# ================================================================
# 帧发送器 (开发机端)
# ================================================================

class FrameSender:
    """帧发送器 (TCP 客户端 / UDP 对端).

    - TCP: connect 建立连接, 断开时 send 返回 False, 由调用方决定何时重连;
    - UDP: 无连接, 直接 sendto 分片数据报.
    """

    def __init__(self, proto: str = 'tcp', host: str = '127.0.0.1', port: int = 8900):
        if proto not in ('tcp', 'udp'):
            raise ValueError(f"不支持的传输协议: {proto} (仅支持 tcp/udp)")
        self._proto = proto
        self._addr = (host, port)
        self._sock = None

    def connect(self, timeout: float = 5.0) -> bool:
        """建立连接 (TCP); UDP 直接创建 socket 即可用. 超时未连上返回 False."""
        if self._proto == 'udp':
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 * 1024 * 1024)
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(self._addr)
        except OSError:
            sock.close()
            return False
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 * 1024 * 1024)
        self._sock = sock
        return True

    def send(self, frame_id: int, width: int, height: int, payload: bytes) -> bool:
        """发送一帧; 成功返回 True, 失败 (未连接/网络异常) 返回 False (调用方丢帧)."""
        if self._sock is None:
            return False
        try:
            if self._proto == 'tcp':
                self._sock.sendall(pack_tcp_frame(frame_id, width, height, payload))
            else:
                for datagram in make_udp_frames(frame_id, width, height, payload):
                    self._sock.sendto(datagram, self._addr)
            return True
        except OSError:
            # 连接异常: 关闭并置空, 下次 send 由调用方触发重连
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
            return False

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ================================================================
# 帧接收器 (S100 / 测试接收端)
# ================================================================

class FrameReceiver:
    """帧接收器 (TCP 服务端 / UDP 绑定端).

    用法:
        rx = FrameReceiver(proto='tcp', port=8900).start()
        while running:
            frame = rx.latest()      # 取最新完整帧 (旧帧自动丢弃)
            if frame is not None:
                ...处理 frame.payload...
        rx.stop()

    设计:
      - 后台线程持续 drain socket (TCP 读字节流 / UDP 收数据报并重组),
        只保留"最新完整帧", 避免链路积压造成延迟累积;
      - 处理端按自己节奏取最新帧, 处理不过来时"尚未被消费"的旧帧在接收端被覆盖
        丢弃 (rx.stats()['drop'] 统计真正丢失的帧; 已消费的帧不计数).
    """

    def __init__(self, proto: str = 'tcp', host: str = '0.0.0.0', port: int = 8900):
        if proto not in ('tcp', 'udp'):
            raise ValueError(f"不支持的传输协议: {proto} (仅支持 tcp/udp)")
        self.proto = proto
        self.host = host
        self.port = port
        self._sock = None
        self._conn = None            # TCP 当前客户端连接
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)  # 新帧到达时通知处理端
        self._slot = None            # 最新完整帧 (Frame)
        self._slot_read = True       # 当前槽位是否已被消费 (read 后置 True)
        self._gen = 0                # 帧代数 (每次发布 +1, 供 wait 判断新帧)
        self._recv_count = 0         # 累计收到完整帧数
        self._drop_count = 0         # 被覆盖且尚未被消费的帧数 (真正丢失的帧)
        self._stop = threading.Event()
        self._thread = None
        self._reassembler = UdpReassembler() if proto == 'udp' else None

    # ---- 生命周期 ----

    def start(self) -> 'FrameReceiver':
        """绑定/监听并启动后台接收线程."""
        if self.proto == 'tcp':
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self._sock.bind((self.host, self.port))
            self._sock.listen(1)
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._recv_loop,
                                        name='frame-recv', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """停止接收线程并释放 socket."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        for s in (self._sock, self._conn):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    # ---- 对外接口 ----

    def latest(self) -> Frame | None:
        """取最新完整帧 (非阻塞), 标记已消费; 尚无帧时返回 None."""
        with self._lock:
            self._slot_read = True
            return self._slot

    def wait(self, timeout: float | None = None) -> Frame | None:
        """阻塞等待新帧到达, 返回最新完整帧; 超时返回 None.

        处理端应使用本方法取帧: 只在真正出现新帧时返回, 不会重复处理同一帧,
        且不依赖 time.sleep 轮询 (Windows 上 time.sleep 粒度约 15.6ms,
        轮询会把处理吞吐压到 ~60fps).

        Args:
            timeout: 最长等待秒数; None=无限等待.
        """
        with self._cond:
            gen = self._gen
            while self._gen == gen:
                if not self._cond.wait(timeout):
                    return None
            self._slot_read = True   # 已消费: 被新帧覆盖时不再计入丢弃
            return self._slot

    def stats(self) -> dict:
        """返回统计: 收到帧数 / 真正丢弃帧数 / TCP 客户端是否已接入.

        丢弃计数语义: 仅统计"在消费前被新帧覆盖"的帧 (处理不过来时的真实丢失);
        已消费 (处理中/已处理) 的帧即使被覆盖也不计数.
        """
        with self._lock:
            return {'recv': self._recv_count, 'drop': self._drop_count,
                    'connected': self._conn is not None}

    # ---- 后台接收 ----

    def _recv_loop(self):
        if self.proto == 'udp':
            self._udp_loop()
        else:
            self._tcp_loop()

    def _tcp_loop(self):
        while not self._stop.is_set():
            # 无客户端时等待接入 (accept 带 0.2s 超时, 保证能响应退出)
            if not self._connected():
                self._accept_one()
                continue
            frame = self._read_tcp_frame()
            if frame is None:               # 对端断开 / 协议不同步 -> 重置连接
                with self._lock:
                    if self._conn is not None:
                        try:
                            self._conn.close()
                        except OSError:
                            pass
                        self._conn = None
                print('[协议] 开发机连接已断开, 等待重新接入 ...')
                continue
            self._publish(frame)

    def _connected(self) -> bool:
        with self._lock:
            return self._conn is not None

    def _accept_one(self) -> bool:
        try:
            conn, addr = self._sock.accept()
        except socket.timeout:
            return False
        except OSError:
            return False
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        conn.settimeout(0.2)
        with self._lock:
            self._conn = conn
        print(f'[协议] 开发机已接入: {addr[0]}:{addr[1]}')
        return True

    def _read_tcp_frame(self) -> Frame | None:
        with self._lock:
            conn = self._conn
        if conn is None:
            return None
        header = recv_exact(conn, FRAME_HEADER_SIZE)
        if header is None:
            return None
        meta = parse_tcp_frame(header)
        if meta is None:                    # 链路不同步: 返回 None 触发重置连接
            return None
        payload = recv_exact(conn, meta['payload_len'])
        if payload is None:
            return None
        return Frame(meta['frame_id'], meta['width'], meta['height'],
                     payload, meta['codec'])

    def _udp_loop(self):
        while not self._stop.is_set():
            try:
                data, _addr = self._sock.recvfrom(UDP_HEADER_SIZE + UDP_CHUNK)
            except socket.timeout:
                if self._reassembler is not None:
                    self._reassembler.cleanup()   # 空闲时清理超时残帧
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue
            if self._reassembler is None:
                continue
            frame = self._reassembler.feed(data)
            if frame is not None:
                self._publish(Frame(*frame))

    def _publish(self, frame: Frame):
        with self._cond:
            # 仅当旧帧还没被消费 (读走) 时, 覆盖才算真正丢弃
            if self._slot is not None and not self._slot_read:
                self._drop_count += 1
            self._slot = frame
            self._slot_read = False
            self._recv_count += 1
            self._gen += 1
            self._cond.notify_all()   # 唤醒等待新帧的处理端
