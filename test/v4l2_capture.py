"""
v4l2_capture.py - 用 v4l2-ctl 直接采图, MJPG 拆成 jpg, 打印实测帧率

在板子上跑, 无需 OpenCV (需要 v4l-utils):
  python3 v4l2_capture.py                        # 默认 MJPG 640x480 采 100 帧
  python3 v4l2_capture.py --count 300 --skip 10
  python3 v4l2_capture.py --width 320 --height 240 --fps 400

采完的 jpg 在 --out 目录里, 拉回电脑看。
"""

import argparse
import os
import re
import subprocess
import time


def run(cmd):
    """执行 shell 命令, 返回 (退出码, 输出)."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr
    except OSError:
        return -1, ''


def parse_fps(text):
    """取 v4l2-ctl 输出的实测帧率 (排除 'Frame rate set to' 请求值回显)."""
    text = re.sub(r'Frame rate set to\s+[\d.]+ fps', '', text)
    m = re.findall(r'([\d.]+) fps', text)
    return float(m[-1]) if m else None


def split_jpegs(data, out_dir):
    """按 JPEG 起止标记 FFD8..FFD9 把连续 MJPG 流拆成 jpg 文件."""
    files, start, idx = [], -1, 0
    i, n = 0, len(data)
    while i < n - 1:
        if start < 0 and data[i] == 0xFF and data[i + 1] == 0xD8:
            start = i                              # 帧起始 (SOI)
        elif start >= 0 and data[i] == 0xFF and data[i + 1] == 0xD9:
            path = os.path.join(out_dir, 'frame_%04d.jpg' % idx)   # 帧结束 (EOI)
            with open(path, 'wb') as f:
                f.write(data[start:i + 2])
            files.append(path)
            idx += 1
            start = -1
        i += 1
    return files


def main():
    ap = argparse.ArgumentParser(description='v4l2-ctl 直接采图')
    ap.add_argument('--device', default='/dev/video0', help='摄像头设备')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--format', default='MJPG', help='像素格式 (默认MJPG)')
    ap.add_argument('--count', type=int, default=100, help='采图帧数')
    ap.add_argument('--skip', type=int, default=10, help='跳过起始帧再采, 排除初始突发')
    ap.add_argument('--fps', type=int, default=400, help='请求帧率, 0=不设置')
    ap.add_argument('--out', default='captures', help='保存目录')
    a = ap.parse_args()

    dev = a.device if not a.device.isdigit() else '/dev/video%d' % int(a.device)
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, 'stream.bin')
    if os.path.exists(raw):
        os.remove(raw)                            # 清残留, 避免采流失败读到旧数据

    # 一条命令: 设格式(可选设帧率) + 流式采帧落盘
    cmd = 'v4l2-ctl --device %s --set-fmt-video=width=%d,height=%d,pixelformat=%s' \
          % (dev, a.width, a.height, a.format)
    if a.fps > 0:
        cmd += ' --set-parm=%d' % a.fps
    cmd += ' --stream-mmap --stream-skip=%d --stream-count=%d --stream-to=%s' \
           % (a.skip, a.count, raw)

    print('采图: %s %dx%d x%d 帧 -> %s' % (a.format, a.width, a.height, a.count, a.out))
    t0 = time.perf_counter()
    code, out = run(cmd)
    elapsed = time.perf_counter() - t0
    print(out.strip())

    if code != 0 or not os.path.exists(raw) or os.path.getsize(raw) == 0:
        print('[失败] v4l2-ctl 未采到数据 (退出码 %d)' % code)
        return 1

    if a.format == 'MJPG':
        with open(raw, 'rb') as f:
            data = f.read()
        files = split_jpegs(data, a.out)
        print('已拆出 %d 帧 -> %s/' % (len(files), a.out))
    else:
        print('原始流已保存: %s (%d bytes)' % (raw, os.path.getsize(raw)))

    fps = parse_fps(out)
    if fps is None and elapsed > 0.2 and a.count >= 50:   # 兜底按墙钟估算
        fps = a.count / elapsed
    print('实测帧率: %s' % ('%.1f fps' % fps if fps else '未知'))
    return 0


if __name__ == '__main__':
    main()
