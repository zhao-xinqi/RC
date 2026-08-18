"""
P3 穿门 - 合成图像验证脚本 (临时)

用程序生成合成图像验证 DoorDetector:
  1. 红色 Π 形门 (高门)  → 应 detected=True, door_type='high'
  2. 红色 Π 形门 (矮门)  → 应 detected=True, door_type='low'
  3. 红色闭合矩形 (收集框) → 应 detected=False
  4. 红色实心圆 (撞球)    → 应 detected=False
  5. 红色空心圆环        → 应 detected=False

运行: python _verify_test.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

# 加载 P3 模块
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODULE_DIR)

from door_detector import DoorDetector, get_config

IMG_W, IMG_H = 640, 480

# ================================================================
# 合成图像生成
# ================================================================

def _base_image():
    """蓝色水下背景"""
    bg = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    bg[:] = (180, 90, 40)   # BGR, 偏蓝绿模拟水下
    return bg


def _draw_pi_door(img, x, y, w, h, beam=30, pillar=30, color=(50, 60, 210)):
    """
    绘制 Π 形门: 顶部横梁 + 左右竖杆
    Args:
        x, y, w, h: 外接矩形
        beam:   横梁厚度
        pillar: 竖杆宽度
        color:  BGR 红色
    """
    # 顶部横梁
    cv2.rectangle(img, (x, y), (x + w, y + beam), color, -1)
    # 左竖杆
    cv2.rectangle(img, (x, y), (x + pillar, y + h), color, -1)
    # 右竖杆
    cv2.rectangle(img, (x + w - pillar, y), (x + w, y + h), color, -1)
    return img


def _draw_box(img, x, y, w, h, thickness=25, color=(50, 60, 210)):
    """绘制闭合矩形 (收集框): 四边"""
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img


def _draw_ball(img, cx, cy, r, color=(50, 60, 210)):
    """绘制实心圆 (撞球)"""
    cv2.circle(img, (cx, cy), r, color, -1)
    return img


def _draw_ring(img, cx, cy, r, thickness=25, color=(50, 60, 210)):
    """绘制空心圆环"""
    cv2.circle(img, (cx, cy), r, color, thickness)
    return img


# ================================================================
# 测试用例
# ================================================================

CASES = [
    ("高门 (Π形, 靠上)",      _draw_pi_door(_base_image(), 170, 40, 300, 260), 'high'),
    ("矮门 (Π形, 靠下)",      _draw_pi_door(_base_image(), 170, 270, 300, 210), 'low'),
    ("收集框 (闭合矩形)",      _draw_box(_base_image(), 220, 120, 200, 250),   None),
    ("撞球 (实心圆)",          _draw_ball(_base_image(), 320, 240, 130),       None),
    ("圆环 (空心圆)",          _draw_ring(_base_image(), 320, 240, 140),       None),
]


def main():
    detector = DoorDetector(get_config())

    print("=" * 62)
    print(f"  P3 穿门 合成图像验证  ({len(CASES)} 个用例)")
    print("=" * 62)

    all_pass = True
    for idx, (name, img, expect_type) in enumerate(CASES):
        result = detector.detect(img)

        detected = result['detected']
        door_type = result.get('door_type')

        if expect_type is not None:
            # 应该检测到, 且类型正确
            ok = detected and (door_type == expect_type)
            if ok:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
            print(f"  [{status}] {name:<18s}  "
                  f"detected={detected}  type={door_type}  "
                  f"(期望: detected=True, type={expect_type})")
        else:
            # 应该拒绝
            ok = not detected
            if ok:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
            print(f"  [{status}] {name:<18s}  "
                  f"detected={detected}  (期望: detected=False, 即被拒)")

        # 输出附加信息 (调试用)
        if detected:
            print(f"          offset={result['offset']:+.1f}px  "
                  f"center_norm={result['center_norm']}  aligned={result['aligned']}")

        # 保存可视化截图 (cv2.imwrite 不支持非 ASCII 文件名/路径, 用 ASCII 临时目录+索引名)
        vis = detector.draw_result(img, result)
        out_dir = os.path.join(tempfile.gettempdir(), "p3_verify_out")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"case_{idx}.png")
        ok = cv2.imwrite(fname, vis)
        if not ok:
            print(f"  [WARN] 截图保存失败: {fname}")

    print("=" * 62)
    print(f"  结果: {'全部通过 ✔' if all_pass else '存在失败 ✘'}")
    print(f"  可视化截图已保存至: {os.path.join(tempfile.gettempdir(), 'p3_verify_out')}")
    print("=" * 62)

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
