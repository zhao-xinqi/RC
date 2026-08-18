"""
P5 放球 - 合成图像验证脚本 (临时)

用程序生成合成图像验证 BoxDetector:
  1. 红色闭合矩形 (篮筐)   → 应 detected=True
  2. 红色 Π 形门 (无底杆)  → 应 detected=False (被底部判据排除)
  3. 红色实心圆 (撞球)     → 应 detected=False (被内部判据排除)
  4. 红色空心圆环         → 应 detected=False (被四边闭合判据排除)

运行: python _verify_test.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

# 加载 P5 模块
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODULE_DIR)

from box_detector import BoxDetector, get_config

IMG_W, IMG_H = 640, 480

# ================================================================
# 合成图像生成
# ================================================================

def _base_image():
    """蓝色水下背景"""
    bg = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    bg[:] = (180, 90, 40)   # BGR, 偏蓝绿模拟水下
    return bg


def _draw_box(img, x, y, w, h, thickness=25, color=(50, 60, 210)):
    """绘制闭合矩形 (篮筐/收集框): 四边"""
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img


def _draw_pi_door(img, x, y, w, h, beam=30, pillar=30, color=(50, 60, 210)):
    """绘制 Π 形门: 顶部横梁 + 左右竖杆 (无底杆)"""
    cv2.rectangle(img, (x, y), (x + w, y + beam), color, -1)   # 横梁
    cv2.rectangle(img, (x, y), (x + pillar, y + h), color, -1) # 左杆
    cv2.rectangle(img, (x + w - pillar, y), (x + w, y + h), color, -1)  # 右杆
    return img


def _draw_ball(img, cx, cy, r, color=(50, 60, 210)):
    """绘制实心圆 (撞球)"""
    cv2.circle(img, (cx, cy), r, color, -1)
    return img


def _draw_ring(img, cx, cy, r, thickness=25, color=(50, 60, 210)):
    """绘制空心圆环"""
    cv2.circle(img, (cx, cy), r, color, thickness)
    return img


def _draw_box_rotated(img, cx, cy, w, h, angle_deg, thickness=25,
                      color=(50, 60, 210)):
    """绘制旋转 angle_deg 度的闭合矩形 (测试朝向角)"""
    bg = img.copy()
    # 画布边长取对角线 + 余量, 确保旋转后的矩形不被裁剪
    diag = int(np.hypot(w, h)) + 80
    box = np.zeros((diag, diag, 3), dtype=np.uint8)
    box[:] = (0, 0, 0)
    x0, y0 = (diag - w) // 2, (diag - h) // 2
    cv2.rectangle(box, (x0, y0), (x0 + w, y0 + h), color, thickness)
    # 旋转并贴回原图
    m = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle_deg, 1.0)
    rot = cv2.warpAffine(box, m, (diag, diag))
    # 找旋转框非零区域, 居中放到 bg
    mask_rot = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(mask_rot)
    if len(xs) == 0:
        return bg
    min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()
    crop = rot[min_y:max_y + 1, min_x:max_x + 1]
    ch, cw = crop.shape[:2]
    px, py = int(cx - cw / 2), int(cy - ch / 2)
    bg[py:py + ch, px:px + cw] = crop
    return bg


# ================================================================
# 测试用例
# ================================================================

CASES = [
    ("篮筐 (闭合矩形)",  _draw_box(_base_image(), 220, 110, 200, 260), True),
    ("倾斜篮筐 (+25°)",  _draw_box_rotated(_base_image(), 320, 240, 200, 260, 25), True),
    ("横放篮筐",         _draw_box(_base_image(), 170, 160, 260, 160),   True),
    ("Π形门 (无底杆)",   _draw_pi_door(_base_image(), 170, 90, 300, 260), False),
    ("撞球 (实心圆)",    _draw_ball(_base_image(), 320, 240, 130),       False),
    ("圆环 (空心圆)",    _draw_ring(_base_image(), 320, 240, 140),       False),
]


def main():
    detector = BoxDetector(get_config())

    print("=" * 62)
    print(f"  P5 放球 合成图像验证  ({len(CASES)} 个用例)")
    print("=" * 62)

    all_pass = True
    for idx, (name, img, expect_det) in enumerate(CASES):
        result = detector.detect(img)

        detected = result['detected']

        if expect_det:
            ok = detected
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{status}] {name:<18s}  detected={detected}  "
                  f"(期望: detected=True)")
        else:
            ok = not detected
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{status}] {name:<18s}  detected={detected}  "
                  f"(期望: detected=False, 即被拒)")

        # 输出附加信息 (调试用)
        if detected:
            print(f"          offset={result['offset']:+.1f}px  "
                  f"angle={result['angle']:+.1f}°  aspect={result['aspect']:.2f}  "
                  f"aligned={result['aligned']}")

        # 保存可视化截图 (cv2.imwrite 不支持非 ASCII 文件名/路径, 用 ASCII 临时目录+索引名)
        vis = detector.draw_result(img, result)
        out_dir = os.path.join(tempfile.gettempdir(), "p5_verify_out")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"case_{idx}.png")
        ok = cv2.imwrite(fname, vis)
        if not ok:
            print(f"  [WARN] 截图保存失败: {fname}")

    print("=" * 62)
    print(f"  结果: {'全部通过 ✔' if all_pass else '存在失败 ✘'}")
    print(f"  可视化截图已保存至: {os.path.join(tempfile.gettempdir(), 'p5_verify_out')}")
    print("=" * 62)

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
