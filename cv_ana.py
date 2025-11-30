# cv_ana.py
import cv2
import numpy as np
from typing import Dict, Any


# ======================================================
# 0. 安全骨架化：没有 ximgproc 也能用
# ======================================================

def _simple_thinning(binary255: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen 细化算法简易实现。
    输入: 0/255 uint8 图像
    输出: 0/255 uint8 骨架图像
    """
    img = (binary255 > 0).astype(np.uint8)
    prev = np.zeros_like(img)
    rows, cols = img.shape

    while True:
        # 子迭代 1
        to_remove = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P = img[i, j]
                if P != 1:
                    continue
                n = [
                    img[i-1, j],   # p2
                    img[i-1, j+1], # p3
                    img[i,   j+1], # p4
                    img[i+1, j+1], # p5
                    img[i+1, j],   # p6
                    img[i+1, j-1], # p7
                    img[i,   j-1], # p8
                    img[i-1, j-1], # p9
                ]
                # 黑邻居数量
                C = sum(n)
                if C < 2 or C > 6:
                    continue
                # 0->1 变化次数
                transitions = sum((n[k] == 0 and n[(k+1) % 8] == 1) for k in range(8))
                if transitions != 1:
                    continue
                if n[0] * n[2] * n[4] != 0:
                    continue
                if n[2] * n[4] * n[6] != 0:
                    continue
                to_remove.append((i, j))
        for (i, j) in to_remove:
            img[i, j] = 0

        # 子迭代 2
        to_remove = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P = img[i, j]
                if P != 1:
                    continue
                n = [
                    img[i-1, j],   # p2
                    img[i-1, j+1], # p3
                    img[i,   j+1], # p4
                    img[i+1, j+1], # p5
                    img[i+1, j],   # p6
                    img[i+1, j-1], # p7
                    img[i,   j-1], # p8
                    img[i-1, j-1], # p9
                ]
                C = sum(n)
                if C < 2 or C > 6:
                    continue
                transitions = sum((n[k] == 0 and n[(k+1) % 8] == 1) for k in range(8))
                if transitions != 1:
                    continue
                if n[0] * n[2] * n[6] != 0:
                    continue
                if n[0] * n[4] * n[6] != 0:
                    continue
                to_remove.append((i, j))
        for (i, j) in to_remove:
            img[i, j] = 0

        if np.array_equal(img, prev):
            break
        prev = img.copy()

    return (img * 255).astype(np.uint8)


def thinning(binary255: np.ndarray) -> np.ndarray:
    """
    封装统一的骨架化接口：
    - 如果环境支持 ximgproc，就用它
    - 否则使用 _simple_thinning
    """
    # 尝试用 ximgproc，如果没有就走 fallback
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary255)
    else:
        return _simple_thinning(binary255)


# ======================================================
# 1. 结构特征提取（传入二值图像）
# ======================================================

def extract_cv_features(bw_img: np.ndarray) -> Dict[str, Any]:
    """
    输入:
        bw_img: 0/255 或 0~255 的 2D numpy 数组 (uint8)
    输出:
        dict 包含：
          - stroke_density
          - connected_components
          - contour_perimeter
          - contour_area
          - corner_points
          - skeleton_length
          - skeleton_branch_points
          - hu_moments (7维 numpy array)
    """
    if bw_img.ndim != 2:
        raise ValueError("extract_cv_features 需要 2D 灰度/二值图像")

    # 统一为 0/255
    if bw_img.max() <= 1:
        bw = (bw_img * 255).astype(np.uint8)
    else:
        bw = bw_img.astype(np.uint8)

    binary = (bw > 0).astype(np.uint8)

    h, w = binary.shape
    total_pixels = h * w

    # 1) 笔画密度
    stroke_density = float(binary.sum() / total_pixels) if total_pixels > 0 else 0.0

    # 2) 连通块数（不含背景）
    num_labels, _ = cv2.connectedComponents(bw)
    connected_components = int(max(num_labels - 1, 0))

    # 3) 外轮廓
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        perimeter = 0.0
        area = 0.0
        corners = 0
    else:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))
        area = float(cv2.contourArea(cnt))

        epsilon = 0.01 * perimeter if perimeter > 0 else 0.01
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        corners = int(len(approx))

    # 4) 骨架
    skeleton = thinning(bw)
    skel_points = np.column_stack(np.where(skeleton > 0))
    skeleton_length = int(len(skel_points))

    # 骨架分叉点数（>=3 邻居）
    skeleton_branch_points = 0
    for (y, x) in skel_points:
        y0 = max(0, y-1)
        y1 = min(h, y+2)
        x0 = max(0, x-1)
        x1 = min(w, x+2)
        neighbors = skeleton[y0:y1, x0:x1]
        count = int(np.count_nonzero(neighbors) - 1)  # exclude self
        if count >= 3:
            skeleton_branch_points += 1

    # 5) Hu Moments（不变矩）
    moments = cv2.moments(bw)
    hu = cv2.HuMoments(moments).flatten()
    # log transform 方便比较
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return {
        "stroke_density": stroke_density,
        "connected_components": connected_components,
        "contour_perimeter": perimeter,
        "contour_area": area,
        "corner_points": corners,
        "skeleton_length": skeleton_length,
        "skeleton_branch_points": skeleton_branch_points,
        "hu_moments": hu,
    }


# ======================================================
# 2. 可视化：骨架对比
# ======================================================

def cv_compare_visual(bw_oracle: np.ndarray, bw_egypt: np.ndarray):
    """
    输入:
        bw_oracle, bw_egypt: 0/255 的二值图 (numpy 数组)
    输出:
        matplotlib 的 figure 对象（给 streamlit.pyplot 用）
    """
    import matplotlib.pyplot as plt

    # 同样保证 0/255
    if bw_oracle.max() <= 1:
        bw_o = (bw_oracle * 255).astype(np.uint8)
    else:
        bw_o = bw_oracle.astype(np.uint8)

    if bw_egypt.max() <= 1:
        bw_e = (bw_egypt * 255).astype(np.uint8)
    else:
        bw_e = bw_egypt.astype(np.uint8)

    skel_o = thinning(bw_o)
    skel_e = thinning(bw_e)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(bw_o, cmap="gray")
    axes[0, 0].set_title("Oracle BW")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(bw_e, cmap="gray")
    axes[0, 1].set_title("Egypt BW")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(skel_o, cmap="gray")
    axes[1, 0].set_title("Oracle Skeleton")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(skel_e, cmap="gray")
    axes[1, 1].set_title("Egypt Skeleton")
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig
