# cv_ana.py  —— 不依赖 cv2 的版本
import numpy as np
from typing import Dict, Any

from skimage.measure import (
    label as sk_label,
    regionprops,
    find_contours,
    moments,
    moments_central,
    moments_normalized,
)


# ======================================================
# 0. 安全骨架化：不再依赖 cv2.ximgproc
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
                    img[i - 1, j],     # p2
                    img[i - 1, j + 1], # p3
                    img[i, j + 1],     # p4
                    img[i + 1, j + 1], # p5
                    img[i + 1, j],     # p6
                    img[i + 1, j - 1], # p7
                    img[i, j - 1],     # p8
                    img[i - 1, j - 1], # p9
                ]
                C = sum(n)
                if C < 2 or C > 6:
                    continue
                transitions = sum(
                    (n[k] == 0 and n[(k + 1) % 8] == 1) for k in range(8)
                )
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
                    img[i - 1, j],     # p2
                    img[i - 1, j + 1], # p3
                    img[i, j + 1],     # p4
                    img[i + 1, j + 1], # p5
                    img[i + 1, j],     # p6
                    img[i + 1, j - 1], # p7
                    img[i, j - 1],     # p8
                    img[i - 1, j - 1], # p9
                ]
                C = sum(n)
                if C < 2 or C > 6:
                    continue
                transitions = sum(
                    (n[k] == 0 and n[(k + 1) % 8] == 1) for k in range(8)
                )
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
    统一骨架化接口。
    以前这里会优先用 cv2.ximgproc，现在完全去掉 cv2 依赖，
    直接使用 Zhang-Suen 细化。
    """
    return _simple_thinning(binary255)


# ======================================================
# 0.x 一些辅助函数（多边形、Hu 矩等）
# ======================================================

def _polygon_perimeter(poly: np.ndarray) -> float:
    """
    poly: (N, 2) 的坐标数组 (row, col)
    计算闭合折线的周长
    """
    if poly.shape[0] < 2:
        return 0.0
    # 相邻点距离
    diffs = np.diff(poly, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1)).sum()
    # 首尾闭合
    seg_len += float(np.linalg.norm(poly[0] - poly[-1]))
    return float(seg_len)


def _polygon_area(poly: np.ndarray) -> float:
    """
    Shoelace 公式计算多边形面积
    poly: (N, 2) (row, col)
    """
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 1]
    y = poly[:, 0]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker 多边形简化算法，用于估计“角点数量”
    points: (N, 2)
    """
    if points.shape[0] < 3:
        return points

    # 起点终点
    start = points[0]
    end = points[-1]

    # 直线向量
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        # 所有点重合
        return points[[0, -1]]

    # 计算每个点到直线的距离
    vecs = points[1:-1] - start
    # 叉积长度 / 线段长度
    cross = np.abs(
        line[0] * vecs[:, 1] - line[1] * vecs[:, 0]
    )
    dists = cross / line_norm
    idx = np.argmax(dists)
    dmax = dists[idx]

    if dmax > epsilon:
        # 递归分段
        left = _rdp(points[: idx + 2], epsilon)
        right = _rdp(points[idx + 1 :], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([start, end])


def _hu_moments_from_binary(binary: np.ndarray) -> np.ndarray:
    """
    用 scikit-image 的 moments 系列计算 Hu 不变矩
    输入 binary: 0/1 或 bool
    输出: 7 维 numpy array（log10 + 符号）
    """
    img = (binary > 0).astype(float)
    m = moments(img, order=3)
    if m[0, 0] == 0:
        return np.zeros(7, dtype=float)

    # 质心
    cy = m[1, 0] / m[0, 0]
    cx = m[0, 1] / m[0, 0]
    mu = moments_central(img, center=(cy, cx), order=3)
    nu = moments_normalized(mu, order=3)

    # 对应 ηpq
    eta20 = nu[2, 0]
    eta02 = nu[0, 2]
    eta11 = nu[1, 1]
    eta30 = nu[3, 0]
    eta03 = nu[0, 3]
    eta21 = nu[2, 1]
    eta12 = nu[1, 2]

    # 7 个 Hu invariants
    phi = np.zeros(7, dtype=float)

    phi[0] = eta20 + eta02
    phi[1] = (eta20 - eta02) ** 2 + 4 * (eta11 ** 2)
    phi[2] = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
    phi[3] = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2
    phi[4] = (
        (eta30 - 3 * eta12)
        * (eta30 + eta12)
        * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
        + (3 * eta21 - eta03)
        * (eta21 + eta03)
        * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    )
    phi[5] = (
        (eta20 - eta02)
        * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
        + 4 * eta11 * (eta30 + eta12) * (eta21 + eta03)
    )
    phi[6] = (
        (3 * eta21 - eta03)
        * (eta30 + eta12)
        * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
        - (eta30 - 3 * eta12)
        * (eta21 + eta03)
        * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    )

    # 与原 cv2 版本一致：log10 + sign
    hu = np.sign(phi) * np.log10(np.abs(phi) + 1e-12)
    return hu


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
    labels = sk_label(binary, connectivity=2)
    connected_components = int(labels.max())  # 背景为 0，前景从 1 开始

    # 3) 外轮廓（用 skimage.find_contours + 自算 perimeter / area）
    contours = find_contours(binary, level=0.5)
    if len(contours) == 0:
        perimeter = 0.0
        area = 0.0
        corners = 0
    else:
        # 用点数最多的轮廓近似“最大外轮廓”
        cnt = max(contours, key=lambda c: c.shape[0])
        perimeter = _polygon_perimeter(cnt)
        area = _polygon_area(cnt)

        # 用 RDP 简化后的顶点数作为“角点数量”近似
        epsilon = 0.01 * perimeter if perimeter > 0 else 0.5
        approx = _rdp(cnt, epsilon)
        corners = int(max(approx.shape[0], 1))

    # 4) 骨架
    skeleton = thinning(bw)
    skel_points = np.column_stack(np.where(skeleton > 0))
    skeleton_length = int(len(skel_points))

    # 骨架分叉点数（>=3 邻居）
    skeleton_branch_points = 0
    for (y, x) in skel_points:
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)
        x0 = max(0, x - 1)
        x1 = min(w, x + 2)
        neighbors = skeleton[y0:y1, x0:x1]
        count = int(np.count_nonzero(neighbors) - 1)  # exclude self
        if count >= 3:
            skeleton_branch_points += 1

    # 5) Hu Moments（不变矩），基于二值图像
    hu = _hu_moments_from_binary(binary)

    return {
        "stroke_density": stroke_density,
        "connected_components": connected_components,
        "contour_perimeter": float(perimeter),
        "contour_area": float(area),
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
