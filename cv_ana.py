import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go


# ===========================
# 1) 黑白预处理
# ===========================
def preprocess_bw(path, script):
    """
    统一圣书体 / 甲骨文的黑白风格，使 CV 特征可比较
    """
    path = Path(path)
    img = Image.open(path).convert("RGB")
    gray = np.array(img.convert("L"))

    if script == "egypt":
        # 圣书体灰白 → 二值化
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # 甲骨文刻痕 → 轻二值化
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 保持黑为笔画
    if bw.mean() < 127:
        bw = 255 - bw

    return bw


# ===========================
# 2) CV 特征提取
# ===========================
def extract_cv_features(bw_img):
    """
    输入：二值图（0=黑，255=白）
    输出：结构化字形特征
    """

    # ---- Binary 0/1 ----
    binary = (bw_img < 128).astype(np.uint8)

    # ---- Connected Components (连通块) ----
    num_labels, _ = cv2.connectedComponents((binary * 255).astype(np.uint8))
    comp_count = num_labels - 1

    # ---- Stroke Density（黑像素密度）----
    stroke_density = binary.mean()

    # ---- Contours (外轮廓) ----
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        perimeter = 0.0
        area = 0.0
        corners = 0
    else:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))
        area = float(cv2.contourArea(cnt))

        # Corner detection（粗略笔画数量 proxy）
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        corners = len(approx)

    # ---- Skeleton（骨架）----
    skeleton = cv2.ximgproc.thinning((binary * 255).astype(np.uint8))
    skel_points = np.column_stack(np.where(skeleton == 255))
    skel_len = len(skel_points)

    # ---- Skeleton branch detection ----
    branch_pts = 0
    for y, x in skel_points:
        # Count skeleton neighbors
        neighbors = skeleton[max(0,y-1):y+2, max(0,x-1):x+2]
        count = np.count_nonzero(neighbors) - 1
        if count >= 3:  # >=3 neighbors → branch point
            branch_pts += 1

    # ---- Hu Moments（七大不变矩） ----
    moments = cv2.moments((binary * 255).astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)  # log transform

    return {
        "stroke_density": stroke_density,
        "connected_components": comp_count,
        "contour_perimeter": perimeter,
        "contour_area": area,
        "corner_points": corners,
        "skeleton_length": skel_len,
        "skeleton_branch_points": branch_pts,
        "hu_moments": hu,
    }


# ===========================
# 3) CV 特征对比
# ===========================
def compare_features(f_oracle, f_egypt):
    """
    计算数值差异，包括 Hu Moments 距离
    """
    diff = {}

    # scalar features
    keys = [
        "stroke_density",
        "connected_components",
        "contour_perimeter",
        "contour_area",
        "corner_points",
        "skeleton_length",
        "skeleton_branch_points",
    ]

    for k in keys:
        diff[k] = f_oracle[k] - f_egypt[k]

    # Hu Moments 距离
    hu_dist = np.linalg.norm(f_oracle["hu_moments"] - f_egypt["hu_moments"])
    diff["hu_distance"] = float(hu_dist)

    return diff


# ===========================
# 4) 生成 CV 雷达图（Plotly）
# ===========================
def cv_radar_plot(f_oracle, f_egypt, title="CV Radar Comparison"):
    """
    输入：两个特征 dict
    输出：Plotly 雷达图 figure
    """

    # 选择可归一比较的标量特征：
    dims = [
        ("stroke_density", "笔画密度"),
        ("connected_components", "连通块数"),
        ("corner_points", "角点数（笔画拐点）"),
        ("skeleton_branch_points", "骨架分叉数"),
        ("contour_perimeter", "外轮廓周长"),
        ("contour_area", "外轮廓面积"),
    ]

    oracle_vals = [f_oracle[k] for k, _ in dims]
    egypt_vals  = [f_egypt[k] for k, _ in dims]

    # 归一化到 0-1
    all_vals = oracle_vals + egypt_vals
    min_v = min(all_vals)
    max_v = max(all_vals)
    if max_v - min_v < 1e-6:
        oracle_norm = [0.5]*len(dims)
        egypt_norm  = [0.5]*len(dims)
    else:
        oracle_norm = [(v-min_v)/(max_v-min_v) for v in oracle_vals]
        egypt_norm  = [(v-min_v)/(max_v-min_v) for v in egypt_vals]

    # 闭合
    oracle_norm += [oracle_norm[0]]
    egypt_norm  += [egypt_norm[0]]
    labels = [name for _, name in dims] + [dims[0][1]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=oracle_norm,
        theta=labels,
        fill='toself',
        name='甲骨文',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatterpolar(
        r=egypt_norm,
        theta=labels,
        fill='toself',
        name='圣书体',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        width=600,
        height=600,
    )
    return fig


# ===========================
# 5) 可视化：骨架、轮廓（可保存）
# ===========================
def cv_compare_visual(bw_oracle, bw_egypt, save_path=None):
    """
    输出：一个包含轮廓 + 骨架并排的可视化图
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0,0].imshow(bw_oracle, cmap='gray')
    axes[0,0].set_title("Oracle (BW)")
    axes[0,0].axis("off")

    axes[0,1].imshow(bw_egypt, cmap='gray')
    axes[0,1].set_title("Egypt (BW)")
    axes[0,1].axis("off")

    # Skeleton 可视化
    skel_o = cv2.ximgproc.thinning((bw_oracle < 128).astype(np.uint8)*255)
    skel_e = cv2.ximgproc.thinning((bw_egypt < 128).astype(np.uint8)*255)

    axes[1,0].imshow(skel_o, cmap='gray')
    axes[1,0].set_title("Oracle Skeleton")
    axes[1,0].axis("off")

    axes[1,1].imshow(skel_e, cmap='gray')
    axes[1,1].set_title("Egypt Skeleton")
    axes[1,1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()

    return fig
