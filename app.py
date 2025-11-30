import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 从你的 cv_ana 里导入这两个（你已经有）
from cv_ana import extract_cv_features, cv_compare_visual


# ================== 基本配置 ==================
DATA_CSV = "tsne_all.csv"   # 主分析脚本输出的总表（含 tsne / umap）
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR      # 如果 csv 里的路径是相对工程根目录，这样就够了

st.set_page_config(
    page_title="甲骨文 vs 圣书体 · Embedding 可视化",
    layout="wide"
)


# ================== 路径修正函数 ==================
def resolve_path(p):
    """
    把 CSV 里的路径转换成绝对路径：
    - 如果本身就是绝对路径，直接用
    - 如果是相对路径，则认为是相对于 PROJECT_ROOT
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ================== 黑白预处理（和分析脚本保持一致） ==================
def preprocess_bw(path, script):
    """
    强制圣书体黑白化，甲骨文轻度二值化。
    返回 PIL Image（RGB）
    """
    real_path = resolve_path(path)
    img = Image.open(real_path).convert("RGB")
    gray = np.array(img.convert("L"))

    if script == "egypt":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    if bw.mean() < 127:
        bw = 255 - bw

    return Image.fromarray(bw).convert("RGB")


# ================== 形状特征（雷达图用） ==================
def compute_shape_features_from_array(gray_arr):
    h, w = gray_arr.shape
    total_pixels = h * w

    binary = (gray_arr < 128).astype(np.uint8)

    # 1. 笔画密度
    stroke_pixels = binary.sum()
    density = stroke_pixels / total_pixels if total_pixels > 0 else 0.0

    # 2. 竖直对称
    mid_w = w // 2
    left = binary[:, :mid_w]
    right = binary[:, -mid_w:]
    right_flipped = np.fliplr(right)
    if left.shape[1] != right_flipped.shape[1]:
        min_w = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :min_w]
        right_flipped = right_flipped[:, :min_w]
    vsym = 1.0 - np.mean(np.abs(left - right_flipped))

    # 3. 水平对称
    mid_h = h // 2
    up = binary[:mid_h, :]
    down = binary[-mid_h:, :]
    down_flipped = np.flipud(down)
    if up.shape[0] != down_flipped.shape[0]:
        min_h = min(up.shape[0], down_flipped.shape[0])
        up = up[:min_h, :]
        down_flipped = down_flipped[:min_h, :]
    hsym = 1.0 - np.mean(np.abs(up - down_flipped))

    # 4. 中心集中度
    ys, xs = np.where(binary == 1)
    if len(xs) == 0:
        centralization = 0.0
    else:
        cx, cy = w / 2.0, h / 2.0
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        if max_dist > 0:
            norm_mean_dist = dists.mean() / max_dist
            centralization = 1.0 - norm_mean_dist
        else:
            centralization = 0.0

    # 5. 连通块数量（0~1 归一）
    cc_img = (binary * 255).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(cc_img)
    comp_count = max(num_labels - 1, 0)
    max_comp_assumed = 6
    comp_norm = min(comp_count, max_comp_assumed) / max_comp_assumed

    return {
        "stroke_density": float(density),
        "vertical_symmetry": float(vsym),
        "horizontal_symmetry": float(hsym),
        "centralization": float(centralization),
        "component_count": float(comp_norm),
    }


def compute_shape_features_for_image(path, script):
    img = preprocess_bw(path, script)
    gray = np.array(img.convert("L"))
    return compute_shape_features_from_array(gray)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_CSV)
    return df


@st.cache_data
def build_shape_feature_table(df):
    """
    针对每张图算特征 → 再按 (label, script) 聚合平均 → 再归一化 0~1
    """
    rows = []
    for _, row in df.iterrows():
        feats = compute_shape_features_for_image(row["file"], row["script"])
        feats["label"] = row["label"]
        feats["script"] = row["script"]
        rows.append(feats)

    feat_df = pd.DataFrame(rows)

    group = feat_df.groupby(["label", "script"]).mean().reset_index()

    feat_cols = [
        "stroke_density",
        "vertical_symmetry",
        "horizontal_symmetry",
        "centralization",
        "component_count",
    ]

    norm_group = group.copy()
    for col in feat_cols:
        col_min = group[col].min()
        col_max = group[col].max()
        if col_max > col_min:
            norm_group[col] = (group[col] - col_min) / (col_max - col_min)
        else:
            norm_group[col] = 0.5  # 全部一样给 0.5

    return group, norm_group


# ================== CV 数值对比辅助函数 ==================
def compute_cv_features_for_image(path, script):
    """
    结合当前 preprocess_bw + cv_ana.extract_cv_features:
    - preprocess_bw → PIL Image (RGB)
    - 转成灰度 → 阈值 → 0/255 二值数组 → 喂给 extract_cv_features
    """
    img = preprocess_bw(path, script)
    gray = np.array(img.convert("L"))
    bw = (gray < 128).astype(np.uint8) * 255
    return bw, extract_cv_features(bw)


def compare_features(f_oracle, f_egypt):
    """
    根据你在 cv_ana 里的设计，这里假设 extract_cv_features 返回的 dict 至少包含：
      stroke_density, connected_components, contour_perimeter, contour_area,
      corner_points, skeleton_length, skeleton_branch_points, hu_moments
    """
    diff = {}

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
        diff[k] = f_oracle.get(k, 0.0) - f_egypt.get(k, 0.0)

    # Hu Moments 距离
    hu_o = f_oracle.get("hu_moments", None)
    hu_e = f_egypt.get("hu_moments", None)
    if hu_o is not None and hu_e is not None:
        hu_dist = float(np.linalg.norm(np.array(hu_o) - np.array(hu_e)))
    else:
        hu_dist = None
    diff["hu_distance"] = hu_dist

    return diff


def cv_radar_plot(f_oracle, f_egypt, title="CV Radar Comparison"):
    """
    CV 特征雷达图（Plotly）
    """
    dims = [
        ("stroke_density", "笔画密度"),
        ("connected_components", "连通块数"),
        ("corner_points", "角点数（拐点）"),
        ("skeleton_branch_points", "骨架分叉数"),
        ("contour_perimeter", "外轮廓周长"),
        ("contour_area", "外轮廓面积"),
    ]

    oracle_vals = [float(f_oracle.get(k, 0.0)) for k, _ in dims]
    egypt_vals  = [float(f_egypt.get(k, 0.0))  for k, _ in dims]

    all_vals = oracle_vals + egypt_vals
    min_v = min(all_vals)
    max_v = max(all_vals)
    if max_v - min_v < 1e-6:
        oracle_norm = [0.5] * len(dims)
        egypt_norm  = [0.5] * len(dims)
    else:
        oracle_norm = [(v - min_v) / (max_v - min_v) for v in oracle_vals]
        egypt_norm  = [(v - min_v) / (max_v - min_v) for v in egypt_vals]

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


# ================== 加载数据 & 特征 ==================
st.sidebar.title("配置")

with st.spinner("加载数据中..."):
    df = load_data()

with st.spinner("计算形状特征中（用于雷达图，仅首次较慢）..."):
    group, norm_group = build_shape_feature_table(df)

labels_all = sorted(df["label"].unique())


# ================== Streamlit UI ==================
st.title("甲骨文 vs 圣书体 · Embedding & CV 可视化（Streamlit）")

tab_global, tab_single, tab_cv = st.tabs([
    "🌐 全局散点图",
    "🔍 单字对比 + 结构雷达图",
    "🧬 CV 字形结构对比"
])


# ---------- Tab 1: 全局散点 ----------
with tab_global:
    st.subheader("全局 UMAP / t-SNE")

    projection = st.radio(
        "选择降维方式：",
        ["UMAP", "t-SNE"],
        horizontal=True,
        key="global_proj"
    )

    color_mode = st.radio(
        "颜色编码：",
        ["按 script 着色（oracle vs egypt）", "按 label 着色（不同字不同颜色）"],
        horizontal=False,
        key="global_color"
    )

    if projection == "UMAP":
        x_col, y_col = "umap_x", "umap_y"
    else:
        x_col, y_col = "tsne_x", "tsne_y"

    if color_mode.startswith("按 script"):
        color_col = "script"
        color_map = {"oracle": "red", "egypt": "blue"}
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_map=color_map,
            hover_data=["file", "label", "script", "gardiner_code"],
            title=f"全局 {projection}: 甲骨文 vs 圣书体"
        )
    else:
        color_col = "label"
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_data=["file", "label", "script", "gardiner_code"],
            title=f"全局 {projection}: 按字着色"
        )

    fig.update_layout(
        width=700,
        height=700,
        legend_title_text=color_col
    )
    # 保持 1:1 比例，防止拉伸
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig, use_container_width=False)


# ---------- Tab 2: 单字对比 + 雷达图 ----------
with tab_single:
    st.subheader("单字：甲骨文 vs 圣书体 对比")

    c1, c2 = st.columns([1, 2])

    with c1:
        selected_label = st.selectbox(
            "选择一个字：",
            labels_all,
            index=labels_all.index("鬼") if "鬼" in labels_all else 0
        )

        proj = st.radio(
            "降维方式：",
            ["UMAP", "t-SNE"],
            horizontal=True,
            key="single_proj"
        )

        show_all_points = st.checkbox(
            "把其它字也显示出来（淡色背景）",
            value=False
        )

    sub = df[df["label"] == selected_label]

    if proj == "UMAP":
        x_col, y_col = "umap_x", "umap_y"
    else:
        x_col, y_col = "tsne_x", "tsne_y"

    with c2:
        st.markdown(f"### {selected_label} 的 {proj} 散点图（甲骨文 vs 圣书体）")

        if show_all_points:
            base = df
            base_color = base["script"].map({"oracle": "rgba(255,0,0,0.15)",
                                             "egypt": "rgba(0,0,255,0.15)"})
            # 先画淡色背景
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=base[x_col],
                y=base[y_col],
                mode="markers",
                marker=dict(color=base_color, size=4),
                showlegend=False,
                hoverinfo="skip"
            ))
        else:
            fig2 = go.Figure()

        # 再画当前字，红=oracle 蓝=egypt
        for script_name, color in [("oracle", "red"), ("egypt", "blue")]:
            sub_s = sub[sub["script"] == script_name]
            if len(sub_s) == 0:
                continue
            fig2.add_trace(go.Scatter(
                x=sub_s[x_col],
                y=sub_s[y_col],
                mode="markers",
                marker=dict(color=color, size=10),
                name=script_name,
                text=sub_s["file"],
                hovertemplate="(%{x}, %{y})<br>%{text}<extra></extra>"
            ))

        fig2.update_layout(
            title=f"{selected_label} - {proj}: 甲骨文 (red) vs 圣书体 (blue)",
            width=700,
            height=700,
            xaxis_title=x_col,
            yaxis_title=y_col,
        )
        fig2.update_yaxes(scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig2, use_container_width=False)

    st.markdown("---")

    # ====== 雷达图区域 ======
    st.markdown(f"### {selected_label} 的结构特征雷达图（甲骨文 vs 圣书体）")

    feat_cols = [
        "stroke_density",
        "vertical_symmetry",
        "horizontal_symmetry",
        "centralization",
        "component_count",
    ]
    feat_names_cn = ["笔画密度", "竖对称", "横对称", "中心集中度", "连通块数"]

    sub_norm = norm_group[norm_group["label"] == selected_label]

    if sub_norm.empty:
        st.info("这个字没有结构特征数据（可能没有对应图片）。")
    else:
        def get_vals(script):
            row = sub_norm[sub_norm["script"] == script]
            if row.empty:
                return None
            vals = [row.iloc[0][c] for c in feat_cols]
            return vals + [vals[0]]

        oracle_vals = get_vals("oracle")
        egypt_vals = get_vals("egypt")

        radar_fig = go.Figure()

        if oracle_vals is not None:
            radar_fig.add_trace(go.Scatterpolar(
                r=oracle_vals,
                theta=feat_names_cn + [feat_names_cn[0]],
                fill="toself",
                name="甲骨文",
                line=dict(color="red"),
            ))

        if egypt_vals is not None:
            radar_fig.add_trace(go.Scatterpolar(
                r=egypt_vals,
                theta=feat_names_cn + [feat_names_cn[0]],
                fill="toself",
                name="圣书体",
                line=dict(color="blue"),
            ))

        radar_fig.update_layout(
            title=f"{selected_label} - 结构特征雷达图",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            width=600,
            height=600,
        )

        st.plotly_chart(radar_fig, use_container_width=False)

        st.caption("说明：特征已在所有字 / 系统上做 0–1 归一，用于比较“形状”而非绝对量。")


# ---------- Tab 3: CV 字形结构对比 ----------
# ---------- Tab 3: CV 字形结构对比 ----------
with tab_cv:
    st.subheader("🧬 CV 字形结构对比：自由选择甲骨文 & 圣书体图像")

    # 选择字
    selected_label_cv = st.selectbox(
        "选择一个（字形类别）：",
        labels_all,
        key="cv_label"
    )

    # 获取该字的全部甲骨文 + 圣书体样本
    df_o = df[(df.label == selected_label_cv) & (df.script == "oracle")]
    df_e = df[(df.label == selected_label_cv) & (df.script == "egypt")]

    if df_o.empty:
        st.error("⚠ 这个字没有甲骨文图像。")
        st.stop()

    if df_e.empty:
        st.error("⚠ 这个字没有圣书体图像。")
        st.stop()

    # ==== 下拉：选择甲骨文的具体图像 ====
    st.markdown("### 🔴 选择甲骨文图像")
    oracle_options = df_o["file"].tolist()

    selected_oracle_file = st.selectbox(
        "选择甲骨文文件：",
        oracle_options,
        index=0,
        key="select_oracle_image"
    )

    # ==== 下拉：选择圣书体的具体图像 ====
    st.markdown("### 🔵 选择圣书体图像")
    egypt_options = df_e["file"].tolist()

    selected_egypt_file = st.selectbox(
        "选择圣书体文件：",
        egypt_options,
        index=0,
        key="select_egypt_image"
    )

    # ======== 计算 CV 特征 ========
    bw_o, feats_o = compute_cv_features_for_image(selected_oracle_file, "oracle")
    bw_e, feats_e = compute_cv_features_for_image(selected_egypt_file, "egypt")

    if bw_o is None or feats_o is None:
        st.error(f"❌ 无法读取甲骨文图像：{resolve_path(selected_oracle_file)}")
        st.stop()

    if bw_e is None or feats_e is None:
        st.error(f"❌ 无法读取圣书体图像：{resolve_path(selected_egypt_file)}")
        st.stop()

    diffs = compare_features(feats_o, feats_e)

    # ======== 显示当前选中的文件路径 ========
    st.markdown("### 📂 当前对比的文件")
    st.write("甲骨文：", resolve_path(selected_oracle_file))
    st.write("圣书体：", resolve_path(selected_egypt_file))

    # ======== 图片预览 ========
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🔴 甲骨文（预处理后）")
        st.image(bw_o, width=250)

    with col2:
        st.markdown("### 🔵 圣书体（预处理后）")
        st.image(bw_e, width=250)

    st.markdown("---")

    # ======== CV 数值对比表 ========
    st.markdown("### 📏 CV 数值对比（甲骨文 - 圣书体）")

    df_show = pd.DataFrame({
        "指标": [
            "笔画密度",
            "连通块数",
            "外轮廓周长",
            "外轮廓面积",
            "角点数量",
            "骨架长度",
            "骨架分叉点数",
            "Hu Moments 距离",
        ],
        "甲骨文": [
            feats_o.get("stroke_density"),
            feats_o.get("connected_components"),
            feats_o.get("contour_perimeter"),
            feats_o.get("contour_area"),
            feats_o.get("corner_points"),
            feats_o.get("skeleton_length"),
            feats_o.get("skeleton_branch_points"),
            None,
        ],
        "圣书体": [
            feats_e.get("stroke_density"),
            feats_e.get("connected_components"),
            feats_e.get("contour_perimeter"),
            feats_e.get("contour_area"),
            feats_e.get("corner_points"),
            feats_e.get("skeleton_length"),
            feats_e.get("skeleton_branch_points"),
            None,
        ],
        "差值(甲 - 埃)": [
            diffs.get("stroke_density"),
            diffs.get("connected_components"),
            diffs.get("contour_perimeter"),
            diffs.get("contour_area"),
            diffs.get("corner_points"),
            diffs.get("skeleton_length"),
            diffs.get("skeleton_branch_points"),
            diffs.get("hu_distance"),
        ]
    })

    st.dataframe(df_show)

    st.markdown("---")

    # ======== 骨架与轮廓可视化 ========
    st.markdown("### 🕸️ 骨架 + 轮廓结构可视化")
    st.pyplot(cv_compare_visual(bw_o, bw_e))

    st.markdown("---")

    # ======== CV 雷达图 ========
    st.markdown(f"### 📊 CV 多维结构雷达图：{selected_label_cv}")

    radar_cv = cv_radar_plot(
        feats_o,
        feats_e,
        title=f"{selected_label_cv} - CV 字形结构雷达图"
    )
    st.plotly_chart(radar_cv, use_container_width=False)

