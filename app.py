import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time

from cv_ana import extract_cv_features, cv_compare_visual


# ================== Global Configuration ==================
DATA_CSV = "tsne_all.csv"   # Output table from main processing pipeline (with tsne/umap)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR      # If CSV paths are relative to project root, this is sufficient

st.set_page_config(
    page_title="Oracle Bone Script vs Hieroglyphs · Embedding Visualization",
    layout="wide"
)


# ================== Path Normalization ==================
def resolve_path(p):
    """Normalize path, convert backslashes to forward slashes, ensure absolute path."""
    p = str(p).replace("\\", "/")
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ================== Binarization / Preprocessing ==================
def preprocess_bw(path, script):
    """
    Normalize visual style for both scripts:
    - Hieroglyphs: Otsu threshold
    - Oracle bone script: fixed threshold (200)
    Returns a PIL RGB image.
    """
    real_path = resolve_path(path)
    img = Image.open(real_path).convert("RGB")
    gray = np.array(img.convert("L"))

    if script == "egypt":
        th = threshold_otsu(gray)
    else:
        th = 200

    bw = (gray >= th).astype(np.uint8) * 255

    # Ensure dark foreground: invert if image is globally darker
    if bw.mean() < 127:
        bw = 255 - bw

    return Image.fromarray(bw).convert("RGB")


# ================== Shape Feature Computation (Radar Chart) ==================
def compute_shape_features_from_array(gray_arr):
    h, w = gray_arr.shape
    total_pixels = h * w

    binary = (gray_arr < 128).astype(np.uint8)

    # 1. Stroke density
    stroke_pixels = binary.sum()
    density = stroke_pixels / total_pixels if total_pixels > 0 else 0.0

    # 2. Vertical symmetry
    mid_w = w // 2
    left = binary[:, :mid_w]
    right = binary[:, -mid_w:]
    right_flipped = np.fliplr(right)

    if left.shape[1] != right_flipped.shape[1]:
        min_w = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :min_w]
        right_flipped = right_flipped[:, :min_w]

    vsym = 1.0 - np.mean(np.abs(left - right_flipped))

    # 3. Horizontal symmetry
    mid_h = h // 2
    upper = binary[:mid_h, :]
    lower = binary[-mid_h:, :]
    lower_flipped = np.flipud(lower)

    if upper.shape[0] != lower_flipped.shape[0]:
        min_h = min(upper.shape[0], lower_flipped.shape[0])
        upper = upper[:min_h, :]
        lower_flipped = lower_flipped[:min_h, :]

    hsym = 1.0 - np.mean(np.abs(upper - lower_flipped))

    # 4. Centrality
    ys, xs = np.where(binary == 1)
    if len(xs) == 0:
        centralization = 0.0
    else:
        cx, cy = w / 2, h / 2
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        if max_dist > 0:
            centralization = 1.0 - dists.mean() / max_dist
        else:
            centralization = 0.0

    # 5. Connected components (normalized)
    if stroke_pixels == 0:
        comp_count = 0
    else:
        labels = cc_label(binary, connectivity=2)
        comp_count = int(labels.max())

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
    return pd.read_csv(DATA_CSV)


@st.cache_data
def build_shape_feature_table(df):
    """
    Compute shape features for every image → group by (label, script) → normalize 0–1.
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
        col_min, col_max = group[col].min(), group[col].max()
        if col_max > col_min:
            norm_group[col] = (group[col] - col_min) / (col_max - col_min)
        else:
            norm_group[col] = 0.5

    return group, norm_group


# ================== CV Feature Comparison Helpers ==================
def compute_cv_features_for_image(path, script):
    """
    Preprocess → gray → binary → call extract_cv_features.
    """
    img = preprocess_bw(path, script)
    gray = np.array(img.convert("L"))
    bw = (gray < 128).astype(np.uint8) * 255
    return bw, extract_cv_features(bw)


def compare_features(f_oracle, f_egypt):
    """
    Compare CV features defined in extract_cv_features.
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

    # Hu Moments distance
    hu_o = f_oracle.get("hu_moments")
    hu_e = f_egypt.get("hu_moments")
    diff["hu_distance"] = (
        float(np.linalg.norm(np.array(hu_o) - np.array(hu_e)))
        if hu_o is not None and hu_e is not None else None
    )

    return diff


def cv_radar_plot(f_oracle, f_egypt, title="CV Radar Comparison"):
    """
    Radar plot for CV features.
    """
    dims = [
        ("stroke_density", "Stroke Density"),
        ("connected_components", "Connected Components"),
        ("corner_points", "Corners"),
        ("skeleton_branch_points", "Skeleton Branches"),
        ("contour_perimeter", "Contour Perimeter"),
        ("contour_area", "Contour Area"),
    ]

    oracle_vals = [float(f_oracle.get(k, 0.0)) for k, _ in dims]
    egypt_vals  = [float(f_egypt.get(k, 0.0))  for k, _ in dims]

    all_vals = oracle_vals + egypt_vals
    min_v, max_v = min(all_vals), max(all_vals)

    if max_v - min_v < 1e-6:
        oracle_norm = egypt_norm = [0.5] * len(dims)
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
        name='Oracle Script',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatterpolar(
        r=egypt_norm,
        theta=labels,
        fill='toself',
        name='Hieroglyphs',
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


# ================== Load Data & Compute Features ==================
st.sidebar.title("Configuration")

mode = st.sidebar.radio(
    "Mode:",
    ["🎮 Mini Game", "📊 Analysis"],
    index=0
)

with st.spinner("Loading dataset..."):
    df = load_data()

with st.spinner("Computing shape features (first run may take longer)..."):
    group, norm_group = build_shape_feature_table(df)

labels_all = sorted(df["label"].unique())


# ================== Streamlit UI ==================
st.title("Oracle Bone Script vs Egyptian Hieroglyphs · Embedding & CV Visualization")

# --------------------------------------------------------------------------
# ----------------------------- ANALYSIS MODE -------------------------------
# --------------------------------------------------------------------------

if mode == "📊 Analysis":

    tab_global, tab_single, tab_cv = st.tabs([
        "🌐 Global Scatter",
        "🔍 Single Character + Shape Radar",
        "🧬 CV Structural Comparison",
    ])

    # ======================================================================
    # -------- Tab 1: Global UMAP / t-SNE ---------------------------------
    # ======================================================================
    with tab_global:
        st.subheader("Global UMAP / t-SNE Projection")

        projection = st.radio(
            "Dimensionality Reduction:",
            ["UMAP", "t-SNE"],
            horizontal=True,
            key="global_proj"
        )

        color_mode = st.radio(
            "Color Encoding:",
            ["By Script (oracle vs egypt)", "By Label"],
            horizontal=False,
            key="global_color"
        )

        x_col, y_col = (
            ("umap_x", "umap_y") if projection == "UMAP" else ("tsne_x", "tsne_y")
        )

        if color_mode.startswith("By Script"):
            color_col = "script"
            color_map = {"oracle": "red", "egypt": "blue"}
            fig = px.scatter(
                df,
                x=x_col, y=y_col,
                color=color_col,
                color_discrete_map=color_map,
                hover_data=["file", "label", "script", "gardiner_code"],
                title=f"Global {projection}: Oracle vs Egypt"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col, y=y_col,
                color="label",
                hover_data=["file", "label", "script", "gardiner_code"],
                title=f"Global {projection}: Colored by Label"
            )

        fig.update_layout(width=700, height=700, legend_title_text=color_col)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig, use_container_width=False)

    # ======================================================================
    # -------- Tab 2: Single Character + Radar Chart ----------------------
    # ======================================================================
    with tab_single:
        st.subheader("Single Character Comparison: Oracle vs Hieroglyphs")

        c1, c2 = st.columns([1, 2])

        with c1:
            selected_label = st.selectbox(
                "Select a character:",
                labels_all,
                index=labels_all.index("鬼") if "鬼" in labels_all else 0
            )

            proj = st.radio(
                "Projection:",
                ["UMAP", "t-SNE"],
                horizontal=True,
                key="single_proj"
            )

            show_all_points = st.checkbox(
                "Show other points as background (faded)",
                value=False
            )

        sub = df[df["label"] == selected_label]
        x_col, y_col = ("umap_x", "umap_y") if proj == "UMAP" else ("tsne_x", "tsne_y")

        with c2:
            st.markdown(f"### {selected_label} — {proj} Scatter (Oracle vs Egypt)")

            if show_all_points:
                base = df
                base_color = base["script"].map({
                    "oracle": "rgba(255,0,0,0.15)",
                    "egypt": "rgba(0,0,255,0.15)"
                })
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=base[x_col], y=base[y_col],
                    mode="markers",
                    marker=dict(color=base_color, size=4),
                    showlegend=False,
                    hoverinfo="skip"
                ))
            else:
                fig2 = go.Figure()

            for script_name, color in [("oracle", "red"), ("egypt", "blue")]:
                sub_s = sub[sub["script"] == script_name]
                if len(sub_s) > 0:
                    fig2.add_trace(go.Scatter(
                        x=sub_s[x_col], y=sub_s[y_col],
                        mode="markers",
                        marker=dict(color=color, size=10),
                        name=script_name,
                        text=sub_s["file"],
                        hovertemplate="(%{x}, %{y})<br>%{text}<extra></extra>"
                    ))

            fig2.update_layout(
                title=f"{selected_label} — {proj}: Oracle (red) vs Egypt (blue)",
                width=700, height=700,
                xaxis_title=x_col, yaxis_title=y_col,
            )
            fig2.update_yaxes(scaleanchor="x", scaleratio=1)

            st.plotly_chart(fig2, use_container_width=False)

        st.markdown("---")

        # ----------------- Radar Plot -----------------
        st.markdown(f"### Structural Radar Chart: {selected_label}")

        feat_cols = [
            "stroke_density",
            "vertical_symmetry",
            "horizontal_symmetry",
            "centralization",
            "component_count",
        ]
        feat_names = [
            "Stroke Density",
            "Vertical Symmetry",
            "Horizontal Symmetry",
            "Centralization",
            "Connected Components",
        ]

        sub_norm = norm_group[norm_group["label"] == selected_label]

        if sub_norm.empty:
            st.info("No structural features found for this label.")
        else:
            def get_vals(script):
                row = sub_norm[sub_norm["script"] == script]
                if row.empty:
                    return None
                vals = [row.iloc[0][c] for c in feat_cols]
                return vals + [vals[0]]

            oracle_vals = get_vals("oracle")
            egypt_vals  = get_vals("egypt")

            radar_fig = go.Figure()

            if oracle_vals:
                radar_fig.add_trace(go.Scatterpolar(
                    r=oracle_vals,
                    theta=feat_names + [feat_names[0]],
                    fill="toself",
                    name="Oracle",
                    line=dict(color="red"),
                ))

            if egypt_vals:
                radar_fig.add_trace(go.Scatterpolar(
                    r=egypt_vals,
                    theta=feat_names + [feat_names[0]],
                    fill="toself",
                    name="Egypt",
                    line=dict(color="blue"),
                ))

            radar_fig.update_layout(
                title=f"{selected_label} — Structural Radar Chart",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                width=600, height=600,
            )

            st.plotly_chart(radar_fig, use_container_width=False)

            st.caption(
                "Note: Features are normalized across all characters and both scripts (0–1)."
            )

    # ======================================================================
    # -------- Tab 3: CV Structural Comparison -----------------------------
    # ======================================================================
    with tab_cv:
        st.subheader("CV-Based Shape Comparison: Oracle vs Hieroglyphs")

        selected_label_cv = st.selectbox(
            "Select a label:",
            labels_all,
            key="cv_label"
        )

        df_o = df[(df.label == selected_label_cv) & (df.script == "oracle")]
        df_e = df[(df.label == selected_label_cv) & (df.script == "egypt")]

        if df_o.empty:
            st.error("No Oracle images available for this label.")
            st.stop()

        if df_e.empty:
            st.error("No Hieroglyph images available for this label.")
            st.stop()

        # Select oracle image
        st.markdown("### 🔴 Select Oracle Image")
        oracle_options = df_o["file"].tolist()
        selected_oracle_file = st.selectbox(
            "Oracle image:",
            oracle_options,
            index=0,
            key="select_oracle_image"
        )

        # Select egypt image
        st.markdown("### 🔵 Select Hieroglyph Image")
        egypt_options = df_e["file"].tolist()
        selected_egypt_file = st.selectbox(
            "Hieroglyph image:",
            egypt_options,
            index=0,
            key="select_egypt_image"
        )

        # Compute features
        bw_o, feats_o = compute_cv_features_for_image(selected_oracle_file, "oracle")
        bw_e, feats_e = compute_cv_features_for_image(selected_egypt_file, "egypt")

        if bw_o is None or feats_o is None:
            st.error(f"Cannot load Oracle image: {resolve_path(selected_oracle_file)}")
            st.stop()

        if bw_e is None or feats_e is None:
            st.error(f"Cannot load Egypt image: {resolve_path(selected_egypt_file)}")
            st.stop()

        diffs = compare_features(feats_o, feats_e)

        st.markdown("### 📂 Selected Files")
        st.write("Oracle:", resolve_path(selected_oracle_file))
        st.write("Egypt:", resolve_path(selected_egypt_file))

        # Preview images
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### 🔴 Oracle (Processed)")
            st.image(bw_o, width=250)

        with col2:
            st.markdown("### 🔵 Egypt (Processed)")
            st.image(bw_e, width=250)

        st.markdown("---")

        # CV metrics table
        st.markdown("### 📏 CV Feature Comparison (Oracle - Egypt)")

        df_show = pd.DataFrame({
            "Metric": [
                "Stroke Density",
                "Connected Components",
                "Contour Perimeter",
                "Contour Area",
                "Corner Points",
                "Skeleton Length",
                "Skeleton Branch Points",
                "Hu Moments Distance",
            ],
            "Oracle": [
                feats_o.get("stroke_density"),
                feats_o.get("connected_components"),
                feats_o.get("contour_perimeter"),
                feats_o.get("contour_area"),
                feats_o.get("corner_points"),
                feats_o.get("skeleton_length"),
                feats_o.get("skeleton_branch_points"),
                None,
            ],
            "Egypt": [
                feats_e.get("stroke_density"),
                feats_e.get("connected_components"),
                feats_e.get("contour_perimeter"),
                feats_e.get("contour_area"),
                feats_e.get("corner_points"),
                feats_e.get("skeleton_length"),
                feats_e.get("skeleton_branch_points"),
                None,
            ],
            "Diff (O - E)": [
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

        # Skeleton & contour visualization
        st.markdown("### 🕸️ Skeleton & Contour Visualization")
        st.pyplot(cv_compare_visual(bw_o, bw_e))

        st.markdown("---")

        # CV radar chart
        st.markdown(f"### 📊 CV Radar Chart: {selected_label_cv}")

        radar_cv = cv_radar_plot(
            feats_o,
            feats_e,
            title=f"{selected_label_cv} — CV Structural Radar"
        )
        st.plotly_chart(radar_cv, use_container_width=False)

# --------------------------------------------------------------------------
# ------------------------------ MINI GAME ---------------------------------
# --------------------------------------------------------------------------

else:

    st.subheader("🎮 Character Guessing Mini Game (2-Minute Challenge)")

    st.markdown(
        "Rules:\n"
        "- The system randomly selects an Oracle or Egyptian image.\n"
        "- Choose the corresponding character label.\n"
        "- After submitting, correctness is shown and a new question appears instantly.\n"
        "- Total time: 2 minutes. Try to get as many correct as possible!"
    )

    # ---- Initialize session state ----
    if "game_start_ts" not in st.session_state:
        st.session_state.game_start_ts = None
    if "quiz_row_idx" not in st.session_state:
        st.session_state.quiz_row_idx = None
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_total" not in st.session_state:
        st.session_state.quiz_total = 0

    def new_question():
        row = df.sample(1).iloc[0]
        st.session_state.quiz_row_idx = int(row.name)

    col_btn1, _ = st.columns(2)
    with col_btn1:
        if st.button("🚀 Start / Restart 2-Min Challenge"):
            st.session_state.game_start_ts = time.time()
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            new_question()
            st.rerun()

    if st.session_state.game_start_ts is None:
        st.info("Click the button above to start the challenge.")
    else:
        elapsed = time.time() - st.session_state.game_start_ts
        remaining = int(120 - elapsed)

        if remaining <= 0:
            st.error("⏰ Time's up!")
            st.write(f"Score: **{st.session_state.quiz_score} / {st.session_state.quiz_total}**")
            st.info("Click restart to try again.")
        else:
            st.markdown(f"⏱ Remaining Time: **{remaining} sec**")
            st.markdown(
                f"Current Score: **{st.session_state.quiz_score}** / **{st.session_state.quiz_total}**"
            )

            if st.session_state.quiz_row_idx is None:
                new_question()

            row = df.loc[st.session_state.quiz_row_idx]

            img_path = resolve_path(row["file"])
            try:
                img = Image.open(img_path).convert("RGB")
                st.image(
                    img,
                    caption=f"Script: {row['script']} (oracle / egypt)",
                    width=260
                )
            except Exception:
                st.error(f"Failed to load image: {img_path}")
                st.stop()

            st.markdown("#### Which character is this?")

            guess = st.selectbox(
                "Choose:",
                labels_all,
                key="quiz_guess"
            )

            if st.button("✅ Submit Answer"):
                correct_label = str(row["label"])
                st.session_state.quiz_total += 1

                if str(guess) == correct_label:
                    st.session_state.quiz_score += 1
                    st.success(f"Correct! It is **{correct_label}**.")
                else:
                    st.error(f"Incorrect. You chose {guess}, correct: **{correct_label}**.")

                extra = [f"Script: `{row['script']}`"]
                if "gardiner_code" in df.columns and not pd.isna(row.get("gardiner_code", None)):
                    extra.append(f"Gardiner Code: `{row['gardiner_code']}`")

                st.markdown(", ".join(extra))

                meaning_map = {
                    "日": "sun",
                    "月": "moon",
                    "星": "star",
                    "人": "person",
                    "帝": "high deity",
                    "鬼": "spirit",
                    "祖": "ancestor spirit",
                    "示": "altar",
                }
                if correct_label in meaning_map:
                    st.markdown(f"Meaning: {meaning_map[correct_label]}")

                new_question()
                st.rerun()
