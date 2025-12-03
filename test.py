import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import clip
import umap
from openTSNE import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ORACLE_ROOT = "oracle_selected_glyphs"
EGYPT_ROOT = "egypt_by_oracle"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_bw(path, script):
    img = Image.open(path).convert("RGB")
    gray = np.array(img.convert("L"))

    if script == "egypt":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    if bw.mean() < 127:
        bw = 255 - bw

    return Image.fromarray(bw).convert("RGB")


def plot_single_word_tsne_umap_images(df, output_dir=OUTPUT_DIR, img_size=32):
    labels = sorted(df["label"].unique())

    for label in labels:
        sub = df[df["label"] == label].copy()
        if len(sub) < 2:
            print(f"[INFO] 字 {label} 样本不足，跳过图片散点。")
            continue

        print(f"[INFO] 生成 {label} 的 UMAP / t-SNE 图片对比图")

        # UMAP
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"{label} - UMAP 图片对比 (Oracle vs Egypt)")
        ax.set_xlabel("umap_x")
        ax.set_ylabel("umap_y")

        for _, row in sub.iterrows():
            x, y = row["umap_x"], row["umap_y"]
            img = preprocess_bw(row["file"], row["script"])
            img = img.resize((img_size, img_size), resample=Image.NEAREST)

            if row["script"] == "oracle":
                ax.scatter([x], [y], c="red", s=10, alpha=0.3)
            else:
                ax.scatter([x], [y], c="blue", s=10, alpha=0.3)

            ab = AnnotationBbox(OffsetImage(img), (x, y), frameon=False)
            ax.add_artist(ab)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{label}_umap_img_compare.png", dpi=300)
        plt.close()

        # t-SNE
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"{label} - t-SNE 图片对比 (Oracle vs Egypt)")
        ax.set_xlabel("tsne_x")
        ax.set_ylabel("tsne_y")

        for _, row in sub.iterrows():
            x, y = row["tsne_x"], row["tsne_y"]
            img = preprocess_bw(row["file"], row["script"])
            img = img.resize((img_size, img_size), resample=Image.NEAREST)

            if row["script"] == "oracle":
                ax.scatter([x], [y], c="red", s=10, alpha=0.3)
            else:
                ax.scatter([x], [y], c="blue", s=10, alpha=0.3)

            ab = AnnotationBbox(OffsetImage(img), (x, y), frameon=False)
            ax.add_artist(ab)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{label}_tsne_img_compare.png", dpi=300)
        plt.close()


def parse_egypt_filename(fname):
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) == 3:
        ch, code, idx = parts
        try:
            idx = int(idx)
        except:
            idx = 0
        return ch, code, idx
    return stem[0], "NA", 0


def load_oracle():
    rows = []
    root = Path(ORACLE_ROOT)
    for d in root.iterdir():
        if not d.is_dir():
            continue
        label = d.name
        idx = 1
        for f in d.iterdir():
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                rows.append({
                    "file": str(f),
                    "label": label,
                    "script": "oracle",
                    "gardiner_code": "NA",
                    "index": idx
                })
                idx += 1
    return pd.DataFrame(rows)


def load_egypt():
    rows = []
    root = Path(EGYPT_ROOT)
    for d, _, files in os.walk(root):
        d = Path(d)
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = d / fname
                label, code, idx = parse_egypt_filename(fname)
                rows.append({
                    "file": str(fpath),
                    "label": label,
                    "script": "egypt",
                    "gardiner_code": code,
                    "index": idx
                })
    return pd.DataFrame(rows)


def compute_embeddings(df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    vecs = []
    for i, row in df.iterrows():
        img = preprocess_bw(row["file"], row["script"])
        t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            v = model.encode_image(t)
            v = v / v.norm(dim=-1, keepdim=True)
        vecs.append(v.cpu().numpy()[0])

        if i % 30 == 0:
            print("[INFO] Embedding:", i, "/", len(df))

    return np.vstack(vecs)


def run_umap(emb):
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    return reducer.fit_transform(emb)


def run_tsne(emb):
    return TSNE(
        n_components=2,
        metric="cosine",
        random_state=42
    ).fit(emb)


def word_comparison(df, label):
    sub = df[df["label"] == label]
    oracle = sub[sub["script"] == "oracle"]
    egypt = sub[sub["script"] == "egypt"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title(f"{label} - Oracle")
    axes[1].set_title(f"{label} - Egypt")

    for _, row in oracle.iterrows():
        img = preprocess_bw(row["file"], "oracle")
        axes[0].imshow(img, cmap="gray")
        axes[0].axis("off")

    for _, row in egypt.iterrows():
        img = preprocess_bw(row["file"], "egypt")
        axes[1].imshow(img, cmap="gray")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{label}_compare.png", dpi=300)
    plt.close()


def image_scatter(df, x, y, outpath, title):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(title)

    for _, row in df.iterrows():
        xi, yi = row[x], row[y]
        img = preprocess_bw(row["file"], row["script"])
        img = img.resize((32, 32), resample=Image.NEAREST)
        ab = AnnotationBbox(OffsetImage(img), (xi, yi), frameon=False)
        ax.add_artist(ab)

    ax.scatter(df[x], df[y], c="gray", alpha=0.01)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    plt.savefig(outpath, dpi=300)
    plt.close()


def make_image_grid(image_list, rows, cols, cell_size=128):
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), "white")
    for idx, img in enumerate(image_list):
        if idx >= rows * cols:
            break
        img = img.resize((cell_size, cell_size), Image.NEAREST)
        r = idx // cols
        c = idx % cols
        grid.paste(img, (c * cell_size, r * cell_size))
    return grid


def export_single_word_images(df, output_dir=OUTPUT_DIR, cell_size=128):
    labels = sorted(df["label"].unique())

    for label in labels:
        sub = df[df["label"] == label]
        oracle = sub[sub["script"] == "oracle"]
        egypt = sub[sub["script"] == "egypt"]

        if len(oracle) > 0:
            imgs = []
            for _, row in oracle.iterrows():
                imgs.append(preprocess_bw(row["file"], "oracle"))
            n = len(imgs)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            grid = make_image_grid(imgs, rows, cols, cell_size)
            grid.save(f"{output_dir}/{label}_oracle.png")

        if len(egypt) > 0:
            imgs = []
            for _, row in egypt.iterrows():
                imgs.append(preprocess_bw(row["file"], "egypt"))
            n = len(imgs)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            grid = make_image_grid(imgs, rows, cols, cell_size)
            grid.save(f"{output_dir}/{label}_egypt.png")


def compute_shape_features_from_array(gray_arr):
    h, w = gray_arr.shape
    total_pixels = h * w

    binary = (gray_arr < 128).astype(np.uint8)

    stroke_pixels = binary.sum()
    density = stroke_pixels / total_pixels if total_pixels > 0 else 0.0

    mid_w = w // 2
    left = binary[:, :mid_w]
    right = binary[:, -mid_w:]
    right_flipped = np.fliplr(right)
    if left.shape[1] != right_flipped.shape[1]:
        min_w = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :min_w]
        right_flipped = right_flipped[:, :min_w]
    vsym = 1.0 - np.mean(np.abs(left - right_flipped))

    mid_h = h // 2
    up = binary[:mid_h, :]
    down = binary[-mid_h:, :]
    down_flipped = np.flipud(down)
    if up.shape[0] != down_flipped.shape[0]:
        min_h = min(up.shape[0], down_flipped.shape[0])
        up = up[:min_h, :]
        down_flipped = down_flipped[:min_h, :]
    hsym = 1.0 - np.mean(np.abs(up - down_flipped))

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


def build_shape_feature_table(df):
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
            norm_group[col] = 0.5

    return group, norm_group


def plot_radar_for_chars(norm_group, target_labels=None, output_dir=OUTPUT_DIR):
    if target_labels is None:
        target_labels = ["日", "月", "星", "人", "帝", "鬼"]

    feat_cols = [
        "stroke_density",
        "vertical_symmetry",
        "horizontal_symmetry",
        "centralization",
        "component_count",
    ]
    feat_names_cn = ["笔画密度", "竖直对称", "水平对称", "中心集中度", "连通块多样度"]

    num_feats = len(feat_cols)
    angles = np.linspace(0, 2 * np.pi, num_feats, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for label in target_labels:
        sub = norm_group[norm_group["label"] == label]
        if sub.empty:
            print(f"[INFO] 字 {label} 没有数据，跳过雷达图。")
            continue

        has_oracle = (sub["script"] == "oracle").any()
        has_egypt = (sub["script"] == "egypt").any()
        if not has_oracle and not has_egypt:
            continue

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.set_title(f"{label} - 结构特征雷达图", fontsize=14)

        def get_values(script_name):
            row = sub[sub["script"] == script_name]
            if row.empty:
                return None
            vals = [row.iloc[0][col] for col in feat_cols]
            return np.array(vals + [vals[0]])

        oracle_vals = get_values("oracle")
        if oracle_vals is not None:
            ax.plot(angles, oracle_vals, "r-", label="甲骨文")
            ax.fill(angles, oracle_vals, "r", alpha=0.15)

        egypt_vals = get_values("egypt")
        if egypt_vals is not None:
            ax.plot(angles, egypt_vals, "b-", label="圣书体")
            ax.fill(angles, egypt_vals, "b", alpha=0.15)

        ax.set_xticks(np.linspace(0, 2 * np.pi, num_feats, endpoint=False))
        ax.set_xticklabels(feat_names_cn, fontsize=10)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)

        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{label}_radar.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"[INFO] 已生成雷达图: {out_path}")


def plot_single_word_tsne_umap(df, output_dir=OUTPUT_DIR):
    labels = sorted(df["label"].unique())

    for label in labels:
        sub = df[df["label"] == label]
        if len(sub) < 2:
            print(f"[INFO] 字 {label} 样本不足，跳过。")
            continue

        print(f"[INFO] 生成 {label} 的 UMAP / t-SNE 对比图")

        colors = sub["script"].map({"oracle": "red", "egypt": "blue"})

        plt.figure(figsize=(6, 6))
        plt.scatter(sub["umap_x"], sub["umap_y"], c=colors, s=40, alpha=0.8)
        plt.title(f"{label} - UMAP: Oracle (red) vs Egypt (blue)")
        plt.xlabel("umap_x")
        plt.ylabel("umap_y")
        plt.savefig(f"{output_dir}/{label}_umap_compare.png", dpi=300)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.scatter(sub["tsne_x"], sub["tsne_y"], c=colors, s=40, alpha=0.8)
        plt.title(f"{label} - t-SNE: Oracle (red) vs Egypt (blue)")
        plt.xlabel("tsne_x")
        plt.ylabel("tsne_y")
        plt.savefig(f"{output_dir}/{label}_tsne_compare.png", dpi=300)
        plt.close()


def main():
    df = pd.concat([load_oracle(), load_egypt()], ignore_index=True)
    print("[INFO] 加载完毕：", len(df), "张图像")

    emb = compute_embeddings(df)
    df["emb"] = list(emb)

    um = run_umap(emb)
    df["umap_x"], df["umap_y"] = um[:, 0], um[:, 1]

    ts = run_tsne(emb)
    df["tsne_x"], df["tsne_y"] = ts[:, 0], ts[:, 1]

    image_scatter(df[df["script"] == "oracle"], "umap_x", "umap_y",
                  f"{OUTPUT_DIR}/umap_oracle_img.png", "UMAP Oracle")

    image_scatter(df[df["script"] == "egypt"], "umap_x", "umap_y",
                  f"{OUTPUT_DIR}/umap_egypt_img.png", "UMAP Egypt")

    image_scatter(df, "umap_x", "umap_y",
                  f"{OUTPUT_DIR}/umap_both_img.png", "UMAP Both")

    image_scatter(df[df["script"] == "oracle"], "tsne_x", "tsne_y",
                  f"{OUTPUT_DIR}/tsne_oracle_img.png", "t-SNE Oracle")

    image_scatter(df[df["script"] == "egypt"], "tsne_x", "tsne_y",
                  f"{OUTPUT_DIR}/tsne_egypt_img.png", "t-SNE Egypt")

    image_scatter(df, "tsne_x", "tsne_y",
                  f"{OUTPUT_DIR}/tsne_both_img.png", "t-SNE Both")

    print("[INFO] 生成每个字的 UMAP / t-SNE 图片散点 …")
    plot_single_word_tsne_umap_images(df)

    print("[INFO] 计算形状特征并生成雷达图 …")
    group, norm_group = build_shape_feature_table(df)
    plot_radar_for_chars(norm_group, target_labels=["日", "月", "星", "人", "帝", "鬼"])


if __name__ == "__main__":
    main()
