import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import torch
import clip
import umap
from openTSNE import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


ROOT_DIR = "oracle_selected_glyphs"
EMB_PATH = "embeddings.npy"
CSV_PATH = "dataset.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_images(root_dir):
    rows = []
    root = Path(root_dir)
    for d, _, files in os.walk(root):
        d = Path(d)
        label = d.name
        if label == root.name:
            continue
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append({
                    "file": str(d / f),
                    "label": label
                })
    return pd.DataFrame(rows)


def compute_embeddings(df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    vecs = []
    for i, row in df.iterrows():
        img = Image.open(row["file"]).convert("RGB")
        t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            v = model.encode_image(t)
            v = v / v.norm(dim=-1, keepdim=True)
        vecs.append(v.cpu().numpy()[0])

        if i % 10 == 0:
            print(f"[INFO] 已处理 {i}/{len(df)}")

    return np.vstack(vecs)


def run_umap(emb):
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42
    )
    return reducer.fit_transform(emb)


def run_tsne(emb):
    return TSNE(
        n_components=2,
        metric="cosine",
        n_jobs=8,
        random_state=42
    ).fit(emb)


def plot_image_scatter(df, x, y, outpath):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(outpath)

    for i in range(len(df)):
        xi, yi = df[x][i], df[y][i]
        img = Image.open(df["file"][i]).convert("RGB").resize((32, 32))
        ab = AnnotationBbox(OffsetImage(img), (xi, yi), frameon=False)
        ax.add_artist(ab)

    ax.scatter(df[x], df[y], alpha=0.01)
    plt.savefig(outpath, dpi=300)
    plt.close()


def interactive_plot(df, x, y, outpath):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="label",
        hover_data=["file"],
        title="Interactive Plot"
    )
    fig.write_html(outpath)
    print("[INFO] 交互图已输出:", outpath)


def query_similar(emb, df, idx, topk=5):
    sims = cosine_similarity([emb[idx]], emb)[0]
    best = sims.argsort()[::-1][1:topk + 1]

    print("\n========= 相似字查询 =========")
    print("查询图片:", df["file"][idx])
    for b in best:
        print(df["file"][b], "   相似度:", sims[b])


def main():
    df = load_images(ROOT_DIR)
    print("[INFO] 找到图像:", len(df))

    emb = compute_embeddings(df)
    np.save(EMB_PATH, emb)
    df.to_csv(CSV_PATH, index=False)

    um = run_umap(emb)
    df["umap_x"] = um[:, 0]
    df["umap_y"] = um[:, 1]
    df.to_csv("umap.csv", index=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(df["umap_x"], df["umap_y"], c="blue")
    plt.title("UMAP")
    plt.savefig(f"{OUTPUT_DIR}/umap.png", dpi=300)
    plt.close()

    plot_image_scatter(df, "umap_x", "umap_y", f"{OUTPUT_DIR}/umap_image_scatter.png")
    interactive_plot(df, "umap_x", "umap_y", f"{OUTPUT_DIR}/interactive_umap.html")

    ts = run_tsne(emb)
    df["tsne_x"] = ts[:, 0]
    df["tsne_y"] = ts[:, 1]
    df.to_csv("tsne.csv", index=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(df["tsne_x"], df["tsne_y"], c="green")
    plt.title("t-SNE")
    plt.savefig(f"{OUTPUT_DIR}/tsne.png", dpi=300)
    plt.close()

    plot_image_scatter(df, "tsne_x", "tsne_y", f"{OUTPUT_DIR}/tsne_image_scatter.png")

    print("\n=== 查询第 0 个图片最相似的 5 个 ===")
    query_similar(emb, df, idx=0, topk=5)

    print("[DONE] 输出已生成。")


if __name__ == "__main__":
    main()
