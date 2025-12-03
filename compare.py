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
import matplotlib.cm as cm


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ================= 配置 =================
ORACLE_ROOT = "oracle_selected_glyphs"   # 甲骨文字形目录
EGYPT_ROOT  = "egypt_by_oracle"          # 圣书体字形目录

EMB_PATH = "embeddings_all.npy"          # 保存所有向量
CSV_PATH = "dataset_all.csv"             # 保存所有样本元信息

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 工具：解析圣书体文件名 =================
def parse_egypt_filename(fname: str):
    """
    期望文件名格式: 鬼_G29_001.png
    返回: (汉字, Gardiner code, index)
    若格式不符合约定，则退化为:
        char = 第一个字符, code = 'NA', index = 0
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) == 3:
        ch, code, idx = parts
        try:
            idx = int(idx)
        except ValueError:
            idx = 0
        return ch, code, idx
    else:
        ch = stem[0]
        return ch, "NA", 0


# ================= 图像黑白预处理 =================
def preprocess_bw_for_script(fpath, script):
    """
    针对脚本类型进行统一的黑白化处理，返回 PIL.Image(RGB)：
      - 圣书体 egypt: 使用 Otsu 做强二值化
      - 甲骨文 oracle: 使用较高固定阈值做轻度二值化
    """
    pil_img = Image.open(fpath).convert("RGB")
    gray = np.array(pil_img.convert("L"))

    if script == "egypt":
        # Otsu 自动阈值
        _, bw = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        # 甲骨文一般已接近黑白，此处做轻量阈值
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 保证背景为白、笔画为黑
    if bw.mean() < 127:
        bw = 255 - bw

    return Image.fromarray(bw).convert("RGB")


# ================= 读取甲骨文 =================
def load_oracle_images(root_dir):
    """
    目录结构假设：
        oracle_selected_glyphs/
            占/
                xxx.png
                yyy.png
            卜/
                ...
    label 取中文子目录名，
    gardiner_code 统一设为 "NA"。
    """
    rows = []
    root = Path(root_dir)

    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir():
            continue
        label = char_dir.name
        idx = 1
        for f in sorted(char_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rows.append({
                    "file": str(f),
                    "label": label,
                    "gardiner_code": "NA",
                    "index": idx,
                    "script": "oracle"
                })
                idx += 1

    return pd.DataFrame(rows)


# ================= 读取圣书体 =================
def load_egypt_images(root_dir):
    """
    圣书体目录结构示例：
        egypt_by_oracle/
            鬼/
                G29/
                    鬼_G29_001.png
                G25/
                    鬼_G25_001.png
            占/
                ...
    """
    rows = []
    root = Path(root_dir)

    for d, _, files in os.walk(root):
        d = Path(d)
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = d / fname
                ch, code, idx = parse_egypt_filename(fname)
                rows.append({
                    "file": str(fpath),
                    "label": ch,
                    "gardiner_code": code,
                    "index": idx,
                    "script": "egypt"
                })

    return pd.DataFrame(rows)


# ================= CLIP Embedding =================
def compute_embeddings(df):
    """
    所有图片在送入 CLIP 前先经过 preprocess_bw_for_script，进行黑白风格统一。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    vecs = []
    for i, row in df.iterrows():
        fpath = row["file"]
        script = row["script"]

        img = preprocess_bw_for_script(fpath, script)
        t = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            v = model.encode_image(t)
            v = v / v.norm(dim=-1, keepdim=True)
        vecs.append(v.cpu().numpy()[0])

        if i % 10 == 0:
            print(f"[INFO] 已处理 {i}/{len(df)}")

    return np.vstack(vecs)


# ================= UMAP & t-SNE =================
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


# ================= 简单散点图（无图片叠加） =================
def simple_scatter(df, x, y, outpath, title, color_col="label"):
    """
    绘制简单散点图，不叠加原始图像。
    点颜色由 color_col 控制（默认按 label 区分）。
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)

    values = df[color_col].astype(str).unique()

    cmap = plt.colormaps.get("tab20")
    color_map = {v: cmap(i % cmap.N) for i, v in enumerate(values)}
    colors = df[color_col].astype(str).map(color_map)

    ax.scatter(df[x], df[y], c=colors, s=10, alpha=0.8)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # 只显示前若干条 legend，避免过度拥挤
    handles = []
    labels = []
    for v in values[:15]:
        handles.append(plt.Line2D([], [], marker="o", linestyle="",
                                  color=color_map[v], label=v))
        labels.append(v)
    if handles:
        ax.legend(handles, labels, title=color_col, bbox_to_anchor=(1.05, 1),
                  loc="upper left")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def generate_global_scatter_plots(df):
    """
    生成以下散点图（UMAP / t-SNE）：
      1) 仅甲骨文
      2) 仅圣书体
      3) 甲骨文 + 圣书体
    颜色按 label 区分。
    """
    df_oracle = df[df["script"] == "oracle"]
    df_egypt  = df[df["script"] == "egypt"]

    # ---- t-SNE ----
    if len(df_oracle) > 0:
        simple_scatter(
            df_oracle,
            "tsne_x", "tsne_y",
            outpath=f"{OUTPUT_DIR}/tsne_oracle_all_chars.png",
            title="t-SNE - 甲骨文（按字着色）",
            color_col="label"
        )
    if len(df_egypt) > 0:
        simple_scatter(
            df_egypt,
            "tsne_x", "tsne_y",
            outpath=f"{OUTPUT_DIR}/tsne_egypt_all_chars.png",
            title="t-SNE - 圣书体（按字着色）",
            color_col="label"
        )

    simple_scatter(
        df,
        "tsne_x", "tsne_y",
        outpath=f"{OUTPUT_DIR}/tsne_both_all_chars.png",
        title="t-SNE - 甲骨文 + 圣书体（按字着色）",
        color_col="label"
    )

    # ---- UMAP ----
    if "umap_x" in df.columns:
        if len(df_oracle) > 0:
            simple_scatter(
                df_oracle,
                "umap_x", "umap_y",
                outpath=f"{OUTPUT_DIR}/umap_oracle_all_chars.png",
                title="UMAP - 甲骨文（按字着色）",
                color_col="label"
            )
        if len(df_egypt) > 0:
            simple_scatter(
                df_egypt,
                "umap_x", "umap_y",
                outpath=f"{OUTPUT_DIR}/umap_egypt_all_chars.png",
                title="UMAP - 圣书体（按字着色）",
                color_col="label"
            )

        simple_scatter(
            df,
            "umap_x", "umap_y",
            outpath=f"{OUTPUT_DIR}/umap_both_all_chars.png",
            title="UMAP - 甲骨文 + 圣书体（按字着色）",
            color_col="label"
        )


# ================= 甲骨文 vs 圣书体 相似度分析 =================
def compare_oracle_egypt(emb, df, out_csv):
    """
    对每个 label 进行比较：
      - 计算 oracle 向量的 centroid
      - 计算 egypt 下整体 centroid 以及各 Gardiner code 的 centroid
      - 计算上述 centroid 与 oracle centroid 的余弦相似度
    """
    rows = []

    for label in sorted(df["label"].unique()):
        oracle_idx = df[(df["label"] == label) & (df["script"] == "oracle")].index
        egypt_idx  = df[(df["label"] == label) & (df["script"] == "egypt")].index

        if len(oracle_idx) == 0 or len(egypt_idx) == 0:
            continue

        oracle_centroid = emb[oracle_idx].mean(axis=0)

        # 整体圣书体 centroid
        egypt_centroid_all = emb[egypt_idx].mean(axis=0)
        sim_all = cosine_similarity(
            [oracle_centroid],
            [egypt_centroid_all]
        )[0, 0]

        rows.append({
            "label": label,
            "gardiner_code": "ALL",
            "n_oracle": len(oracle_idx),
            "n_egypt": len(egypt_idx),
            "cosine_similarity": sim_all,
            "cosine_distance": 1 - sim_all
        })

        # 按 Gardiner code 细分
        for code, sub in df[(df["label"] == label) & (df["script"] == "egypt")].groupby("gardiner_code"):
            idxs = sub.index
            egypt_centroid_code = emb[idxs].mean(axis=0)
            sim_code = cosine_similarity(
                [oracle_centroid],
                [egypt_centroid_code]
            )[0, 0]

            rows.append({
                "label": label,
                "gardiner_code": code,
                "n_oracle": len(oracle_idx),
                "n_egypt": len(idxs),
                "cosine_similarity": sim_code,
                "cosine_distance": 1 - sim_code
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[INFO] 甲骨文 vs 圣书体 相似度结果已写入:", out_csv)


# ================= 主流程 =================
def main():
    # 1. 载入两套数据
    df_oracle = load_oracle_images(ORACLE_ROOT)
    df_egypt  = load_egypt_images(EGYPT_ROOT)

    print("[INFO] 甲骨文图片数量:", len(df_oracle))
    print("[INFO] 圣书体图片数量:", len(df_egypt))

    df = pd.concat([df_oracle, df_egypt], ignore_index=True)
    print("[INFO] 总图片数量:", len(df))

    # 2. 计算 embedding
    emb = compute_embeddings(df)
    np.save(EMB_PATH, emb)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print("[INFO] 向量与数据集元信息已保存")

    # 3. UMAP & t-SNE
    print("[INFO] 运行 UMAP ...")
    um = run_umap(emb)
    df["umap_x"] = um[:, 0]
    df["umap_y"] = um[:, 1]
    df.to_csv("umap_all.csv", index=False, encoding="utf-8-sig")

    print("[INFO] 运行 t-SNE ...")
    ts = run_tsne(emb)
    df["tsne_x"] = ts[:, 0]
    df["tsne_y"] = ts[:, 1]
    df.to_csv("tsne_all.csv", index=False, encoding="utf-8-sig")

    # 4. 全局散点图（oracle-only / egypt-only / both）
    print("[INFO] 生成 UMAP / t-SNE 散点图 ...")
    generate_global_scatter_plots(df)

    # 5. 甲骨文 vs 圣书体 相似度分析
    compare_oracle_egypt(
        emb,
        df,
        out_csv=os.path.join(OUTPUT_DIR, "oracle_vs_egypt_similarity.csv")
    )

    print("[DONE] 处理完成，图像与数据已输出至 outputs/ 目录。")


if __name__ == "__main__":
    main()
