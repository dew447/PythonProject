# radar_module.py
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel
from skimage.measure import shannon_entropy
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def compute_visual_complexity(img_array):
    gx = sobel(img_array, axis=1)
    gy = sobel(img_array, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    return float(magnitude.mean() / 255.0)


def compute_symmetry_score(img_array):
    h, w = img_array.shape
    left = img_array[:, :w // 2]
    right = np.fliplr(img_array[:, w // 2:])
    right = right[:, :left.shape[1]]
    diff = np.abs(left - right).mean()
    return float(1 - diff / 255.0)


def compute_entropy(img_array):
    return float(shannon_entropy(img_array))


def compute_contour_complexity(img_array):
    _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return 0.0

    total_length = sum(len(c) for c in contours)
    h, w = img_array.shape
    norm = (h + w)
    return float(total_length / norm)


def semantic_score(clip_model, device, emb_image, concept="symbol"):
    text = clip.tokenize(concept).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return float(cosine_similarity([emb_image], text_emb.cpu().numpy())[0][0])


def cluster_tightness(df, idx):
    x, y = df["umap_x"][idx], df["umap_y"][idx]
    d = np.sqrt((df["umap_x"] - x) ** 2 + (df["umap_y"] - y) ** 2)
    return float(np.exp(-d.mean()))


def plot_radar(label, features, save_path):
    labels = [
        "象形度",
        "聚类紧密度",
        "结构相似度",
        "语义匹配度",
        "模型一致度",
        "独特性",
        "轮廓复杂度",
        "图像熵",
        "视觉复杂度",
    ]

    values = features[:]
    N = len(values)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_title(f"Glyph Radar: {label}", fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_radar_for_all(df, embeddings, output_dir="outputs/radar"):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    results = []

    print("[INFO] 开始生成雷达图...")

    for idx, row in df.iterrows():
        img = Image.open(row["file"]).convert("L").resize((128, 128))
        img_array = np.array(img)

        picto = compute_symmetry_score(img_array)
        tight = cluster_tightness(df, idx)
        structure = float(cosine_similarity([embeddings[idx]], embeddings).mean())
        semantic = semantic_score(clip_model, device, embeddings[idx])
        consistency = float(np.std(embeddings[idx]))
        uniqueness = float(1 - structure)
        contour = compute_contour_complexity(img_array)
        entropy = compute_entropy(img_array)
        visual = compute_visual_complexity(img_array)

        features = [
            picto,
            tight,
            structure,
            semantic,
            consistency,
            uniqueness,
            contour,
            entropy,
            visual,
        ]

        stem = Path(row["file"]).stem
        folder_idx = int(stem) if stem.isdigit() else idx % 1000
        global_idx = idx

        short_name = f"{row['label']}_{folder_idx:03d}_{global_idx:04d}.png"
        save_path = os.path.join(output_dir, short_name)

        plot_radar(row["label"], features, save_path)
        print(f"[OK] Radar saved: {save_path}")

        results.append([
            row["file"],
            row["label"],
            *features,
        ])

    columns = [
        "file",
        "label",
        "象形度",
        "聚类紧密度",
        "结构相似度",
        "语义匹配度",
        "模型一致度",
        "独特性",
        "轮廓复杂度",
        "图像熵",
        "视觉复杂度",
    ]

    df_out = pd.DataFrame(results, columns=columns)
    df_out.to_csv(
        os.path.join(output_dir, "glyph_radar_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("[DONE] 所有雷达图已生成，输出目录:", output_dir)
