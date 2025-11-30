import os
import glob
import pandas as pd
from PIL import Image
import torch
import clip
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ------------------------
# 1. 载入 CLIP
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ------------------------
# 2. 扫描两个数据集
# ------------------------
def load_images_from_folder(root, script_name):
    rows = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                fp = os.path.join(dirpath, f)

                label = os.path.basename(dirpath)  # 使用文件夹名作为类别
                rows.append({
                    "file_path": fp,
                    "label": label,
                    "script": script_name
                })
    return rows

egypt_rows = load_images_from_folder("egypt_selected_glyphs", "egypt")
oracle_rows = load_images_from_folder("oracle_selected_glyphs", "oracle")

df = pd.DataFrame(egypt_rows + oracle_rows)
df.to_csv("combined_dataset.csv", index=False, encoding="utf-8-sig")
print(f"[INFO] 共加载 {len(df)} 张图像")

# ------------------------
# 3. CLIP embedding
# ------------------------
embeddings = []

for idx, row in df.iterrows():
    img = Image.open(row["file_path"]).convert("RGB")
    img_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(img_input)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.cpu().numpy()[0])

embeddings = np.array(embeddings)

# ------------------------
# 4. t-SNE 降维
# ------------------------
print("[INFO] 正在执行 t-SNE ...")
tsne = TSNE(
    n_components=2,
    perplexity=20,
    learning_rate=200,
    init='random',
    random_state=42
)

coords = tsne.fit_transform(embeddings)
df["tsne_x"] = coords[:,0]
df["tsne_y"] = coords[:,1]

df.to_csv("combined_with_tsne.csv", index=False, encoding="utf-8-sig")

# ------------------------
# 5. 绘制散点图
# ------------------------
plt.figure(figsize=(10,10))

for script in ["egypt", "oracle"]:
    _df = df[df["script"] == script]
    plt.scatter(_df["tsne_x"], _df["tsne_y"], label=script, alpha=0.6)

plt.legend()
plt.title("Egypt vs Oracle Bone — CLIP+t-SNE")
plt.savefig("embedding_tsne.png", dpi=300)
plt.show()

print("[INFO] embedding_tsne.png 已生成")
