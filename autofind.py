import os

# 13 个目标字
FOLDERS = [
    "日", "月", "星",
    "帝", "示", "祖", "鬼",
    "祀", "祭", "卜", "占",
    "史", "祝", "巫"
]

ROOT = "oracle_selected_glyphs"

def main():
    if not os.path.exists(ROOT):
        os.makedirs(ROOT)

    for name in FOLDERS:
        path = os.path.join(ROOT, name)
        os.makedirs(path, exist_ok=True)
        print(f"[OK] 创建文件夹: {path}")

    print("\n全部文件夹创建完成！")

if __name__ == "__main__":
    main()
