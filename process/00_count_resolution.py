from pathlib import Path
from PIL import Image
from collections import Counter
from math import gcd

# ===== 修改这里：你的主文件夹路径 =====
ROOT_DIR = r"./images\20260616_output\equal_2"

# 支持的图片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

resolution_counter = Counter()
aspect_ratio_counter = Counter()
total_images = 0
bad_files = []


def get_resolution_bucket(width, height):
    """
    按图片长边统计分辨率：
    <1K, 1K~2K, 2K~3K ...
    """
    long_side = max(width, height)

    if long_side < 1000:
        return "<1K"

    lower = long_side // 1000
    upper = lower + 1
    return f"{lower}K~{upper}K"


def get_aspect_ratio(width, height):
    """
    统计最简长宽比，例如：
    1920x1080 -> 16:9
    1024x768 -> 4:3
    """
    g = gcd(width, height)
    return f"{width // g}:{height // g}"


root = Path(ROOT_DIR)

for file in root.rglob("*"):
    if file.suffix.lower() not in IMAGE_EXTS:
        continue

    try:
        with Image.open(file) as img:
            width, height = img.size

        total_images += 1

        resolution_bucket = get_resolution_bucket(width, height)
        aspect_ratio = get_aspect_ratio(width, height)

        resolution_counter[resolution_bucket] += 1
        aspect_ratio_counter[aspect_ratio] += 1

    except Exception as e:
        bad_files.append((str(file), str(e)))


print(f"图片总数：{total_images}")

print("\n分辨率统计：")
for k, v in sorted(resolution_counter.items()):
    print(f"{k}: {v} 张")

print("\n长宽比统计：")
for k, v in aspect_ratio_counter.most_common():
    print(f"{k}: {v} 张")

if bad_files:
    print("\n无法读取的文件：")
    for path, err in bad_files:
        print(path, err)
