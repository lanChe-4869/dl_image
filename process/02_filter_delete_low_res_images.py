import os
import shutil
from pathlib import Path

# 主目录
source_dir = Path(r"images\single\20260812_single\images") # ./images/20260615

# 输出目录
output_dir = Path(r"images\single\20260812_single\dirty_1")
output_dir.mkdir(parents=True, exist_ok=True)

# 500KB
size_threshold = 500 * 1024

image_extensions = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".gif",
    ".tif", ".tiff"
}

moved_count = 0

for file_path in source_dir.rglob("*"):

    if not file_path.is_file():
        continue

    if file_path.suffix.lower() not in image_extensions:
        continue

    try:
        if file_path.stat().st_size >= size_threshold:
            continue

        # 获取直属父目录名称
        parent_folder = file_path.parent.name

        # 从 "12345_苹果手机" 提取 id=12345
        image_id = parent_folder.split("_")[0]

        # 新文件名
        new_name = f"{image_id}_{file_path.name}"

        target_path = output_dir / new_name

        shutil.move(str(file_path), str(target_path))

        moved_count += 1
        print(f"移动: {file_path} -> {target_path}")

    except Exception as e:
        print(f"处理失败: {file_path}")
        print(e)

print(f"\n完成，共移动 {moved_count} 张图片")
