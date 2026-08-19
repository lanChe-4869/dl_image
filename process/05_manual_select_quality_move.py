"""
手动筛选优质图像：从 source_dir 中查看小窗预览，把优质图像移动到 output_dir。

操作：
- 按 1 / Enter：选择为优质图像，并移动到 output_dir
- 按 2 / Space：跳过当前图像，不移动
- 按 B：撤销上一步移动的优质图像，移回原位置
- 按 Q / Esc：退出

示例：筛选 50 张优质图像后自动结束
python ./process/05_manual_select_quality_move.py `
  --source_dir E:\workspace\vsCode\download_video\images\process_image\20260616_output_1\crop `
  --output_dir E:\workspace\vsCode\download_video\images\process_image\20260703_crop_halfbody `
  --target_count 1500   

python ./process/05_manual_select_quality_move.py `
  --source_dir E:\workspace\vsCode\download_video\images\process_image\20260703_crop_halfbody `
  --output_dir E:\workspace\vsCode\download_video\images\process_image\20260703_crop_big `
  --target_count 1500

递归筛选子目录，并在 output_dir 中保留相对目录结构：
python ./process/05_manual_select_quality_move.py \
  --source_dir ./images/input_photos \
  --output_dir ./images/selected_quality \
  --target_count 50 \
  --recursive

注意：图像是“移动”到 output_dir，不是复制。
"""

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def cv_imread(file_path: Path):
    """支持中文路径的 OpenCV 读取。"""
    img_array = np.fromfile(str(file_path), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def get_resize_dim(img, max_width: int = 1200, max_height: int = 800) -> Tuple[int, int]:
    """按窗口最大尺寸等比例缩放；小图不放大。"""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale > 1:
        return w, h
    return max(1, int(w * scale)), max(1, int(h * scale))


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def list_images(source_dir: Path, output_dir: Path, recursive: bool = False) -> List[Path]:
    """列出待筛选图片，自动跳过 output_dir 里的图片。"""
    iterator: Iterable[Path] = source_dir.rglob("*") if recursive else source_dir.iterdir()
    images = []
    for p in iterator:
        if not is_image(p):
            continue
        if is_inside(p, output_dir):
            continue
        images.append(p)
    return sorted(images, key=lambda x: str(x).lower())


def unique_destination(dest_path: Path) -> Path:
    """目标文件已存在时自动加 _1、_2，避免覆盖。"""
    if not dest_path.exists():
        return dest_path

    parent = dest_path.parent
    stem = dest_path.stem
    suffix = dest_path.suffix
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def draw_overlay(img, text_lines: List[str]):
    """在图片左上角绘制操作提示，不修改原图。"""
    display = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    x, y = 20, 36
    line_h = 32

    for idx, text in enumerate(text_lines):
        yy = y + idx * line_h
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(display, (x - 8, yy - th - 8), (x + tw + 8, yy + 8), (255, 255, 255), -1)
        cv2.putText(display, text, (x, yy), font, font_scale, (0, 140, 0), thickness, cv2.LINE_AA)
    return display


def move_to_output(src_path: Path, source_dir: Path, output_dir: Path, keep_structure: bool) -> Path:
    """把选中的优质图像移动到 output_dir。"""
    if keep_structure:
        rel_path = src_path.relative_to(source_dir)
        dst_path = output_dir / rel_path
    else:
        dst_path = output_dir / src_path.name

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path = unique_destination(dst_path)
    shutil.move(str(src_path), str(dst_path))
    return dst_path


def restore_from_output(output_path: Path, original_path: Path) -> Path:
    """撤销上一步移动：从 output_dir 移回原位置。"""
    original_path.parent.mkdir(parents=True, exist_ok=True)
    restore_path = unique_destination(original_path)
    shutil.move(str(output_path), str(restore_path))
    return restore_path


def filter_quality_images(
    source_dir: str,
    output_dir: str,
    target_count: int = 50,
    recursive: bool = False,
    keep_structure: bool = True,
    max_width: int = 1200,
    max_height: int = 800,
):
    """手动筛选优质图像，移动到 output_dir，达到 target_count 后自动停止。"""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if target_count <= 0:
        raise ValueError(f"target_count 必须大于 0，当前值: {target_count}")

    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(source_dir, output_dir, recursive=recursive)
    if not images:
        print(f"没有找到待筛选图片: {source_dir}")
        return

    window_name = "Manual Quality Image Selector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("=== 优质图像手动筛选启动 ===")
    print(f"Source     : {source_dir}")
    print(f"Output     : {output_dir}")
    print(f"TargetCount: {target_count}")
    print("[1] 或 [Enter] : 选为优质图像，并移动到 output_dir")
    print("[2] 或 [Space] : 跳过当前图像，不移动")
    print("[B]            : 撤销上一步移动的优质图像")
    print("[Q] 或 [Esc]   : 退出")

    moved_stack: List[Tuple[Path, Path]] = []  # [(output_path, original_path)]
    selected_count = 0
    skipped_count = 0
    unreadable_count = 0

    i = 0
    total = len(images)
    while i < total:
        if selected_count >= target_count:
            print(f"已选满 {target_count} 张优质图像，自动结束筛选。")
            break

        img_path = images[i]

        # 图片可能已经被外部移动/删除，跳过。
        if not img_path.exists():
            i += 1
            continue

        img = cv_imread(img_path)
        if img is None:
            print(f"[WARN] 无法读取，跳过: {img_path}")
            unreadable_count += 1
            i += 1
            continue

        new_w, new_h = get_resize_dim(img, max_width=max_width, max_height=max_height)
        cv2.resizeWindow(window_name, new_w, new_h)

        rel_text = str(img_path.relative_to(source_dir)) if is_inside(img_path, source_dir) else img_path.name
        overlay = [
            f"{i + 1}/{total} | Selected {selected_count}/{target_count} | 1/Enter SELECT | 2/Space SKIP | B Undo | Q Quit",
            rel_text[:120],
        ]
        display = draw_overlay(img, overlay)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(0) & 0xFF

        if key in [ord("1"), 13]:
            dst_path = move_to_output(
                src_path=img_path,
                source_dir=source_dir,
                output_dir=output_dir,
                keep_structure=keep_structure,
            )
            moved_stack.append((dst_path, img_path))
            selected_count += 1
            print(f"[SELECT] {img_path} -> {dst_path} ({selected_count}/{target_count})")
            i += 1

        elif key in [ord("2"), 32]:
            skipped_count += 1
            print(f"[SKIP]   {img_path}")
            i += 1

        elif key in [ord("b"), ord("B")]:
            if not moved_stack:
                print("[UNDO] 没有可撤销的移动")
                continue
            output_path, original_path = moved_stack.pop()
            if output_path.exists():
                restored_path = restore_from_output(output_path, original_path)
                selected_count = max(0, selected_count - 1)
                print(f"[UNDO]  {output_path} -> {restored_path} ({selected_count}/{target_count})")
                # 回到上一张，方便重新判断。
                i = max(0, i - 1)
            else:
                print(f"[UNDO] output 文件不存在，无法撤销: {output_path}")

        elif key in [ord("q"), ord("Q"), 27]:
            print("收到退出指令，停止筛选。")
            break

        else:
            print(f"未识别按键: {key}，请按 1/2/B/Q")

    cv2.destroyAllWindows()
    print("\n处理结束")
    print(f"已移动优质图像数量: {selected_count}")
    print(f"跳过数量: {skipped_count}")
    print(f"无法读取数量: {unreadable_count}")
    print(f"目标数量: {target_count}")
    print(f"Output 文件夹: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="手动筛选优质图像：选中的图像会移动到 output_dir，不是复制")
    parser.add_argument("--source_dir", type=str, required=True, help="待筛选图片所在文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="优质图像移动到的目标文件夹")
    parser.add_argument("--target_count", type=int, default=50, help="移动多少张优质图像后自动结束，默认 50")
    parser.add_argument("--recursive", action="store_true", help="递归筛选 source_dir 下所有子目录图片，并在 output_dir 中保留相对目录结构")
    parser.add_argument("--flat_output", action="store_true", help="移动到 output_dir 时不保留子目录结构，全部平铺到 output_dir")
    parser.add_argument("--max_width", type=int, default=1200, help="预览窗口最大宽度")
    parser.add_argument("--max_height", type=int, default=800, help="预览窗口最大高度")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_quality_images(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_count=args.target_count,
        recursive=args.recursive,
        keep_structure=not args.flat_output,
        max_width=args.max_width,
        max_height=args.max_height,
    )
