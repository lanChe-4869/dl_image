import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def build_name_index(root: Path) -> Dict[str, List[Path]]:
    """
    按文件名建立索引。
    key 使用 stem，也就是不带扩展名的文件名，方便 dirty 是 jpg、debug 是 jpg、origin 是 png 时也能匹配。
    """
    index: Dict[str, List[Path]] = {}
    if not root.exists():
        return index

    for p in root.rglob("*"):
        if not is_image(p):
            continue
        index.setdefault(p.stem, []).append(p)

    return index


def choose_best_match(
    dirty_path: Path,
    candidates: List[Path],
    source_root: Path,
    dirty_root: Path,
) -> Optional[Path]:
    """
    优先匹配相对路径，其次匹配同名文件。
    """
    if not candidates:
        return None

    dirty_rel = dirty_path.relative_to(dirty_root)
    expected_same_rel = (source_root / dirty_rel)

    # 1. 完全相对路径一致
    if expected_same_rel.exists():
        return expected_same_rel

    # 2. stem 相同，但扩展名不同，且相对父目录一致
    expected_parent = expected_same_rel.parent
    same_parent = [p for p in candidates if p.parent == expected_parent]
    if same_parent:
        return same_parent[0]

    # 3. 只按文件名匹配
    return candidates[0]


def copy_unique(src: Path, dst: Path) -> Path:
    """
    复制文件，若目标已存在，自动追加 _1、_2，避免覆盖。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    final_dst = dst
    if final_dst.exists():
        stem = final_dst.stem
        suffix = final_dst.suffix
        parent = final_dst.parent
        idx = 1
        while final_dst.exists():
            final_dst = parent / f"{stem}_{idx}{suffix}"
            idx += 1

    shutil.copy2(src, final_dst)
    return final_dst


def collect_dirty_origin_debug(
    dirty_dir: Path,
    origin_dir: Path,
    debug_dir: Path,
    output_dir: Path,
    flatten: bool = False,
):
    dirty_dir = dirty_dir.resolve()
    origin_dir = origin_dir.resolve()
    debug_dir = debug_dir.resolve()
    output_dir = output_dir.resolve()

    output_origin_dir = output_dir / "origin"
    output_debug_dir = output_dir / "debug"
    output_dirty_dir = output_dir / "dirty"

    origin_index = build_name_index(origin_dir)
    debug_index = build_name_index(debug_dir)

    dirty_images = [p for p in dirty_dir.rglob("*") if is_image(p)]

    stats = {
        "dirty_total": 0,
        "dirty_copied": 0,
        "origin_found": 0,
        "origin_missing": 0,
        "debug_found": 0,
        "debug_missing": 0,
    }

    missing_origin = []
    missing_debug = []

    for dirty_path in dirty_images:
        stats["dirty_total"] += 1
        dirty_rel = dirty_path.relative_to(dirty_dir)

        if flatten:
            dirty_dst_rel = dirty_path.name
        else:
            dirty_dst_rel = dirty_rel

        # 也复制 dirty 本身，方便三者对照
        copy_unique(dirty_path, output_dirty_dir / dirty_dst_rel)
        stats["dirty_copied"] += 1

        origin_match = choose_best_match(
            dirty_path=dirty_path,
            candidates=origin_index.get(dirty_path.stem, []),
            source_root=origin_dir,
            dirty_root=dirty_dir,
        )

        debug_match = choose_best_match(
            dirty_path=dirty_path,
            candidates=debug_index.get(dirty_path.stem, []),
            source_root=debug_dir,
            dirty_root=dirty_dir,
        )

        if origin_match is not None:
            dst_name = origin_match.name if flatten else dirty_rel.with_suffix(origin_match.suffix)
            copy_unique(origin_match, output_origin_dir / dst_name)
            stats["origin_found"] += 1
        else:
            missing_origin.append(str(dirty_rel).replace("\\", "/"))
            stats["origin_missing"] += 1

        if debug_match is not None:
            dst_name = debug_match.name if flatten else dirty_rel.with_suffix(debug_match.suffix)
            copy_unique(debug_match, output_debug_dir / dst_name)
            stats["debug_found"] += 1
        else:
            missing_debug.append(str(dirty_rel).replace("\\", "/"))
            stats["debug_missing"] += 1

    # 保存缺失清单
    output_dir.mkdir(parents=True, exist_ok=True)
    if missing_origin:
        (output_dir / "missing_origin.txt").write_text("\n".join(missing_origin), encoding="utf-8")
    if missing_debug:
        (output_dir / "missing_debug.txt").write_text("\n".join(missing_debug), encoding="utf-8")

    print("\nDone.")
    print(f"Dirty copied:  {output_dirty_dir}")
    print(f"Origin copied: {output_origin_dir}")
    print(f"Debug copied:  {output_debug_dir}")
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if missing_origin:
        print(f"\nMissing origin list: {output_dir / 'missing_origin.txt'}")
    if missing_debug:
        print(f"Missing debug list:  {output_dir / 'missing_debug.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="根据 dirty 文件夹里的图片名，从 origin 和 debug 文件夹中复制对应原图/检测图到 dirty_origin。"
    )
    parser.add_argument("--dirty_dir", type=str, required=True, help="dirty 图片文件夹")
    parser.add_argument("--origin_dir", type=str, required=True, help="原图 origin 文件夹")
    parser.add_argument("--debug_dir", type=str, required=True, help="检测 debug 图文件夹")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，例如 ./dirty_origin，会创建 dirty/origin/debug 三个子目录",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="是否拍平输出，不保留相对目录结构。默认保留相对目录结构。",
    )

    args = parser.parse_args()

    collect_dirty_origin_debug(
        dirty_dir=Path(args.dirty_dir),
        origin_dir=Path(args.origin_dir),
        debug_dir=Path(args.debug_dir),
        output_dir=Path(args.output_dir),
        flatten=args.flatten,
    )


if __name__ == "__main__":
    main()


"""
PowerShell example:

python ./process/06_collect_dirty_origin_debug.py `
  --dirty_dir images/process_image/20260615_output/equal_2/dirty `
  --origin_dir images/20260615_output/equal_2 `
  --debug_dir images/process_image/20260615_output/equal_2/debug/process_ok `
  --output_dir ./images/process_image/20260615_output/equal_2/dirty_origin

输出结构：

dirty_origin/
  dirty/    # dirty 中的裁剪图
  origin/   # 从 origin 找到的原图
  debug/    # 从 debug 找到的检测图
  missing_origin.txt
  missing_debug.txt
"""
