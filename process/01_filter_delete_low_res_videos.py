import argparse
import json
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
    ".m4v", ".flv", ".wmv", ".ts", ".m2ts",
    ".mpg", ".mpeg"
}


def run_ffprobe(video_path: Path) -> dict | None:
    """
    使用 ffprobe 获取视频流信息。
    返回第一个 video stream 的信息。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )

        info = json.loads(result.stdout)
        streams = info.get("streams", [])

        if not streams:
            return None

        return streams[0]

    except Exception as e:
        print(f"[WARN] ffprobe 失败，跳过: {video_path}")
        print(f"       {e}")
        return None


def get_rotation(stream: dict) -> int:
    """
    读取视频旋转信息。
    有些手机竖屏视频实际存储为横屏尺寸，但带 rotate metadata。
    """
    rotation = 0

    tags = stream.get("tags", {})
    if "rotate" in tags:
        try:
            rotation = int(float(tags["rotate"]))
        except Exception:
            pass

    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                rotation = int(float(side_data["rotation"]))
            except Exception:
                pass

    return rotation % 360


def get_display_size(stream: dict) -> tuple[int, int] | None:
    """
    返回视频实际显示方向下的 width, height。
    如果 rotation 是 90 或 270，则交换宽高。
    """
    width = stream.get("width")
    height = stream.get("height")

    if width is None or height is None:
        return None

    width = int(width)
    height = int(height)

    rotation = get_rotation(stream)

    if rotation in {90, 270}:
        width, height = height, width

    return width, height


def should_delete_video(video_path: Path, min_short_side: int = 1080) -> tuple[bool, str]:
    stream = run_ffprobe(video_path)

    if stream is None:
        return False, "无法读取视频信息，保留"

    size = get_display_size(stream)

    if size is None:
        return False, "无法读取分辨率，保留"

    width, height = size
    short_side = min(width, height)

    if short_side < min_short_side:
        return True, f"{width}x{height}, 短边 {short_side} < {min_short_side}"

    return False, f"{width}x{height}, 短边 {short_side} >= {min_short_side}"


def iter_video_files(input_dir: Path, recursive: bool = False):
    if recursive:
        files = input_dir.rglob("*")
    else:
        files = input_dir.iterdir()

    for path in files:
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def filter_videos(
    input_dir: str,
    min_short_side: int = 1920,
    recursive: bool = False,
    dry_run: bool = False,
):
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹: {input_dir}")

    video_files = list(iter_video_files(input_dir, recursive=recursive))

    print(f"[INFO] 输入文件夹: {input_dir}")
    print(f"[INFO] 视频数量: {len(video_files)}")
    print(f"[INFO] 删除规则: 短边 < {min_short_side} 的视频会被删除")
    print(f"[INFO] dry_run: {dry_run}")
    print()

    deleted_count = 0
    kept_count = 0
    failed_count = 0

    for video_path in video_files:
        delete_flag, reason = should_delete_video(video_path, min_short_side)

        if delete_flag:
            if dry_run:
                print(f"[DRY-RUN DELETE] {video_path} | {reason}")
            else:
                try:
                    video_path.unlink()
                    print(f"[DELETE] {video_path} | {reason}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[ERROR] 删除失败: {video_path}")
                    print(f"        {e}")
                    failed_count += 1
        else:
            print(f"[KEEP] {video_path} | {reason}")
            kept_count += 1

    print()
    print("[DONE]")
    print(f"保留: {kept_count}")
    print(f"删除: {deleted_count}")
    print(f"失败: {failed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="删除输入文件夹中低于 1080p 的视频，横屏和竖屏都按短边 >= 1080 判断"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default= 'swing_data/raw_data/01_bili_data',
        required=True,
        help="输入视频文件夹路径",
    )

    parser.add_argument(
        "--min_short_side",
        type=int,
        default=1080,
        help="最小短边分辨率，默认 1080",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归处理子文件夹",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印会删除哪些文件，不真正删除",
    )

    args = parser.parse_args()

    filter_videos(
        input_dir=args.input_dir,
        min_short_side=args.min_short_side,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()


'''
python ./process/01_filter_delete_low_res_videos.py --input_dir swing_data/raw_data/03_bili_data

'''