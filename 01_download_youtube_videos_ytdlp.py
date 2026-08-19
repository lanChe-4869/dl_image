#!/usr/bin/env python3
"""
Download YouTube videos from a text file with yt-dlp.

This script is designed to replace the pytubefix-based download_videos()
function in the original pipeline.

Default behavior:
- read one YouTube video ID or URL per line
- download VIDEO-ONLY stream, no audio
- do not download subtitles, thumbnails, or metadata side files
- save to swing_data/raw_data/<youtube_id>.<ext>
- prefer MP4 video-only if available, because the existing ffmpeg commands
  expect files like swing_data/raw_data/<youtube_id>.mp4

Install:
    pip install -U yt-dlp tqdm

Usage:
    python download_youtube_videos_ytdlp.py --video_links_path video_links.txt

For absolute best video-only quality, even if the file becomes .webm:
    python download_youtube_videos_ytdlp.py --video_links_path video_links.txt --absolute-best

For MP4-compatible downstream trimming:
    python download_youtube_videos_ytdlp.py --video_links_path video_links.txt --prefer-mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
import yt_dlp
from yt_dlp.utils import DownloadError


YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


def read_video_links(video_links_path: str | Path) -> list[str]:
    """
    Read one YouTube video ID or full URL per line.

    Empty lines and lines starting with # are ignored.
    If a line is already a URL, it is used as-is.
    Otherwise, it is treated as a YouTube video ID.
    """
    path = Path(video_links_path)
    if not path.exists():
        raise FileNotFoundError(f"video_links file not found: {path}")

    urls: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        item = raw_line.strip()
        if not item or item.startswith("#"):
            continue

        if item.startswith(("http://", "https://")):
            urls.append(item)
        else:
            urls.append(YOUTUBE_WATCH_URL.format(video_id=item))

    if not urls:
        raise ValueError(f"No valid video IDs or URLs found in: {path}")

    return urls


def build_ydl_opts(
    output_dir: str | Path,
    format_selector: str,
    archive_path: str | Path | None = None,
    cookies_path: str | Path | None = None,
    concurrent_fragments: int = 4,
    verbose: bool = False,
) -> dict:
    """
    Build yt-dlp options.

    format_selector examples:
      - "bestvideo[ext=mp4]/bestvideo": prefer MP4 video-only, fallback to any best video-only
      - "bestvideo": absolute best video-only stream, often WebM/VP9/AV1 on YouTube
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        # Download video-only stream. No "+bestaudio" is used.
        "format": format_selector,

        # Keep output names compatible with the previous pipeline:
        # swing_data/raw_data/<youtube_id>.<actual_ext>
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),

        # We pass one watch URL at a time, but this prevents accidental playlist expansion.
        "noplaylist": True,

        # Resume partial downloads and retry transient network failures.
        "continuedl": True,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": concurrent_fragments,

        # Do not create extra files.
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writethumbnail": False,
        "writeinfojson": False,

        # Keep logs visible enough for debugging.
        "quiet": False,
        "no_warnings": False,
        "verbose": verbose,
    }

    if archive_path is not None:
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        # yt-dlp will skip videos already recorded in this archive.
        opts["download_archive"] = str(archive_path)

    if cookies_path is not None:
        # Useful when YouTube asks for sign-in/age verification.
        opts["cookiefile"] = str(cookies_path)

    return opts


def download_videos(
    video_links_path: str | Path,
    output_dir: str | Path = "swing_data/raw_data",
    prefer_mp4: bool = True,
    archive_path: str | Path | None = "swing_data/raw_data/downloaded.txt",
    cookies_path: str | Path | None = None,
    concurrent_fragments: int = 4,
    verbose: bool = False,
) -> None:
    urls = read_video_links(video_links_path)

    if prefer_mp4:
        # Best MP4 video-only if available; otherwise fall back to best video-only.
        # This keeps most outputs as .mp4 for compatibility with existing ffmpeg commands.
        format_selector = "bestvideo[ext=mp4]/bestvideo"
    else:
        # Absolute best video-only stream. On YouTube this may be .webm, VP9, or AV1.
        format_selector = "bestvideo"

    ydl_opts = build_ydl_opts(
        output_dir=output_dir,
        format_selector=format_selector,
        archive_path=archive_path,
        cookies_path=cookies_path,
        concurrent_fragments=concurrent_fragments,
        verbose=verbose,
    )

    failed: list[str] = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in tqdm(urls, desc="Downloading video-only streams"):
            try:
                ydl.download([url])
            except DownloadError as exc:
                failed.append(url)
                print(f"\n[FAILED] {url}\n{exc}\n", file=sys.stderr)
            except Exception as exc:
                failed.append(url)
                print(f"\n[FAILED] {url}\n{type(exc).__name__}: {exc}\n", file=sys.stderr)

    print(f"\nFinished. Total: {len(urls)}, failed: {len(failed)}")

    if failed:
        failed_path = Path(output_dir) / "failed_downloads.txt"
        failed_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"Failed URL list saved to: {failed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download highest-quality YouTube video-only streams with yt-dlp."
    )

    parser.add_argument(
        "--video_links_path",
        type=str,
        default="video_links.txt",
        help="Path to a txt file containing one YouTube video ID or URL per line.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swing_data/raw_data",
        help="Directory for downloaded raw videos.",
    )

    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument(
        "--prefer-mp4",
        action="store_true",
        default=True,
        help="Prefer best MP4 video-only stream, fallback to best video-only. Default.",
    )
    quality_group.add_argument(
        "--absolute-best",
        action="store_true",
        help="Download absolute best video-only stream, even if it is WebM/VP9/AV1.",
    )

    parser.add_argument(
        "--archive_path",
        type=str,
        default="swing_data/raw_data/downloaded.txt",
        help="yt-dlp archive file used to skip already downloaded videos. Use '' to disable.",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Optional cookies.txt path for YouTube sign-in/age-restricted videos.",
    )
    parser.add_argument(
        "--concurrent_fragments",
        type=int,
        default=4,
        help="Number of fragments to download concurrently for DASH/HLS videos.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose yt-dlp logs.",
    )

    args = parser.parse_args()

    archive_path = args.archive_path if args.archive_path else None
    prefer_mp4 = not args.absolute_best

    download_videos(
        video_links_path=args.video_links_path,
        output_dir=args.output_dir,
        prefer_mp4=prefer_mp4,
        archive_path=archive_path,
        cookies_path=args.cookies,
        concurrent_fragments=args.concurrent_fragments,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()


'''
python 01_download_youtube_videos_ytdlp.py `
  --video_links_path video_links.txt `
  --output_dir swing_data/raw_data

'''