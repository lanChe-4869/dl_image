# download_bili_batch.py
import argparse
import os
import re
from pathlib import Path

from tqdm import tqdm
from yt_dlp import YoutubeDL


BVID_RE = re.compile(r"(BV[0-9A-Za-z]{10})")


def normalize_bili_url(line: str) -> str | None:
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    if line.startswith("http://") or line.startswith("https://"):
        return line

    match = BVID_RE.search(line)
    if match:
        bvid = match.group(1)
        return f"https://www.bilibili.com/video/{bvid}"

    print(f"[WARN] 无法识别这一行，已跳过: {line}")
    return None


def load_urls(links_path: str) -> list[str]:
    urls = []

    with open(links_path, "r", encoding="utf-8") as f:
        for line in f:
            url = normalize_bili_url(line)
            if url:
                urls.append(url)

    return urls


def download_bili_videos(
    links_path: str,
    output_dir: str = "swing_data/raw_data",
    cookiefile: str = "cookies.txt",
):
    os.makedirs(output_dir, exist_ok=True)

    urls = load_urls(links_path)

    print(f"[INFO] 共读取到 {len(urls)} 个 B 站视频链接")

    ydl_opts = {
        # 只下载最高质量视频流，不下载音频
        "format": "bv",

        # 输出到 raw_data，用 display_id 尽量保留 BV 号
        "paths": {
            "home": output_dir,
        },
        "outtmpl": {
            "default": "%(display_id)s.%(ext)s",
        },

        # 使用你已经成功导出的 cookie
        "cookiefile": cookiefile,

        # B 站请求头
        "http_headers": {
            "Referer": "https://www.bilibili.com/",
            "Origin": "https://www.bilibili.com",
        },

        # 批量下载时放慢一点，降低 412 风险
        "sleep_interval_requests": 2,
        "sleep_interval": 5,
        "max_sleep_interval": 12,

        # 失败重试与断点续传
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "ignoreerrors": True,

        # 已下载记录，重复运行时自动跳过
        "download_archive": "downloaded_bili.txt",

        # 不下载附加内容
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writethumbnail": False,
        "writeinfojson": False,
        "noplaylist": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        for url in tqdm(urls, desc="Downloading Bilibili videos"):
            try:
                ydl.download([url])
            except Exception as e:
                print(f"[ERROR] 下载失败: {url}")
                print(e)


def main():
    parser = argparse.ArgumentParser(description="Batch download Bilibili video-only streams")
    parser.add_argument("--links_path", type=str, default="bili_links.txt")
    parser.add_argument("--output_dir", type=str, default="swing_data/raw_data")
    parser.add_argument("--cookiefile", type=str, default="cookies.txt")

    args = parser.parse_args()

    download_bili_videos(
        links_path=args.links_path,
        output_dir=args.output_dir,
        cookiefile=args.cookiefile,
    )


if __name__ == "__main__":
    main()

'''
02. 下载视频
python 02_download_bilibili_video_only.py `
  --links_path ./video_links/bili_links_3.txt `
  --output_dir swing_data/raw_data/03_bili_data `
  --cookiefile cookies.txt

  
yt-dlp `
  -f "bv" `
  --cookies cookies.txt `
  --referer "https://www.bilibili.com/" `
  --sleep-requests 2 `
  --sleep-interval 5 `
  --max-sleep-interval 12 `
  -o "swing_data/raw_data/%(id)s.%(ext)s" `
  "https://www.bilibili.com/video/BV11B4y127Kw"
  
'''