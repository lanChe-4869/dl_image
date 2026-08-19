# 02_collect_bilibili_bvids_login.py
import argparse
import re
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright


BVID_RE = re.compile(r"BV[0-9A-Za-z]{10}")


def collect_bvids(
    keywords: list[str],
    pages_per_keyword: int = 5,
    output_path: str = "bili_links.txt",
    sleep_seconds: float = 2.0,
    user_data_dir: str = "./chrome_bili_profile",
    login_check: bool = True,
):
    all_bvids = set()

    Path(user_data_dir).mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # 重点：launch_persistent_context 会保存 cookies / localStorage
        # channel="chrome" 表示用你电脑上安装的 Google Chrome
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel="chrome",
            headless=False,
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )

        page = context.new_page()

        if login_check:
            page.goto("https://www.bilibili.com/", wait_until="domcontentloaded", timeout=60_000)
            print("\n[INFO] 浏览器已打开 B 站。")
            print("[INFO] 如果这是第一次运行，请在打开的 Chrome 窗口里手动登录 B 站。")
            input("[INFO] 登录完成后，回到终端按 Enter 继续；如果已经登录，也直接按 Enter：")

        for keyword in keywords:
            for page_no in range(1, pages_per_keyword + 1):
                url = f"https://search.bilibili.com/all?keyword={quote(keyword)}&page={page_no}"
                print(f"[SEARCH] {keyword} page={page_no}")

                try:
                    page.goto(url, wait_until="networkidle", timeout=60_000)
                    time.sleep(sleep_seconds)

                    html = page.content()
                    bvids = set(BVID_RE.findall(html))

                    print(f"  found {len(bvids)} BV ids")
                    all_bvids.update(bvids)

                except Exception as e:
                    print(f"[WARN] 搜索失败: keyword={keyword}, page={page_no}")
                    print(e)

        context.close()

    all_bvids = sorted(all_bvids)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for bvid in all_bvids:
            f.write(bvid + "\n")

    print(f"[DONE] saved {len(all_bvids)} BV ids to {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["双人舞蹈", "双人编舞", "双人街舞", "双人翻跳"],
    )
    parser.add_argument("--pages_per_keyword", type=int, default=5)
    parser.add_argument("--output_path", type=str, default="bili_links.txt")
    parser.add_argument("--sleep_seconds", type=float, default=2.0)

    # 持久化 Chrome 登录状态的位置
    parser.add_argument("--user_data_dir", type=str, default="./chrome_bili_profile")

    # 如果你已经登录过，可以加 --no_login_check 跳过等待
    parser.add_argument("--no_login_check", action="store_true")

    args = parser.parse_args()

    collect_bvids(
        keywords=args.keywords,
        pages_per_keyword=args.pages_per_keyword,
        output_path=args.output_path,
        sleep_seconds=args.sleep_seconds,
        user_data_dir=args.user_data_dir,
        login_check=not args.no_login_check,
    )


if __name__ == "__main__":
    main()

'''
01. 第一次运行需要登录账号一下，登录账号的推荐更准确
python 02_collect_bilibili_bvids.py `
  --keywords 双人 `
  --pages_per_keyword 10 `
  --output_path ./video_links/bili_links_1.txt `
  --user_data_dir ./chrome_bili_profile

02. 登录状态会保存在：./chrome_bili_profile
双人舞蹈 双人编舞 双人街舞 双人翻跳 双人爵士舞 双人现代舞
以后不用再登录，可以运行：

python 02_collect_bilibili_bvids.py `
  --keywords 双人舞蹈 `
  --pages_per_keyword 20 `
  --output_path ./video_links/bili_links_3.txt `
  --user_data_dir ./chrome_bili_profile `
  --no_login_check

'''