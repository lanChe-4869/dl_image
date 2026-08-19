# 05_manual_click_collect_douyin_model_ids.py

import argparse
import re
from pathlib import Path
from urllib.parse import quote, urlparse, parse_qs

from playwright.sync_api import sync_playwright


ID_RE = re.compile(r"(?<!\d)(\d{16,22})(?!\d)")


def extract_model_id_from_url(url: str) -> str:
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        qs = parse_qs(parsed.query or "")

        for key in ["modal_id", "aweme_id", "item_id", "group_id"]:
            values = qs.get(key)
            if values and values[0] and values[0].isdigit():
                return values[0].strip()

        m = re.search(r"/(?:note|video)/(\d{16,22})(?:/|$)", path)
        if m:
            return m.group(1)

        m = ID_RE.search(url)
        if m:
            return m.group(1)

    except Exception:
        return ""

    return ""


def extract_ids_from_text(text: str) -> set[str]:
    if not text:
        return set()

    ids = set()

    patterns = [
        r'"aweme_id"\s*:\s*"?(\d{16,22})"?',
        r'"group_id"\s*:\s*"?(\d{16,22})"?',
        r'"item_id"\s*:\s*"?(\d{16,22})"?',
        r'"modal_id"\s*:\s*"?(\d{16,22})"?',
    ]

    for pat in patterns:
        for m in re.finditer(pat, text):
            ids.add(m.group(1))

    return ids


def load_existing_ids(path: Path) -> set[str]:
    ids = set()

    if not path.exists():
        return ids

    for line in path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines():

        raw = line.strip()

        if not raw or raw.startswith("#"):
            continue

        mid = extract_model_id_from_url(raw) or (
            raw if raw.isdigit() else ""
        )

        if mid:
            ids.add(mid)

    return ids


def append_id(path: Path, model_id: str, args, source_url: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)

    if args.write_urls:
        line = f"https://www.douyin.com/note/{model_id}"
    elif args.write_with_source_url:
        line = f"{model_id}\t{source_url}"
    else:
        line = model_id

    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def create_browser_context(playwright, args):
    kwargs = {
        "user_data_dir": args.user_data_dir,
        "headless": args.headless,
        "viewport": {
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
    }

    if args.channel:
        kwargs["channel"] = args.channel

    return playwright.chromium.launch_persistent_context(**kwargs)


def try_save_model_id(
    model_id: str,
    seen_ids: set[str],
    processed_ids: set[str],
    output_txt: Path,
    args,
    source_url: str,
    reason: str,
) -> bool:
    if not model_id:
        return False

    # 先查历史已处理 ID
    if model_id in processed_ids:
        print(f"[SKIP-PROCESSED] {model_id} 已在 processed_modal_ids.txt 中，不记录")
        return False

    # 再查本次输出文件里的 ID
    if model_id in seen_ids:
        print(f"[SKIP-SEEN] {model_id} 本次已记录，不重复写入")
        return False

    seen_ids.add(model_id)
    append_id(output_txt, model_id, args, source_url)

    print(f"[SAVED] {model_id}  new_total={len(seen_ids)}  reason={reason}")
    print(f"        {source_url}")

    return True


def maybe_collect_from_page(
    page,
    seen_ids: set[str],
    processed_ids: set[str],
    output_txt: Path,
    args,
    reason: str,
) -> bool:
    url = page.url or ""

    model_id = extract_model_id_from_url(url)

    if model_id:
        return try_save_model_id(
            model_id=model_id,
            seen_ids=seen_ids,
            processed_ids=processed_ids,
            output_txt=output_txt,
            args=args,
            source_url=url,
            reason=reason,
        )

    if args.scan_page_content:
        try:
            html = page.content()
            ids = extract_ids_from_text(html)

            for mid in sorted(ids):
                saved = try_save_model_id(
                    model_id=mid,
                    seen_ids=seen_ids,
                    processed_ids=processed_ids,
                    output_txt=output_txt,
                    args=args,
                    source_url=url,
                    reason="page_content",
                )

                if saved:
                    return True

        except Exception:
            pass

    return False


def main():
    parser = argparse.ArgumentParser(
        description="手动点击抖音帖子详情页，自动收集 URL 里的 model_id 到 txt"
    )

    parser.add_argument("--keyword", type=str, default="双人合照")
    parser.add_argument("--search_url", type=str, default="")
    parser.add_argument("--output_txt", type=str, default="./douyin_model_ids/model_ids.txt")

    # 新增：历史已处理 ID 文件
    parser.add_argument(
        "--processed_ids_txt",
        type=str,
        default="",
        help="历史已处理 model_id 文件，命中的 ID 会跳过，不写入 output_txt"
    )

    parser.add_argument(
        "--target_count",
        type=int,
        default=0,
        help="本次新收集到多少个 ID 后自动结束；0 表示不自动结束"
    )

    parser.add_argument("--user_data_dir", type=str, default="./chrome_douyin_profile")
    parser.add_argument("--channel", type=str, default="chrome")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_pause", action="store_true")

    parser.add_argument("--viewport_width", type=int, default=1400)
    parser.add_argument("--viewport_height", type=int, default=900)
    parser.add_argument("--poll_seconds", type=float, default=0.5)

    parser.add_argument("--write_urls", action="store_true")
    parser.add_argument("--write_with_source_url", action="store_true")
    parser.add_argument("--scan_page_content", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    output_txt = Path(args.output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_existing_ids(output_txt)

    processed_ids = set()
    if args.processed_ids_txt:
        processed_path = Path(args.processed_ids_txt)
        processed_ids = load_existing_ids(processed_path)

    print(f"[OUTPUT] {output_txt}")
    print(f"[EXISTING] 当前输出文件已有 ID: {len(seen_ids)}")
    print(f"[PROCESSED] 历史已处理 ID: {len(processed_ids)}")

    if args.search_url:
        start_url = args.search_url
    else:
        start_url = f"https://www.douyin.com/search/{quote(args.keyword)}"

    last_url = ""
    saved_this_run = 0

    with sync_playwright() as p:
        context = create_browser_context(p, args)
        page = context.new_page()

        def on_frame_navigated(frame):
            nonlocal saved_this_run

            if frame == page.main_frame:
                try:
                    saved = maybe_collect_from_page(
                        page,
                        seen_ids,
                        processed_ids,
                        output_txt,
                        args,
                        "navigation",
                    )
                    if saved:
                        saved_this_run += 1
                except Exception:
                    pass

        page.on("framenavigated", on_frame_navigated)

        print(f"[OPEN] {start_url}")
        page.goto(start_url, wait_until="domcontentloaded", timeout=60_000)

        if not args.no_pause:
            print()
            print("请在浏览器里登录抖音，并切到你想看的搜索结果类型，例如 图文 / 图片 / 综合。")
            print("之后你手动点击任意帖子详情页，脚本会自动保存 URL 里的 model_id。")
            print("如果 model_id 已在 processed_ids_txt 中，会自动跳过。")
            input("准备好后按 Enter 开始监听...")

        print("[LISTENING] 开始监听。Ctrl+C 可结束。")

        try:
            while True:
                current_url = page.url or ""

                if current_url != last_url:
                    last_url = current_url

                    if args.debug:
                        print(f"[URL] {current_url}")

                    saved = maybe_collect_from_page(
                        page,
                        seen_ids,
                        processed_ids,
                        output_txt,
                        args,
                        "url_changed",
                    )

                    if saved:
                        saved_this_run += 1

                else:
                    if args.scan_page_content:
                        saved = maybe_collect_from_page(
                            page,
                            seen_ids,
                            processed_ids,
                            output_txt,
                            args,
                            "poll",
                        )

                        if saved:
                            saved_this_run += 1

                if args.target_count and saved_this_run >= args.target_count:
                    print(f"[DONE] 本次新收集已达到 target_count={args.target_count}")
                    break

                page.wait_for_timeout(int(args.poll_seconds * 1000))

        except KeyboardInterrupt:
            print("\n[STOP] 收到 Ctrl+C，结束监听。")

        finally:
            try:
                context.close()
            except Exception:
                pass

    print()
    print("[ALL DONE]")
    print(f"本次新保存 ID 数: {saved_this_run}")
    print(f"当前输出文件累计 ID 数: {len(seen_ids)}")
    print(f"历史已处理 ID 数: {len(processed_ids)}")
    print(f"输出文件: {output_txt}")


if __name__ == "__main__":
    main()

'''
python 05_manual_click_collect_douyin_model_ids.py `
  --keyword 景点打卡 `
  --output_txt images\double\20260819_double\0819_model_ids.txt `
  --processed_ids_txt images\all_processed_modal_ids.txt
'''