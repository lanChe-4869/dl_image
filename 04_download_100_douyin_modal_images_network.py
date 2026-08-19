# 05_download_100_douyin_modal_images_network.py
# 目标：
#   基于 04_download_one_douyin_post_images_network.py 的“网络响应捕获图片”逻辑，
#   扩展为批量版：从搜索页点击卡片获取 modal_id，并下载 100 组 modal_id 对应帖子的图片。
#
# 核心流程：
#   1. 打开抖音搜索页，例如“双人合照”
#   2. 你手动登录并切到合适的搜索结果页，例如“图文 / 图片 / 综合”
#   3. 脚本点击当前屏幕可见卡片，读取 URL 里的 modal_id
#   4. 对这个 modal_id 立刻使用“网络响应捕获”的方式下载该帖子多张图片
#   5. 每个 modal_id 单独一个文件夹
#   6. 处理完该 modal_id 后返回搜索页继续下一个
#   7. 默认处理 100 组 modal_id 后停止
#
# 安装：
#   pip install playwright pillow
#   playwright install chromium
#
# 小规模测试：
#   python 05_download_100_douyin_modal_images_network.py ^
#     --keyword 双人合照 ^
#     --output_dir ./douyin_100_modal_images/双人合照 ^
#     --target_count 5 ^
#     --debug
#
# 正式运行：
#   python 05_download_100_douyin_modal_images_network.py ^
#     --keyword 双人合照 ^
#     --output_dir ./douyin_100_modal_images/双人合照 ^
#     --target_count 100
#
# 如果你已经有 modal_urls.txt：
#   python 05_download_100_douyin_modal_images_network.py ^
#     --modal_urls_file ./douyin_modal_ids/双人合照/modal_urls.txt ^
#     --output_dir ./douyin_100_modal_images/双人合照 ^
#     --target_count 100 ^
#     --no_pause

import argparse
import hashlib
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse, parse_qs

from PIL import Image, UnidentifiedImageError
from playwright.sync_api import sync_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError


IMAGE_HOST_KEYWORDS = [
    "douyinpic.com",
    "byteimg.com",
    "pstatp.com",
    "douyinstatic.com",
    "snssdk.com",
]


def md5_bytes(data: bytes, n: int = 16) -> str:
    return hashlib.md5(data).hexdigest()[:n]


def sanitize_filename(name: str, max_len: int = 80) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.strip(". ")
    return (name or "untitled")[:max_len]


def get_modal_id(url: str) -> str:
    try:
        qs = parse_qs(urlparse(url).query)
        values = qs.get("modal_id")
        if values and values[0]:
            return str(values[0]).strip()
    except Exception:
        pass
    return ""


def is_modal_url(url: str) -> bool:
    return bool(get_modal_id(url))


def is_image_response_url(url: str) -> bool:
    lower = (url or "").lower()

    if not lower.startswith("http"):
        return False

    if not any(host in lower for host in IMAGE_HOST_KEYWORDS):
        return False

    bad_keywords = [
        "avatar",
        "aweme-avatar",
        "emoji",
        "sprite",
        "icon",
        "logo",
        ".svg",
    ]

    if any(k in lower for k in bad_keywords):
        return False

    return True


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


def get_title(page) -> str:
    candidates = []

    try:
        title = page.title()
        if title:
            candidates.append(title)
    except Exception:
        pass

    js = """
    () => {
        const out = [];

        const metas = [
            'meta[property="og:title"]',
            'meta[name="description"]',
            'meta[property="og:description"]'
        ];

        for (const selector of metas) {
            const el = document.querySelector(selector);
            if (el && el.content) out.push(el.content);
        }

        const nodes = Array.from(document.querySelectorAll(
            'h1, h2, [class*="title"], [class*="desc"], [class*="content"], [class*="note"]'
        ));

        for (const el of nodes.slice(0, 40)) {
            const t = (el.innerText || el.textContent || '').trim();
            if (t && t.length >= 2) out.push(t);
        }

        return out;
    }
    """

    try:
        candidates.extend(page.evaluate(js))
    except Exception:
        pass

    for c in candidates:
        c = re.sub(r"\s+", " ", str(c)).strip()
        c = c.replace(" - 抖音", "")
        if c and len(c) >= 2:
            return c[:80]

    return ""


def make_output_folder(root: Path, modal_id: str, title: str) -> Path:
    title = sanitize_filename(title, 50)

    if title and title != "untitled":
        return root / f"{modal_id}_{title}"

    return root / modal_id


def analyze_image_bytes(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        width, height = img.size
        fmt = img.format or "UNKNOWN"
        return True, img.convert("RGB"), width, height, fmt
    except (UnidentifiedImageError, OSError, ValueError):
        return False, None, 0, 0, "UNKNOWN"


def is_content_photo(width: int, height: int, byte_len: int, args) -> tuple[bool, str]:
    long_side = max(width, height)
    short_side = min(width, height)

    if byte_len < args.min_bytes:
        return False, f"bytes_too_small:{byte_len}"

    if width < args.min_width or height < args.min_height:
        return False, f"width_height_too_small:{width}x{height}"

    if long_side < args.min_long_side:
        return False, f"long_side_too_small:{long_side}"

    if short_side < args.min_short_side:
        return False, f"short_side_too_small:{short_side}"

    ratio = long_side / max(short_side, 1)
    if ratio > args.max_aspect_ratio:
        return False, f"aspect_ratio_too_large:{ratio:.2f}"

    return True, "accepted"


def write_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def click_next(page, args) -> bool:
    selectors = [
        'button[aria-label*="下一"]',
        'button[aria-label*="next"]',
        '[aria-label*="下一"]',
        '[class*="next"]',
        '[class*="Next"]',
        'div[role="button"][aria-label*="下一"]',
    ]

    for selector in selectors:
        try:
            loc = page.locator(selector)
            if loc.count() > 0:
                loc.first.click(timeout=1200)
                page.wait_for_timeout(int(args.step_wait_seconds * 1000))
                return True
        except Exception:
            pass

    try:
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(int(args.step_wait_seconds * 1000))
        return True
    except Exception:
        return False


def maybe_save_candidate(candidate: dict, out_dir: Path, seen_hashes: set[str], args, records: list[dict]) -> bool:
    data = candidate["body"]
    url = candidate["url"]
    content_type = candidate.get("content_type", "")
    image_hash = md5_bytes(data)

    ok_img, img, width, height, fmt = analyze_image_bytes(data)

    rec = {
        "url": url,
        "content_type": content_type,
        "bytes": len(data),
        "hash": image_hash,
        "pil_ok": ok_img,
        "width": width,
        "height": height,
        "format": fmt,
        "saved": False,
    }

    if not ok_img:
        rec["reason"] = "pil_unreadable"
        records.append(rec)
        return False

    is_content, reason = is_content_photo(width, height, len(data), args)
    rec["reason"] = reason

    if image_hash in seen_hashes:
        rec["reason"] = "duplicate_hash"
        records.append(rec)
        return False

    if not is_content:
        if args.save_rejected:
            reject_dir = out_dir / "_rejected"
            reject_dir.mkdir(parents=True, exist_ok=True)
            reject_path = reject_dir / f"reject_{image_hash}_{width}x{height}.jpg"
            try:
                img.save(reject_path, format="JPEG", quality=90)
                rec["reject_path"] = str(reject_path)
            except Exception:
                pass

        records.append(rec)
        return False

    seen_hashes.add(image_hash)
    index = len(seen_hashes)
    out_path = out_dir / f"image_{index:03d}_{image_hash}_{width}x{height}.jpg"
    img.save(out_path, format="JPEG", quality=args.jpg_quality, subsampling=0)

    rec["saved"] = True
    rec["saved_path"] = str(out_path)
    records.append(rec)

    print(f"      [SAVED] {out_path.name} bytes={len(data)} size={width}x{height}")

    return True


def get_visible_card_click_points(page, args) -> list[dict]:
    js = """
    (opts) => {
        const minW = opts.minW;
        const minH = opts.minH;
        const maxW = opts.maxW;
        const maxH = opts.maxH;
        const minX = opts.minX;
        const minY = opts.minY;
        const vw = window.innerWidth;
        const vh = window.innerHeight;

        function visibleRect(el) {
            const r = el.getBoundingClientRect();
            if (!r || r.width <= 0 || r.height <= 0) return null;

            const x1 = Math.max(0, r.left);
            const y1 = Math.max(0, r.top);
            const x2 = Math.min(vw, r.right);
            const y2 = Math.min(vh, r.bottom);
            const w = x2 - x1;
            const h = y2 - y1;

            if (w <= 0 || h <= 0) return null;

            return {
                left: x1,
                top: y1,
                right: x2,
                bottom: y2,
                width: w,
                height: h
            };
        }

        function hasVisual(el) {
            const tag = el.tagName ? el.tagName.toLowerCase() : "";

            if (["img", "video", "canvas"].includes(tag)) return true;

            const style = window.getComputedStyle(el);
            const bg = style.backgroundImage || "";

            if (bg && bg !== "none" && bg.includes("url(")) return true;

            if (el.querySelector && el.querySelector("img, video, canvas")) return true;

            return false;
        }

        function isBadText(t) {
            if (!t) return false;

            const bad = [
                "精选", "推荐", "关注", "朋友", "我的", "直播", "放映厅",
                "短剧", "小游戏", "下载抖音", "创作者中心", "发布视频",
                "用户服务协议", "隐私政策", "我的喜欢", "我的收藏",
                "观看历史", "稍后再看", "我的作品", "我的预约"
            ];

            return bad.some(x => t.includes(x));
        }

        const candidates = [];
        const baseNodes = Array.from(
            document.querySelectorAll("img, video, canvas, [style*='background-image']")
        );

        for (const base of baseNodes) {
            let el = base;
            let chosen = null;

            for (let depth = 0; depth < 6 && el; depth++, el = el.parentElement) {
                const r = visibleRect(el);
                if (!r) continue;

                if (
                    r.width >= minW &&
                    r.height >= minH &&
                    r.width <= maxW &&
                    r.height <= maxH &&
                    r.left >= minX &&
                    r.top >= minY &&
                    hasVisual(el)
                ) {
                    chosen = {el, r};

                    if (r.width > maxW * 0.85 || r.height > maxH * 0.85) {
                        break;
                    }
                }
            }

            if (!chosen) continue;

            const card = chosen.el;
            const r = chosen.r;
            const text = (card.innerText || card.textContent || "").trim().slice(0, 160);

            if (isBadText(text)) continue;

            const x = Math.round(r.left + r.width / 2);
            const y = Math.round(r.top + r.height / 2);

            if (x < minX || y < minY || x > vw - 20 || y > vh - 20) continue;

            candidates.push({
                x,
                y,
                w: Math.round(r.width),
                h: Math.round(r.height),
                left: Math.round(r.left),
                top: Math.round(r.top),
                tag: card.tagName,
                text
            });
        }

        const seen = new Set();
        const deduped = [];

        for (const c of candidates) {
            const key = `${Math.round(c.x / 35)}_${Math.round(c.y / 35)}`;

            if (seen.has(key)) continue;

            seen.add(key);
            deduped.push(c);
        }

        deduped.sort((a, b) => (a.top - b.top) || (a.left - b.left));

        return deduped.slice(0, opts.maxPoints);
    }
    """

    try:
        return page.evaluate(
            js,
            {
                "minW": args.card_min_width,
                "minH": args.card_min_height,
                "maxW": args.card_max_width,
                "maxH": args.card_max_height,
                "minX": args.card_min_x,
                "minY": args.card_min_y,
                "maxPoints": args.max_clicks_per_scroll,
            },
        ) or []
    except PlaywrightError as e:
        print(f"[WARN] 获取卡片点失败: {e}")
        return []


def close_modal_return_search(page, search_url: str):
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(600)
    except Exception:
        pass

    if is_modal_url(page.url):
        try:
            page.go_back(wait_until="domcontentloaded", timeout=5000)
            page.wait_for_timeout(700)
        except Exception:
            pass

    if is_modal_url(page.url):
        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=10000)
            page.wait_for_timeout(1000)
        except Exception:
            pass


def process_open_modal(page, modal_url: str, args) -> dict:
    modal_id = get_modal_id(modal_url) or "unknown_modal"
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    captured = []
    captured_seen = set()

    def handle_response(response):
        try:
            url = response.url
            if not is_image_response_url(url):
                return

            headers = response.headers
            content_type = headers.get("content-type", "")

            if "image" not in content_type.lower():
                return

            body = response.body()

            if not body:
                return

            key = md5_bytes(body, 24)

            if key in captured_seen:
                return

            captured_seen.add(key)
            captured.append(
                {
                    "url": url,
                    "content_type": content_type,
                    "body": body,
                    "time": time.time(),
                }
            )
        except Exception:
            pass

    page.on("response", handle_response)

    title = get_title(page)
    out_dir = make_output_folder(output_root, modal_id, title)
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = out_dir / "_debug"
    if args.debug or args.save_rejected:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"    [MODAL] {modal_id}")
    print(f"    [TITLE] {title or '(无标题)'}")
    print(f"    [OUT] {out_dir}")

    (out_dir / "post_meta.json").write_text(
        json.dumps(
            {
                "url": modal_url,
                "modal_id": modal_id,
                "title": title,
                "expected_count": args.expected_count,
                "method": "network_response_capture_batch",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    seen_hashes = set()
    processed_capture_count = 0
    records = []
    no_new_rounds = 0

    for step in range(args.max_steps):
        page.wait_for_timeout(int(args.capture_wait_seconds * 1000))

        new_candidates = captured[processed_capture_count:]
        processed_capture_count = len(captured)

        print(
            f"    [STEP {step + 1}/{args.max_steps}] "
            f"captured_total={len(captured)} new={len(new_candidates)} saved={len(seen_hashes)}"
        )

        saved_this_round = 0

        for cand in new_candidates:
            if maybe_save_candidate(cand, out_dir, seen_hashes, args, records):
                saved_this_round += 1

        if args.debug:
            try:
                page.screenshot(path=str(debug_dir / f"screen_{step + 1:03d}.png"), full_page=False)
            except Exception:
                pass

        (out_dir / "download_records.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.expected_count and len(seen_hashes) >= args.expected_count:
            print(f"    [DONE] reached expected_count={args.expected_count}")
            break

        if saved_this_round == 0:
            no_new_rounds += 1
        else:
            no_new_rounds = 0

        if no_new_rounds >= args.stop_after_no_new:
            print(f"    [DONE] no new saved image for {no_new_rounds} rounds")
            break

        click_next(page, args)

    try:
        page.remove_listener("response", handle_response)
    except Exception:
        pass

    return {
        "modal_id": modal_id,
        "modal_url": modal_url,
        "title": title,
        "output_dir": str(out_dir),
        "saved_count": len(seen_hashes),
        "captured_count": len(captured),
    }


def load_modal_urls_from_file(path: str, target_count: int | None = None) -> list[str]:
    urls = []
    seen = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()

            if not raw or raw.startswith("#"):
                continue

            if raw.isdigit():
                # 只有 modal_id 时，构造一个搜索 URL。keyword 不重要，只要 modal_id 可打开。
                url = f"https://www.douyin.com/search/%E5%8F%8C%E4%BA%BA%E5%90%88%E7%85%A7?modal_id={raw}"
            else:
                url = raw

            mid = get_modal_id(url)

            if not mid or mid in seen:
                continue

            seen.add(mid)
            urls.append(url)

            if target_count and len(urls) >= target_count:
                break

    return urls


def collect_and_download_from_search(args):
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    processed_path = output_root / "processed_modal_ids.txt"
    summary_path = output_root / "batch_summary.jsonl"

    processed_ids = set()

    if processed_path.exists() and not args.ignore_existing:
        for line in processed_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            mid = line.strip()
            if mid:
                processed_ids.add(mid)

    with sync_playwright() as p:
        context = create_browser_context(p, args)
        search_page = context.new_page()

        search_url = f"https://www.douyin.com/search/{quote(args.keyword)}"

        print(f"[OPEN SEARCH] {search_url}")
        search_page.goto(search_url, wait_until="domcontentloaded", timeout=60_000)

        if not args.no_pause:
            print()
            print("如果没有登录，请先在弹出的浏览器中登录抖音。")
            print("请手动切到你想要的结果类型，比如“图文 / 图片 / 综合”。")
            print("确认页面上能看到搜索结果卡片后，回到终端按 Enter。")
            input("Press Enter to continue...")

        try:
            search_page.wait_for_load_state("domcontentloaded", timeout=10_000)
        except PlaywrightTimeoutError:
            pass

        total_done = 0

        for scroll_idx in range(1, args.scroll_times + 1):
            if total_done >= args.target_count:
                break

            print(f"\n[SCROLL] {scroll_idx}/{args.scroll_times}")
            print(f"  已完成: {total_done}/{args.target_count}")

            points = get_visible_card_click_points(search_page, args)
            print(f"  当前屏幕候选卡片: {len(points)}")

            if args.debug:
                debug_dir = output_root / "_debug_search"
                debug_dir.mkdir(parents=True, exist_ok=True)

                (debug_dir / f"points_{scroll_idx:03d}.json").write_text(
                    json.dumps(points, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                try:
                    search_page.screenshot(path=str(debug_dir / f"screen_{scroll_idx:03d}.png"), full_page=False)
                except Exception:
                    pass

            for point_idx, point in enumerate(points, start=1):
                if total_done >= args.target_count:
                    break

                try:
                    search_page.mouse.click(point["x"], point["y"])
                    search_page.wait_for_timeout(int(args.after_click_wait_seconds * 1000))

                    modal_url = search_page.url
                    modal_id = get_modal_id(modal_url)

                    if not modal_id:
                        print(f"    [MISS] point={point_idx}, no modal_id")
                        close_modal_return_search(search_page, search_url)
                        continue

                    if modal_id in processed_ids:
                        print(f"    [SKIP] duplicate/processed modal_id={modal_id}")
                        close_modal_return_search(search_page, search_url)
                        continue

                    processed_ids.add(modal_id)

                    print(f"    [OPEN MODAL] {modal_id}")

                    result = process_open_modal(search_page, modal_url, args)
                    result["source"] = "search_click"
                    result["scroll_idx"] = scroll_idx
                    result["point_idx"] = point_idx
                    write_jsonl(summary_path, result)

                    total_done += 1

                    with processed_path.open("w", encoding="utf-8") as f:
                        for mid in sorted(processed_ids):
                            f.write(mid + "\n")

                    close_modal_return_search(search_page, search_url)

                except Exception as e:
                    print(f"    [ERROR] {type(e).__name__}: {e}")
                    write_jsonl(
                        summary_path,
                        {
                            "source": "search_click",
                            "scroll_idx": scroll_idx,
                            "point_idx": point_idx,
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )

                    try:
                        close_modal_return_search(search_page, search_url)
                    except Exception:
                        pass

            if total_done >= args.target_count:
                break

            try:
                search_page.mouse.wheel(0, args.search_scroll_pixels)
                search_page.wait_for_timeout(int(args.sleep_seconds * 1000))
            except PlaywrightError as e:
                print(f"[WARN] 搜索页滚动失败: {e}")
                search_page.wait_for_timeout(2000)

        context.close()

    print()
    print("[ALL DONE]")
    print(f"本次完成 modal 组数: {total_done}")
    print(f"输出目录: {output_root}")
    print(f"汇总: {summary_path}")


def download_from_modal_urls_file(args):
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    processed_path = output_root / "processed_modal_ids.txt"
    summary_path = output_root / "batch_summary.jsonl"

    processed_ids = set()

    if processed_path.exists() and not args.ignore_existing:
        for line in processed_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            mid = line.strip()
            if mid:
                processed_ids.add(mid)

    modal_urls = load_modal_urls_from_file(args.modal_urls_file, args.target_count)

    print(f"[INFO] 从文件读取 modal URL: {len(modal_urls)}")

    with sync_playwright() as p:
        context = create_browser_context(p, args)

        total_done = 0

        for index, modal_url in enumerate(modal_urls, start=1):
            if total_done >= args.target_count:
                break

            modal_id = get_modal_id(modal_url)

            if not modal_id:
                continue

            if modal_id in processed_ids:
                print(f"[SKIP] {index}/{len(modal_urls)} processed modal_id={modal_id}")
                continue

            print(f"\n[MODAL FILE] {index}/{len(modal_urls)} modal_id={modal_id}")

            page = context.new_page()

            try:
                page.goto(modal_url, wait_until="domcontentloaded", timeout=60_000)

                if index == 1 and not args.no_pause:
                    print()
                    print("如果没有登录，请先在弹出的浏览器中登录抖音。")
                    print("确认帖子弹窗打开后，回到终端按 Enter。")
                    input("Press Enter to continue...")

                page.wait_for_timeout(int(args.initial_wait_seconds * 1000))

                result = process_open_modal(page, modal_url, args)
                result["source"] = "modal_urls_file"
                result["file_index"] = index
                write_jsonl(summary_path, result)

                processed_ids.add(modal_id)
                total_done += 1

                with processed_path.open("w", encoding="utf-8") as f:
                    for mid in sorted(processed_ids):
                        f.write(mid + "\n")

            except Exception as e:
                print(f"[ERROR] modal_id={modal_id} {type(e).__name__}: {e}")
                write_jsonl(
                    summary_path,
                    {
                        "source": "modal_urls_file",
                        "file_index": index,
                        "modal_id": modal_id,
                        "modal_url": modal_url,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )

            finally:
                try:
                    page.close()
                except Exception:
                    pass

        context.close()

    print()
    print("[ALL DONE]")
    print(f"本次完成 modal 组数: {total_done}")
    print(f"输出目录: {output_root}")
    print(f"汇总: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="批量下载 100 组抖音 modal_id 图文图片，基于单帖网络响应捕获逻辑"
    )

    parser.add_argument("--keyword", type=str, default="双人合照")
    parser.add_argument("--output_dir", type=str, default="./douyin_100_modal_images/双人合照")
    parser.add_argument("--target_count", type=int, default=100)

    # 如果传这个，就不从搜索页点击，而是直接按文件里的 modal_url / modal_id 批量下载
    parser.add_argument("--modal_urls_file", type=str, default=None)

    parser.add_argument("--user_data_dir", type=str, default="./chrome_douyin_profile")
    parser.add_argument("--channel", type=str, default="chrome")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_pause", action="store_true")

    parser.add_argument("--viewport_width", type=int, default=1400)
    parser.add_argument("--viewport_height", type=int, default=900)

    # 搜索页点击参数
    parser.add_argument("--scroll_times", type=int, default=120)
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    parser.add_argument("--search_scroll_pixels", type=int, default=2600)
    parser.add_argument("--card_min_width", type=int, default=120)
    parser.add_argument("--card_min_height", type=int, default=120)
    parser.add_argument("--card_max_width", type=int, default=520)
    parser.add_argument("--card_max_height", type=int, default=760)
    parser.add_argument("--card_min_x", type=int, default=160)
    parser.add_argument("--card_min_y", type=int, default=80)
    parser.add_argument("--max_clicks_per_scroll", type=int, default=18)
    parser.add_argument("--after_click_wait_seconds", type=float, default=1.3)

    # 单帖下载参数，基本沿用你上传的单帖脚本
    parser.add_argument("--initial_wait_seconds", type=float, default=3.0)
    parser.add_argument("--step_wait_seconds", type=float, default=1.0)
    parser.add_argument("--capture_wait_seconds", type=float, default=0.8)
    parser.add_argument("--expected_count", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--stop_after_no_new", type=int, default=8)

    # 内容图过滤；不是最终质量筛选，只是排除头像/图标/小图
    parser.add_argument("--min_bytes", type=int, default=50_000)
    parser.add_argument("--min_width", type=int, default=400)
    parser.add_argument("--min_height", type=int, default=400)
    parser.add_argument("--min_long_side", type=int, default=700)
    parser.add_argument("--min_short_side", type=int, default=400)
    parser.add_argument("--max_aspect_ratio", type=float, default=3.5)

    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--save_rejected", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")

    args = parser.parse_args()

    if args.modal_urls_file:
        download_from_modal_urls_file(args)
    else:
        collect_and_download_from_search(args)


if __name__ == "__main__":
    main()


'''
python 04_download_100_douyin_modal_images_network.py `
  --keyword 双人合照 `
  --output_dir .images/douyin_100_modal_images/双人合照 `
  --target_count 100

'''