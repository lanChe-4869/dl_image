# 07_click_collect_douyin_model_ids.py
# 目标：
#   根据关键词打开抖音搜索页，通过“点击可见卡片”的方式获取每个帖子的 model_id，
#   然后写入 txt 文件。这个脚本只采集 ID，不下载图片。
#
# 为什么要点击：
#   抖音搜索页很多卡片不会在 DOM 中直接暴露 note/aweme 链接；
#   点击卡片后，URL 通常会变成：
#       https://www.douyin.com/search/关键词?modal_id=7169...
#   或者跳转到：
#       https://www.douyin.com/note/7169...
#       https://www.douyin.com/video/7169...
#   这时才能稳定拿到 model_id。
#
# 安装：
#   pip install playwright
#   playwright install chromium
#
# 小规模测试：
#   python 07_click_collect_douyin_model_ids.py ^
#     --keyword 双人合照 ^
#     --output_txt ./douyin_model_ids/双人合照/model_ids.txt ^
#     --target_count 20 ^
#     --debug
#
# 正式运行：
#   python 07_click_collect_douyin_model_ids.py ^
#     --keyword 双人合照 ^
#     --output_txt ./douyin_model_ids/双人合照/model_ids.txt ^
#     --target_count 100

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse, parse_qs

from playwright.sync_api import sync_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError


ID_RE = re.compile(r"(?<!\d)(\d{16,22})(?!\d)")


def extract_model_id(url: str) -> str:
    """从点击后的 URL 中提取 model_id。优先 modal_id，其次 /note/id、/video/id、/share/video/id。"""
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)

        for key in ("modal_id", "aweme_id", "item_id", "note_id", "group_id"):
            values = qs.get(key)
            if values and values[0] and values[0].isdigit():
                return values[0].strip()

        path = parsed.path or ""
        patterns = [
            r"/(?:note|video)/(\d{16,22})(?:/|$)",
            r"/share/(?:note|video)/(\d{16,22})(?:/|$)",
            r"/discover/?(?:\?|$).*",  # fallback below only
        ]
        for pat in patterns[:2]:
            m = re.search(pat, path)
            if m:
                return m.group(1)

        # 最后兜底：URL 任意位置出现的 16-22 位数字
        m = ID_RE.search(url)
        if m:
            return m.group(1)
    except Exception:
        pass

    return ""


def normalize_note_url(model_id: str) -> str:
    return f"https://www.douyin.com/note/{model_id}"


def create_browser_context(playwright, args):
    kwargs = {
        "user_data_dir": args.user_data_dir,
        "headless": args.headless,
        "viewport": {"width": args.viewport_width, "height": args.viewport_height},
    }
    if args.channel:
        kwargs["channel"] = args.channel
    return playwright.chromium.launch_persistent_context(**kwargs)


def write_ids(output_txt: Path, ids: list[str], write_urls: bool = False):
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    if write_urls:
        text = "\n".join(normalize_note_url(x) for x in ids)
    else:
        text = "\n".join(ids)
    if text:
        text += "\n"
    output_txt.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_ids(output_txt: Path) -> list[str]:
    if not output_txt.exists():
        return []
    out = []
    seen = set()
    for line in output_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        mid = extract_model_id(line.strip()) or line.strip()
        if mid and mid.isdigit() and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def is_search_like_url(url: str) -> bool:
    lower = (url or "").lower()
    return "/search/" in lower or "douyin.com/search" in lower


def close_back_to_search(page, search_url: str, args):
    """关闭弹窗或回到搜索页。"""
    for _ in range(2):
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(350)
        except Exception:
            pass

        current = page.url
        if is_search_like_url(current) and not extract_model_id(current):
            return

        # modal_id 可能还在 query 里，go_back 通常能回搜索列表
        try:
            page.go_back(wait_until="domcontentloaded", timeout=5000)
            page.wait_for_timeout(700)
        except Exception:
            pass

        current = page.url
        if is_search_like_url(current) and not extract_model_id(current):
            return

    # 兜底：直接回搜索页
    try:
        page.goto(search_url, wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_timeout(int(args.after_return_wait_seconds * 1000))
    except Exception:
        pass


def get_visible_card_click_points(page, args) -> list[dict]:
    """基本沿用你上传脚本里的卡片定位逻辑：从 img/video/canvas/background-image 往父级找可见卡片。"""
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
            return {left:x1, top:y1, right:x2, bottom:y2, width:w, height:h};
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
                "观看历史", "稍后再看", "我的作品", "我的预约", "搜索", "消息"
            ];
            return bad.some(x => t.includes(x));
        }

        const candidates = [];
        const baseNodes = Array.from(document.querySelectorAll(
            "img, video, canvas, [style*='background-image']"
        ));

        for (const base of baseNodes) {
            let el = base;
            let chosen = null;

            for (let depth = 0; depth < opts.parentDepth && el; depth++, el = el.parentElement) {
                const r = visibleRect(el);
                if (!r) continue;

                if (
                    r.width >= minW && r.height >= minH &&
                    r.width <= maxW && r.height <= maxH &&
                    r.left >= minX && r.top >= minY &&
                    hasVisual(el)
                ) {
                    chosen = {el, r};
                    if (r.width > maxW * 0.85 || r.height > maxH * 0.85) break;
                }
            }

            if (!chosen) continue;
            const card = chosen.el;
            const r = chosen.r;
            const text = (card.innerText || card.textContent || "").trim().slice(0, 180);
            if (isBadText(text)) continue;

            const x = Math.round(r.left + r.width / 2);
            const y = Math.round(r.top + Math.min(r.height * 0.45, r.height / 2));
            if (x < minX || y < minY || x > vw - 20 || y > vh - 20) continue;

            candidates.push({
                x, y,
                w: Math.round(r.width), h: Math.round(r.height),
                left: Math.round(r.left), top: Math.round(r.top),
                tag: card.tagName, text
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
        deduped.sort((a,b) => (a.top - b.top) || (a.left - b.left));
        return deduped.slice(0, opts.maxPoints);
    }
    """
    try:
        return page.evaluate(js, {
            "minW": args.card_min_width,
            "minH": args.card_min_height,
            "maxW": args.card_max_width,
            "maxH": args.card_max_height,
            "minX": args.card_min_x,
            "minY": args.card_min_y,
            "maxPoints": args.max_clicks_per_scroll,
            "parentDepth": args.parent_depth,
        }) or []
    except PlaywrightError as e:
        print(f"[WARN] 获取卡片点击点失败: {e}")
        return []


def click_point_and_get_id(page, point: dict, args) -> tuple[str, str, str]:
    """点击一个点，返回 (model_id, url, status)。"""
    before_url = page.url
    try:
        page.mouse.click(point["x"], point["y"])
    except Exception as e:
        return "", page.url, f"click_error:{type(e).__name__}"

    page.wait_for_timeout(int(args.after_click_wait_seconds * 1000))

    # 有些点击需要等 URL 变化/弹窗加载；多轮检查
    final_url = page.url
    model_id = extract_model_id(final_url)
    if model_id:
        return model_id, final_url, "ok_url"

    # 兜底：如果页面内容中出现 /note/id 或 modal_id，也尝试读一下
    try:
        html = page.content()
        model_id = extract_model_id(html)
        if model_id:
            return model_id, final_url, "ok_html_fallback"
    except Exception:
        pass

    if final_url == before_url:
        return "", final_url, "no_url_change"
    return "", final_url, "no_id_after_click"


def collect_by_click(args):
    output_txt = Path(args.output_txt)
    debug_dir = output_txt.parent / "_debug_collect_ids"
    log_path = output_txt.parent / "collect_model_ids_log.jsonl"

    ids = [] if args.ignore_existing else load_existing_ids(output_txt)
    seen = set(ids)
    write_ids(output_txt, ids, args.write_urls)

    with sync_playwright() as p:
        context = create_browser_context(p, args)
        page = context.new_page()
        search_url = f"https://www.douyin.com/search/{quote(args.keyword)}"

        print(f"[OPEN SEARCH] {search_url}")
        page.goto(search_url, wait_until="domcontentloaded", timeout=60_000)

        if not args.no_pause:
            print()
            print("如果没有登录，请先在弹出的浏览器中登录抖音。")
            print("请手动切到你想要的结果类型，比如：图文 / 图片 / 综合。")
            print("确认页面上能看到搜索结果卡片后，回到终端按 Enter。")
            input("Press Enter to continue...")

        try:
            page.wait_for_load_state("domcontentloaded", timeout=10_000)
        except PlaywrightTimeoutError:
            pass

        miss_rounds = 0

        for scroll_idx in range(1, args.scroll_times + 1):
            if len(ids) >= args.target_count:
                break

            print(f"\n[SCROLL] {scroll_idx}/{args.scroll_times} 已采集 {len(ids)}/{args.target_count}")
            points = get_visible_card_click_points(page, args)
            print(f"  当前屏幕候选卡片: {len(points)}")

            if args.debug:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"points_{scroll_idx:03d}.json").write_text(
                    json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                try:
                    page.screenshot(path=str(debug_dir / f"screen_{scroll_idx:03d}.png"), full_page=False)
                except Exception:
                    pass

            found_this_scroll = 0

            for point_idx, point in enumerate(points, start=1):
                if len(ids) >= args.target_count:
                    break

                model_id, opened_url, status = click_point_and_get_id(page, point, args)

                row = {
                    "time": time.time(),
                    "keyword": args.keyword,
                    "scroll_idx": scroll_idx,
                    "point_idx": point_idx,
                    "point": point,
                    "status": status,
                    "model_id": model_id,
                    "opened_url": opened_url,
                }
                append_jsonl(log_path, row)

                if model_id and model_id not in seen:
                    seen.add(model_id)
                    ids.append(model_id)
                    found_this_scroll += 1
                    write_ids(output_txt, ids, args.write_urls)
                    print(f"    [FOUND] {len(ids):03d}/{args.target_count} model_id={model_id}")
                elif model_id:
                    print(f"    [SKIP] duplicate model_id={model_id}")
                else:
                    print(f"    [MISS] point={point_idx} status={status}")

                close_back_to_search(page, search_url, args)

            if found_this_scroll == 0:
                miss_rounds += 1
            else:
                miss_rounds = 0

            if miss_rounds >= args.stop_after_empty_scrolls:
                print(f"[STOP] 连续 {miss_rounds} 屏没有采集到新 ID")
                break

            if len(ids) >= args.target_count:
                break

            try:
                page.mouse.wheel(0, args.search_scroll_pixels)
                page.wait_for_timeout(int(args.sleep_seconds * 1000))
            except PlaywrightError as e:
                print(f"[WARN] 搜索页滚动失败: {e}")
                page.wait_for_timeout(2000)

        context.close()

    print()
    print("[ALL DONE]")
    print(f"采集到 model_id 数量: {len(ids)}")
    print(f"输出 TXT: {output_txt}")
    print(f"日志 JSONL: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="通过点击抖音搜索页卡片采集多个帖子的 model_id 到 txt")

    parser.add_argument("--keyword", type=str, default="双人合照")
    parser.add_argument("--output_txt", type=str, default="./douyin_model_ids/双人合照/model_ids.txt")
    parser.add_argument("--target_count", type=int, default=100)
    parser.add_argument("--write_urls", action="store_true", help="写入 https://www.douyin.com/note/{model_id}，否则只写 model_id")

    parser.add_argument("--user_data_dir", type=str, default="./chrome_douyin_profile")
    parser.add_argument("--channel", type=str, default="chrome")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_pause", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")

    parser.add_argument("--viewport_width", type=int, default=1400)
    parser.add_argument("--viewport_height", type=int, default=900)

    # 搜索页点击参数：默认值基本参考你上传的批量下载脚本
    parser.add_argument("--scroll_times", type=int, default=120)
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    parser.add_argument("--search_scroll_pixels", type=int, default=2600)
    parser.add_argument("--card_min_width", type=int, default=120)
    parser.add_argument("--card_min_height", type=int, default=120)
    parser.add_argument("--card_max_width", type=int, default=520)
    parser.add_argument("--card_max_height", type=int, default=760)
    parser.add_argument("--card_min_x", type=int, default=160)
    parser.add_argument("--card_min_y", type=int, default=80)
    parser.add_argument("--parent_depth", type=int, default=6)
    parser.add_argument("--max_clicks_per_scroll", type=int, default=18)
    parser.add_argument("--after_click_wait_seconds", type=float, default=1.5)
    parser.add_argument("--after_return_wait_seconds", type=float, default=0.8)
    parser.add_argument("--stop_after_empty_scrolls", type=int, default=10)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    collect_by_click(args)


if __name__ == "__main__":
    main()

'''
python 05_click_collect_douyin_model_ids.py `
  --keyword 写真 `
  --output_txt ./images/0615_model_ids.txt `
  --target_count 100 `
  --debug

'''