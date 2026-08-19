#!/usr/bin/env python3
"""用 qwen3.7-plus 为照片打标并按结果分流。"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
CATEGORY_DIRS = {
    "single": Path("单人照"), "two": Path("双人照"),
    "multiple": Path("多人照"), "others": Path("others"),
}
SINGLE_DIRS = {
    "close_selfie": "自拍近景", "half_body": "半身照", "full_body": "全身照",
}

PROMPT = r"""
你是严谨的照片分类员。观察输入图片，识别“主人物”，只输出一个合法 JSON 对象。

第一层 category 只能是：
- single：恰好 1 个主人物。
- two：恰好 2 个主人物。
- multiple：3 个或更多主人物。
- others：无主人物的风景、场景、物品或背景照；多张图片拼接的拼图/组合图；表情包、聊天截图、纯文字图、海报等不适合按真实人物数分类的图片。

规则：
- 按构图意图识别主人物。远处、边缘、偶然入镜、明显作为背景的路人不计入主人物。
- 合影参与者都算主人物。镜面中的同一个人，或照片、屏幕、广告牌里的人像，不重复计数。
- 主人物只露出身体一部分仍算一个人。
- 多图拼接、宫格、前后对比拼图，无论其中有几个人，一律为 others。
- 表情包、网络梗图、聊天截图、文字主体图片一律为 others。
- 动漫、插画、玩偶、雕塑等非现实人物图片归为 others。

仅当 category=single 时设置 single_type：
- close_selfie：自拍近景；有自拍视角，或非常近的脸部、头肩、胸部以上人像。普通头肩近照即使自拍线索不明显也归此类。
- half_body：半身照；不是近景，通常从头到腰、胯或大腿附近，膝盖以下未完整呈现。
- full_body：全身照；从头到脚基本完整可见。脚部因边缘轻微裁切但显然是全身构图，也归此类。

边界：非 single 时 single_type 必须是 null。单人照的脸很近或明显本人手持拍摄时优先 close_selfie；展示至腰/大腿归 half_body；只有头脚基本都可见才归 full_body。不要创造新标签。

严格输出：
{
  "category": "single|two|multiple|others",
  "single_type": "close_selfie|half_body|full_body|null",
  "main_person_count": 0,
  "confidence": 0.0,
  "reason": "简短中文理由"
}
main_person_count 是主人物数量；拼图、表情包或无法可靠计数可填 0。confidence 范围 0 到 1。
不要输出 Markdown、代码围栏或 JSON 之外的内容。
""".strip()


class ClassifyError(RuntimeError):
    pass


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_images(input_dir: Path, output_dir: Path, limit: int | None) -> list[Path]:
    output_root = output_dir.resolve()
    paths = [p for p in sorted(input_dir.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS
             and not is_under(p.resolve(), output_root)]
    return paths[:limit] if limit is not None else paths


def response_text(response: Any) -> str:
    status = getattr(response, "status_code", 200)
    if status != 200:
        raise ClassifyError(
            f"DashScope 请求失败：status={status}, code={getattr(response, 'code', 'unknown')}, "
            f"message={getattr(response, 'message', repr(response))}"
        )
    try:
        content = response.output.choices[0].message.content
        if isinstance(content, str):
            text = content
        else:
            text = next(x["text"] for x in content if isinstance(x, dict) and "text" in x)
    except Exception as exc:
        raise ClassifyError(f"无法读取 DashScope 响应：{response!r}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ClassifyError("DashScope 返回空文本")
    return text.strip()


def parse_result(text: str) -> dict[str, Any]:
    # JSON 模式通常不会返回围栏；这里对偶发的 ```json ... ``` 做兼容。
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClassifyError(f"模型输出不是合法 JSON：{text[:300]}") from exc
    if not isinstance(value, dict):
        raise ClassifyError("模型输出必须为 JSON 对象")

    category = value.get("category")
    subtype = value.get("single_type")
    if category not in CATEGORY_DIRS:
        raise ClassifyError(f"非法 category：{category!r}")
    if category == "single" and subtype not in SINGLE_DIRS:
        raise ClassifyError(f"单人照的 single_type 非法：{subtype!r}")
    if category != "single":
        subtype = None

    count = value.get("main_person_count", 0)
    count = max(0, int(count)) if isinstance(count, (int, float)) and not isinstance(count, bool) else 0
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = value.get("reason", "")
    return {
        "category": category, "single_type": subtype,
        "main_person_count": count, "confidence": confidence,
        "reason": reason.strip() if isinstance(reason, str) else str(reason),
    }


def classify(path: Path, *, model: str, api_key: str, attempts: int) -> dict[str, Any]:
    import dashscope
    messages = [
        {"role": "system", "content": [{"text": PROMPT}]},
        {"role": "user", "content": [
            {"image": image_to_data_url(path)},
            {"text": "请分析图片，并严格按照指定 JSON 格式输出分类结果。"},
        ]},
    ]
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=api_key, model=model, messages=messages,
                response_format={"type": "json_object"},
                enable_thinking=False, result_format="message",
            )
            return {
                **parse_result(response_text(response)), "attempt": attempt,
                "request_id": getattr(response, "request_id", None),
            }
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(8.0, 2 ** (attempt - 1) + random.random()))
    raise ClassifyError(f"尝试 {attempts} 次后仍失败：{last_error}")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def route_image(source: Path, input_dir: Path, output_dir: Path,
                result: dict[str, Any], mode: str) -> Path:
    folder = output_dir / CATEGORY_DIRS[result["category"]]
    if result["category"] == "single":
        folder /= SINGLE_DIRS[result["single_type"]]
    # 保留输入目录的子目录层次，避免同名图片互相覆盖。
    target = unique_path(folder / source.relative_to(input_dir).parent / source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "move":
        shutil.move(str(source), str(target))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="待分类图片目录")
    parser.add_argument("--output-dir", default="classified_photos", help="分类输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("copy", "move", "none"), default="copy",
                        help="复制分流、移动分流或只打标签（默认 copy）")
    parser.add_argument("--workers", type=int, default=5, help="并发请求数")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, help="只处理前 N 张，用于测试")
    parser.add_argument("--resume", action="store_true", help="跳过已有标签的图片")
    parser.add_argument("--dashscope-base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1:
        parser.error("--workers 和 --attempts 必须大于 0")

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_ALIBABA_API_KEY")
    if not api_key:
        raise SystemExit("请先设置 DASHSCOPE_API_KEY（也兼容 DASHSCOPE_ALIBABA_API_KEY）")
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")
    if input_dir == output_dir:
        raise SystemExit("输入目录与输出目录不能相同")

    import dashscope
    dashscope.base_http_api_url = args.dashscope_base_url
    labels_dir = output_dir / "_labels"
    images = discover_images(input_dir, output_dir, args.limit)
    pending: list[tuple[Path, Path]] = []
    for image in images:
        relative = image.relative_to(input_dir)
        label = labels_dir / relative.parent / f"{relative.name}.json"
        if not (args.resume and label.exists()):
            pending.append((image, label))

    print(f"发现 {len(images)} 张；待处理 {len(pending)} 张；模型 {args.model}")
    ok = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(classify, image, model=args.model, api_key=api_key,
                        attempts=args.attempts): (image, label)
            for image, label in pending
        }
        for future in as_completed(futures):
            image, label = futures[future]
            try:
                result = future.result()
                routed = None if args.mode == "none" else route_image(
                    image, input_dir, output_dir, result, args.mode)
                record = {
                    "image_file": str(image), "routed_file": str(routed) if routed else None,
                    "model": args.model, **result,
                }
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                append_jsonl(output_dir / "labels.jsonl", record)
                ok += 1
                detail = f"/{result['single_type']}" if result["single_type"] else ""
                print(f"[OK {ok + failed}/{len(pending)}] {image.name} -> {result['category']}{detail}")
            except Exception as exc:
                failed += 1
                append_jsonl(output_dir / "errors.jsonl", {"image_file": str(image), "error": str(exc)})
                print(f"[ERR {ok + failed}/{len(pending)}] {image.name}: {exc}", file=sys.stderr)
    print(f"完成：成功 {ok}，失败 {failed}。输出目录：{output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
