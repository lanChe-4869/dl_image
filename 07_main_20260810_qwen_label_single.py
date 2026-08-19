#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

LABEL_PROMPT = r"""
你是一个单人照片动作与表情打标助手。请观察输入的单人目标图片，提取适合用于“单人动作编辑”的姿态与表情标签。

任务要求：

1. 图片中包含一个人，请按照观看者视角描述人物的动作姿态。
2. 输出只包含一个 JSON 对象，字段必须为："pose"、"expression"。
3. "pose" 字段需要描述人物的整体动作姿态，包括：人物在画面中的站姿或坐姿；身体朝向与倾斜方向；头部朝向与倾斜状态；画面左侧手臂和画面右侧手臂分别在做什么姿势；整体动作呈现出的状态。
4. 手臂描述需要简洁，但必须分别说明画面左侧手臂和画面右侧手臂的大致动作，例如“画面左侧手臂抬起弯曲”“画面右侧手臂自然下垂”等。
5. 如果画面出现腿部信息，则必须说明腿部的大致动作，例如双腿站立、单腿弯曲、双腿交叉、迈步等；如果腿部不可见，则不需要描述。
6. 重点保留对后续单人动作编辑有帮助的姿态信息，不需要详细描述手指、服装、背景、道具等细节。
7. "expression" 字段描述人物的面部表情，重点包括微笑程度、嘴部状态、眼睛状态以及整体情绪氛围。
8. 对图片中无法明确判断的动作或表情不要过度推测，只描述能够观察到的信息。
9. 不要推测人物身份、姓名、年龄、职业等信息。
10. 输出必须是严格合法的 JSON，不要输出 Markdown、解释文字或额外内容。
11. 控制 "pose" 的长度，描述简洁清晰，避免重复信息，并确保可直接用于后续动作编辑。

输出格式：
{
"pose": "人物……",
"expression": "人物……"
}

例如某个单人图片的标签结果如下，控制好 pose 的字数：
{
"pose": "人物站立并面向镜头，身体微微向画面右侧倾斜，头部轻微向左侧歪斜，画面左侧手臂抬起并弯曲，画面右侧手臂自然下垂，双腿自然站立，整体呈现轻松活泼的姿态。",
"expression": "人物面向镜头露出明显笑容，眼睛微微眯起，整体表情明亮开心。"
}
""".strip()

USER_TASK = "请根据规则标注这张单人照片，并只输出包含 pose 和 expression 的合法 JSON 对象。"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class LabelError(RuntimeError):
    pass


def image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def discover_images(input_dir: Path, limit: int | None = None) -> list[Path]:
    paths = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if limit is not None:
        paths = paths[:limit]
    return paths


def normalize_separator(text: str) -> str:
    text = text.strip()

    # 把英文分号替换成中文分号
    text = text.replace(";", "；")

    # 常见情况：模型用了句号分隔左右人物
    if "；" not in text:
        markers = [
            "右侧人物",
            "右边人物",
            "右侧的人物",
            "画面右侧人物",
            "画面右边人物",
        ]

        for marker in markers:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx].rstrip("，。； ") + "；" + text[idx:].lstrip("，。； ")
                break

    return text


def validate_label(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise LabelError("模型输出不是 JSON 对象")

    required = {"pose", "expression"}
    if not required.issubset(value):
        raise LabelError(f"模型输出缺少字段，必须包含 {sorted(required)}")

    pose = value["pose"]
    expression = value["expression"]

    if not isinstance(pose, str) or not pose.strip():
        raise LabelError("字段 'pose' 必须是非空字符串")

    if not isinstance(expression, str) or not expression.strip():
        raise LabelError("字段 'expression' 必须是非空字符串")

    pose = normalize_separator(pose)
    expression = normalize_separator(expression)

    # 最后仍然没有分号，再兜底补一个，避免整张图失败
    if "；" not in pose:
        pose = "左侧人物" + pose if not pose.startswith("左侧人物") else pose
        pose = pose + "；右侧人物动作未被模型明确分开描述。"

    if "；" not in expression:
        expression = "左侧人物" + expression if not expression.startswith("左侧人物") else expression
        expression = expression + "；右侧人物表情未被模型明确分开描述。"

    return {
        "pose": pose.strip(),
        "expression": expression.strip(),
    }



def extract_dashscope_text(response: Any) -> str:
    status_code = getattr(response, "status_code", 200)
    if status_code != 200:
        code = getattr(response, "code", "unknown")
        message = getattr(response, "message", repr(response))
        raise LabelError(f"DashScope 请求失败：status={status_code}, code={code}, message={message}")

    try:
        content = response.output.choices[0].message.content
        if isinstance(content, list):
            text = content[0]["text"]
        else:
            text = content
    except Exception as exc:
        raise LabelError(f"无法读取 DashScope 响应文本：{response!r}") from exc

    if not isinstance(text, str):
        raise LabelError(f"DashScope 返回文本类型异常：{type(text).__name__}")

    return text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def call_one_image(
    image_path: Path,
    *,
    model: str,
    api_key: str,
    attempts: int,
) -> dict[str, Any]:
    import dashscope

    image_data_url = image_to_data_url(image_path)

    messages = [
        {
            "role": "system",
            "content": [{"text": LABEL_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"text": USER_TASK},
                {"image": image_data_url},
            ],
        },
    ]

    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                enable_thinking=False,
            )

            raw_text = extract_dashscope_text(response)
            label = validate_label(json.loads(raw_text))

            return {
                "image_id": image_path.stem,
                "image_file": str(image_path),
                **label,
                "attempt": attempt,
                "request_id": getattr(response, "request_id", None),
            }

        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(8.0, (2 ** (attempt - 1)) + random.random()))

    raise LabelError(f"{image_path.name} 在 {attempts} 次尝试后仍失败：{last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="输入图片文件夹")
    parser.add_argument("--output-dir", default="output_labels", help="输出 JSON 文件夹")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="跳过已经存在的同名 JSON")
    parser.add_argument("--dashscope-base-url", default=DEFAULT_DASHSCOPE_BASE_URL)

    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_ALIBABA_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DASHSCOPE_ALIBABA_API_KEY")

    import dashscope
    dashscope.base_http_api_url = args.dashscope_base_url

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_jsonl = output_dir / "labels.jsonl"
    errors_jsonl = output_dir / "errors.jsonl"

    images = discover_images(input_dir, args.limit)

    pending: list[Path] = []
    for image_path in images:
        output_path = output_dir / f"{image_path.stem}.json"
        if args.resume and output_path.exists():
            continue
        pending.append(image_path)

    print(f"发现图片 {len(images)} 张；待处理 {len(pending)} 张。")

    ok = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                call_one_image,
                image_path,
                model=args.model,
                api_key=api_key,
                attempts=args.attempts,
            ): image_path
            for image_path in pending
        }

        for future in as_completed(futures):
            image_path = futures[future]
            try:
                result = future.result()

                output_path = output_dir / f"{image_path.stem}.json"
                write_json(output_path, {
                    "pose": result["pose"],
                    "expression": result["expression"],
                })

                append_jsonl(labels_jsonl, result)

                ok += 1
                print(f"[OK {ok + failed}/{len(pending)}] {image_path.name}")

            except Exception as exc:
                failed += 1
                error = {
                    "image_file": str(image_path),
                    "error": str(exc),
                }
                append_jsonl(errors_jsonl, error)
                print(f"[ERR {ok + failed}/{len(pending)}] {image_path.name}: {exc}", file=sys.stderr)

    print(f"完成：成功 {ok}，失败 {failed}。输出目录：{output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

'''
python main_20260810_qwen_label_single.py `
 E:\workspace\vsCode\photo-pre\data\06_single\00_origin_2K `
 --output-dir E:\workspace\vsCode\photo-pre\data\06_single\05_label_json `
 --workers 30 --resume

output_json/
├── 001.json
├── 002.json
├── labels.jsonl
└── errors.jsonl

{
  "pose": "...",
  "expression": "..."
}


'''