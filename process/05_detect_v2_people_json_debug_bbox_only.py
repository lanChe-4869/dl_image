import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Detection:
    name: str
    cls_id: int
    conf: float
    box: List[float]
    mask_box: Optional[List[float]] = None

    @property
    def safe_box(self) -> List[float]:
        # bbox-only mode: 为了后续 pose 关键点提取，所有主体筛选、JSON 输出、debug 主框都只使用 YOLO bbox。
        # 不再把 segmentation mask 外接框合并进来，避免框被 mask_box 放大。
        return self.box


def clamp(v, low, high):
    return max(low, min(high, v))


def round_box(box, ndigits=2):
    return [round(float(v), ndigits) for v in box]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def box_iou(a, b):
    inter = intersection_area(a, b)
    denom = box_area(a) + box_area(b) - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def box_intersection_over_min_area(a, b):
    inter = intersection_area(a, b)
    min_area = min(box_area(a), box_area(b))
    if min_area <= 0:
        return 0.0
    return inter / min_area


def union_boxes(boxes: Sequence[Sequence[float]]) -> List[float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def vertical_overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    min_h = max(1.0, min(ay2 - ay1, by2 - by1))
    return inter_h / min_h


def normalized_person_center_distance(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)

    aw = max(1.0, a[2] - a[0])
    ah = max(1.0, a[3] - a[1])
    bw = max(1.0, b[2] - b[0])
    bh = max(1.0, b[3] - b[1])

    scale = max(1.0, min(aw, bw) + min(ah, bh))
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / scale


def are_duplicate_person_boxes(a, b, iou_thres=0.95, cover_thres=1.10):
    return box_iou(a, b) >= iou_thres or box_intersection_over_min_area(a, b) >= cover_thres


def merge_detection_cluster(cluster: Sequence[Detection]) -> Detection:
    best = max(cluster, key=lambda d: d.conf)
    merged_box = union_boxes([d.box for d in cluster])
    mask_boxes = [d.mask_box for d in cluster if d.mask_box is not None]
    merged_mask_box = union_boxes(mask_boxes) if mask_boxes else None
    return Detection(
        name=best.name,
        cls_id=best.cls_id,
        conf=max(d.conf for d in cluster),
        box=merged_box,
        mask_box=merged_mask_box,
    )


def deduplicate_person_detections(person_dets: Sequence[Detection], iou_thres=0.95, cover_thres=1.10) -> List[Detection]:
    clusters: List[List[Detection]] = []
    sorted_dets = sorted(person_dets, key=lambda d: (box_area(d.safe_box), d.conf), reverse=True)

    for det in sorted_dets:
        det_box = det.safe_box
        placed = False
        for cluster in clusters:
            if any(
                are_duplicate_person_boxes(
                    det_box,
                    member.safe_box,
                    iou_thres=iou_thres,
                    cover_thres=cover_thres,
                )
                for member in cluster
            ):
                cluster.append(det)
                placed = True
                break
        if not placed:
            clusters.append([det])

    unique_dets = [merge_detection_cluster(cluster) for cluster in clusters]
    unique_dets.sort(key=lambda d: box_area(d.safe_box), reverse=True)
    return unique_dets


def is_likely_occluded_second_person(
    det_box,
    largest_box,
    img_w,
    img_h,
    min_abs_area_ratio=0.002,
    max_center_distance=1.8,
    min_vertical_overlap=0.25,
    max_bottom_gap_ratio=0.45,
):
    img_area = max(1.0, float(img_w * img_h))
    area = box_area(det_box)
    if area / img_area < min_abs_area_ratio:
        return False

    center_dist = normalized_person_center_distance(det_box, largest_box)
    v_overlap = vertical_overlap_ratio(det_box, largest_box)

    det_h = max(1.0, det_box[3] - det_box[1])
    largest_h = max(1.0, largest_box[3] - largest_box[1])
    bottom_gap = abs(det_box[3] - largest_box[3]) / max(1.0, min(det_h, largest_h))

    if center_dist <= max_center_distance and v_overlap >= min_vertical_overlap:
        return True
    if center_dist <= max_center_distance * 0.85 and bottom_gap <= max_bottom_gap_ratio:
        return True
    return False


def select_main_person_detections(
    person_dets: Sequence[Detection],
    img_w: int,
    img_h: int,
    required_people: int = 2,
    top_k: int = 2,
    min_area_ratio_to_largest: float = 0.12,
    min_abs_area_ratio: float = 0.004,
    allow_occluded_second_person: bool = True,
    occluded_min_abs_area_ratio: float = 0.002,
    occluded_max_center_distance: float = 1.8,
    occluded_min_vertical_overlap: float = 0.25,
) -> Tuple[bool, List[Detection], str]:
    if required_people <= 0:
        required_people = 2

    if len(person_dets) < required_people:
        return False, list(person_dets), f"not_enough_unique_people_unique{len(person_dets)}"

    img_area = max(1.0, float(img_w * img_h))
    sorted_dets = sorted(person_dets, key=lambda d: box_area(d.safe_box), reverse=True)

    largest_det = sorted_dets[0]
    largest_box = largest_det.safe_box
    largest_area = box_area(largest_box)
    if largest_area <= 0:
        return False, [], "largest_person_area_is_zero"

    main_candidates = []
    notes = []

    for idx, det in enumerate(sorted_dets):
        box = det.safe_box
        area = box_area(box)

        if idx == 0:
            main_candidates.append(det)
            continue

        if area / img_area >= min_abs_area_ratio and area >= largest_area * min_area_ratio_to_largest:
            main_candidates.append(det)
            continue

        if allow_occluded_second_person and is_likely_occluded_second_person(
            det_box=box,
            largest_box=largest_box,
            img_w=img_w,
            img_h=img_h,
            min_abs_area_ratio=occluded_min_abs_area_ratio,
            max_center_distance=occluded_max_center_distance,
            min_vertical_overlap=occluded_min_vertical_overlap,
        ):
            main_candidates.append(det)
            notes.append("second_person_may_be_occluded_or_partial")
            continue

    if len(main_candidates) < required_people:
        return False, main_candidates, f"not_enough_main_people_candidates{len(main_candidates)}_unique{len(person_dets)}"

    if top_k > 0:
        main_candidates = main_candidates[:top_k]

    reason = "ok"
    if notes:
        reason += "_" + "_".join(sorted(set(notes)))
    return True, main_candidates, reason


def polygon_to_box(poly):
    if poly is None or len(poly) == 0:
        return None
    arr = np.asarray(poly)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    xs = arr[:, 0]
    ys = arr[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def get_detections(model, image_rgb, conf_thres, imgsz=None, device=None) -> List[Detection]:
    predict_kwargs = {"source": image_rgb, "conf": conf_thres, "verbose": False}
    if imgsz is not None and imgsz > 0:
        predict_kwargs["imgsz"] = imgsz
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    if not results:
        return []

    result = results[0]
    if result.boxes is None:
        return []

    names = result.names or {}
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    # bbox-only mode: 不读取 segmentation mask。
    # 即使使用 *-seg 模型，也只输出 YOLO boxes.xyxy，避免 mask 外接框影响人体框。
    masks_xy = None

    detections = []
    for i, (b, c, s) in enumerate(zip(xyxy, cls, conf)):
        cls_id = int(c)
        name = str(names.get(cls_id, cls_id)).lower()
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]

        mask_box = None
        if masks_xy is not None and i < len(masks_xy):
            mask_box = polygon_to_box(masks_xy[i])

        detections.append(Detection(name=name, cls_id=cls_id, conf=float(s), box=[x1, y1, x2, y2], mask_box=mask_box))
    return detections


def load_image_rgb(img_path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")


def save_json(data, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_debug_image(img, out_path, quality=75):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        out_path = out_path.with_suffix(".jpg")
        suffix = ".jpg"

    if suffix in {".jpg", ".jpeg"}:
        img.save(out_path, quality=int(quality), optimize=True)
    elif suffix == ".webp":
        img.save(out_path, quality=int(quality), method=6)
    else:
        img.save(out_path, optimize=True)


def copy_file_preserve_path(src_path: Path, dst_path: Path):
    """复制文件并保留目录结构；如果目标已存在，直接覆盖，避免重复运行产生大量副本。"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_path), str(dst_path))


def draw_detection_debug_image(
    img: Image.Image,
    debug_path: Path,
    raw_person_dets: Sequence[Detection],
    unique_person_dets: Sequence[Detection],
    selected_person_dets: Sequence[Detection],
    status: str,
    notes: str,
    debug_max_side=1600,
    debug_quality=75,
):
    src_w, src_h = img.size
    scale = 1.0
    if debug_max_side is not None and debug_max_side > 0:
        longest = max(src_w, src_h)
        if longest > debug_max_side:
            scale = debug_max_side / float(longest)

    if scale < 1.0:
        canvas = img.resize((max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))), Image.Resampling.LANCZOS)
    else:
        canvas = img.copy()

    draw = ImageDraw.Draw(canvas)
    canvas_w, canvas_h = canvas.size
    selected_ids = {id(d) for d in selected_person_dets}

    font_size = 18 if max(canvas_w, canvas_h) >= 1200 else 14
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    def sx(v):
        return int(round(float(v) * scale))

    def sy(v):
        return int(round(float(v) * scale))

    def rect(box, color, width=4):
        x1, y1, x2, y2 = box
        x1 = clamp(sx(x1), 0, canvas_w - 1)
        x2 = clamp(sx(x2), 0, canvas_w - 1)
        y1 = clamp(sy(y1), 0, canvas_h - 1)
        y2 = clamp(sy(y2), 0, canvas_h - 1)
        draw_width = max(2, int(round(width)))
        for i in range(draw_width):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

    def label(text, xy, color):
        x, y = xy
        x = clamp(sx(x), 0, canvas_w - 1)
        y = clamp(sy(y), 0, canvas_h - 1)
        try:
            bbox = draw.textbbox((x, y), text, font=font)
            pad = 3
            draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill="white")
        except Exception:
            pass
        draw.text((x, y), text, fill=color, font=font)

    header_color = "green" if status == "ok" else "red"
    header = f"{status.upper()} | raw={len(raw_person_dets)} unique={len(unique_person_dets)} main={len(selected_person_dets)}"
    if notes:
        header += f" | {notes}"
    label(header[:180], (16 / max(scale, 1e-9), 16 / max(scale, 1e-9)), header_color)

    for det in raw_person_dets:
        rect(det.box, "gray", 2)
        label(f"raw {det.conf:.2f}", (det.box[0], max(0, det.box[1] - 14)), "gray")

    for idx, det in enumerate(unique_person_dets, start=1):
        if id(det) in selected_ids:
            continue
        rect(det.safe_box, "orange", 3)
        label(f"unique_{idx} {det.conf:.2f}", (det.safe_box[0], max(0, det.safe_box[1] - 28)), "orange")

    for idx, det in enumerate(selected_person_dets, start=1):
        rect(det.safe_box, "lime", 5)
        label(f"MAIN_{idx} {det.conf:.2f}", (det.safe_box[0], max(0, det.safe_box[1] - 32)), "lime")

    save_debug_image(canvas, debug_path.with_suffix(".jpg"), quality=debug_quality)


def build_notes(raw_person_dets, unique_person_dets, selected_person_dets, main_ok, main_reason, required_people, img_w, img_h) -> str:
    notes = []

    if len(raw_person_dets) == 0:
        notes.append("no person detected")
    elif len(unique_person_dets) < required_people:
        notes.append("not enough unique people detected")
    elif not main_ok:
        notes.append("not enough main people after filtering")

    if "occluded_or_partial" in main_reason:
        notes.append("second person may be occluded or partial")

    if len(selected_person_dets) >= 2:
        a = selected_person_dets[0].safe_box
        b = selected_person_dets[1].safe_box
        if box_iou(a, b) >= 0.30 or box_intersection_over_min_area(a, b) >= 0.60:
            notes.append("people boxes overlap")

        v_overlap = vertical_overlap_ratio(a, b)
        center_dist = normalized_person_center_distance(a, b)
        if center_dist <= 1.0 and v_overlap >= 0.5:
            notes.append("people are close or possibly occluded")

    for det in selected_person_dets:
        x1, y1, x2, y2 = det.safe_box
        if x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2:
            notes.append("person box touches image boundary")
            break

    return "; ".join(sorted(set(notes)))


def process_image(img_path: Path, rel_path: Path, json_root: Path, debug_root: Path, fail_origin_root: Path, fail_debug_root: Path, model, args):
    img = load_image_rgb(img_path)
    img_w, img_h = img.size
    image_rgb = np.array(img)

    detections = get_detections(model=model, image_rgb=image_rgb, conf_thres=args.conf, imgsz=args.imgsz, device=args.device)
    raw_person_dets = [d for d in detections if d.name == "person"]

    if args.disable_person_dedup:
        unique_person_dets = list(raw_person_dets)
    else:
        unique_person_dets = deduplicate_person_detections(
            raw_person_dets,
            iou_thres=args.person_duplicate_iou,
            cover_thres=args.person_duplicate_cover,
        )

    required_people = args.min_people if args.min_people > 0 else 2
    if args.top_k > 0:
        required_people = min(required_people, args.top_k)

    main_ok, selected_person_dets, main_reason = select_main_person_detections(
        person_dets=unique_person_dets,
        img_w=img_w,
        img_h=img_h,
        required_people=required_people,
        top_k=args.top_k,
        min_area_ratio_to_largest=args.min_main_area_ratio_to_largest,
        min_abs_area_ratio=args.min_main_abs_area_ratio,
        allow_occluded_second_person=not args.disable_occluded_second_person,
        occluded_min_abs_area_ratio=args.occluded_min_abs_area_ratio,
        occluded_max_center_distance=args.occluded_max_center_distance,
        occluded_min_vertical_overlap=args.occluded_min_vertical_overlap,
    )

    notes = build_notes(
        raw_person_dets=raw_person_dets,
        unique_person_dets=unique_person_dets,
        selected_person_dets=selected_person_dets,
        main_ok=main_ok,
        main_reason=main_reason,
        required_people=required_people,
        img_w=img_w,
        img_h=img_h,
    )

    main_people = []
    for idx, det in enumerate(selected_person_dets, start=1):
        main_people.append({"id": idx, "label": f"person_{idx}", "box": round_box(det.safe_box, ndigits=args.box_digits)})

    confidence = float(min(d.conf for d in selected_person_dets)) if selected_person_dets else 0.0
    status = "ok" if main_ok else "fail"

    data = {
        "name": str(rel_path).replace("\\", "/"),
        "image_width": img_w,
        "image_height": img_h,
        "main_people": main_people,
        "confidence": round(confidence, 4),
        "notes": notes,
    }

    if args.include_extra_fields:
        data["_extra"] = {
            "status": status,
            "main_reason": main_reason,
            "raw_person_count": len(raw_person_dets),
            "unique_person_count": len(unique_person_dets),
            "selected_person_count": len(selected_person_dets),
            "model_conf": args.conf,
            "imgsz": args.imgsz,
        }

    json_path = (json_root / rel_path).with_suffix(".json")
    debug_path = (debug_root / rel_path).with_suffix(".jpg")

    save_json(data, json_path)
    draw_detection_debug_image(
        img=img,
        debug_path=debug_path,
        raw_person_dets=raw_person_dets,
        unique_person_dets=unique_person_dets,
        selected_person_dets=selected_person_dets,
        status=status,
        notes=notes,
        debug_max_side=args.debug_max_side,
        debug_quality=args.debug_quality,
    )

    # 检测失败时，额外复制一份原图和 debug 图到 fail 文件夹。
    # 注意：这里只复制，不移动，不影响 input_dir 里的原图。
    if status == "fail":
        fail_origin_path = fail_origin_root / rel_path
        fail_debug_path = (fail_debug_root / rel_path).with_suffix(".jpg")
        copy_file_preserve_path(img_path, fail_origin_path)
        copy_file_preserve_path(debug_path.with_suffix(".jpg"), fail_debug_path)

    return status


def main():
    parser = argparse.ArgumentParser(description="对已裁剪图片重新做人像检测，输出每张图的 bbox-only JSON 人体检测框和压缩可视化 debug 图。")

    parser.add_argument("--input_dir", type=str, required=True, help="输入主文件夹，递归处理其中所有图片")
    parser.add_argument("--output_dir", type=str, required=True, help="输出根目录，会创建 json 和 debug 两个子文件夹")
    parser.add_argument("--json_dir", type=str, default=None, help="可选：单独指定 JSON 输出目录")
    parser.add_argument("--debug_dir", type=str, default=None, help="可选：单独指定 debug 图输出目录")
    parser.add_argument("--fail_dir", type=str, default=None, help="可选：单独指定 fail 输出目录。默认 output_dir/fail，里面会有 origin 和 debug 两个子目录")

    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="YOLO 模型路径或模型名。可用 seg 模型，但本脚本只使用 bbox，不使用 mask_box")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO 检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=0, help="YOLO 推理尺寸，0 表示模型默认；小人多可设 1280")
    parser.add_argument("--device", type=str, default=None, help="可选：指定 YOLO device，例如 cpu、0、intel:gpu")

    parser.add_argument("--top_k", type=int, default=2, help="输出面积最大的几个主体人物。双人图建议 2；设 0 表示不过滤数量")
    parser.add_argument("--min_people", type=int, default=2, help="期望至少确认的人数，双人图建议 2")

    parser.add_argument("--min_main_area_ratio_to_largest", type=float, default=0.12, help="第二主体面积至少达到最大主体面积的比例")
    parser.add_argument("--min_main_abs_area_ratio", type=float, default=0.004, help="主体人物框面积至少占整图面积比例")

    parser.add_argument("--disable_occluded_second_person", action="store_true", help="关闭半身/遮挡第二主体兜底判断")
    parser.add_argument("--occluded_min_abs_area_ratio", type=float, default=0.002, help="半身/遮挡第二主体兜底的最低绝对面积比例")
    parser.add_argument("--occluded_max_center_distance", type=float, default=1.8, help="半身/遮挡第二主体与最大主体的最大归一化中心距离")
    parser.add_argument("--occluded_min_vertical_overlap", type=float, default=0.25, help="半身/遮挡第二主体与最大主体的最小纵向重叠比例")

    parser.add_argument("--disable_person_dedup", action="store_true", help="关闭 person 重复检测去重")
    parser.add_argument("--person_duplicate_iou", type=float, default=0.95, help="两个 person 框 IoU 大于该值时视为重复")
    parser.add_argument("--person_duplicate_cover", type=float, default=1.10, help="交集/较小框面积大于该值时视为重复")

    parser.add_argument("--debug_max_side", type=int, default=1600, help="debug 图最长边，0 表示不缩放")
    parser.add_argument("--debug_quality", type=int, default=75, help="debug JPG/WebP 保存质量")
    parser.add_argument("--box_digits", type=int, default=2, help="JSON box 坐标小数位数")
    parser.add_argument("--include_extra_fields", action="store_true", help="在 JSON 中额外写入 _extra 调试字段")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    json_root = Path(args.json_dir) if args.json_dir else output_dir / "json"
    debug_root = Path(args.debug_dir) if args.debug_dir else output_dir / "debug"
    fail_root = Path(args.fail_dir) if args.fail_dir else output_dir / "fail"
    fail_origin_root = fail_root / "origin"
    fail_debug_root = fail_root / "debug"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    json_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)
    fail_origin_root.mkdir(parents=True, exist_ok=True)
    fail_debug_root.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    model = YOLO(args.model)

    stats = {}
    for img_path in tqdm(image_paths, desc="Detecting people"):
        rel_path = img_path.relative_to(input_dir)
        try:
            status = process_image(
                img_path=img_path,
                rel_path=rel_path,
                json_root=json_root,
                debug_root=debug_root,
                fail_origin_root=fail_origin_root,
                fail_debug_root=fail_debug_root,
                model=model,
                args=args,
            )
        except Exception as e:
            status = "error"
            print(f"[ERROR] {img_path}: {e}")
            # 异常时也复制原图到 fail/origin，方便后续排查；没有可视化图则不复制 debug。
            try:
                copy_file_preserve_path(img_path, fail_origin_root / rel_path)
            except Exception as copy_e:
                print(f"[WARN] failed to copy error image to fail/origin: {copy_e}")
        stats[status] = stats.get(status, 0) + 1

    print("\nDone.")
    print(f"JSON output:        {json_root}")
    print(f"Debug output:       {debug_root}")
    print(f"Fail origin output: {fail_origin_root}")
    print(f"Fail debug output:  {fail_debug_root}")
    print("Stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


"""
PowerShell example:

python ./process/05_detect_v2_people_json_debug_bbox_only.py `
  --input_dir ./images/process_image/20260615_output/equal_2/output_crop `
  --output_dir ./images/process_image/20260615_output_detect_v1/equal_2 `
  --model yolo11s-seg_openvino_model-640 `
  --conf 0.5 `
  --imgsz 640 `
  --top_k 2 `
  --min_people 2 `
  --person_duplicate_iou 0.95 `
  --person_duplicate_cover 1.10 `
  --min_main_area_ratio_to_largest 0.12 `
  --min_main_abs_area_ratio 0.004
"""
