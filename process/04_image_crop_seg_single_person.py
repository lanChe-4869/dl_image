import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

'''
单人照片裁剪业务逻辑：
1. YOLO 检测 person，优先使用 segmentation mask 外接框与 bbox 的并集；
2. 对重复 person 框去重合并；
3. 从有效人物中选择面积最大的一个作为唯一主体；
4. 将与主体关联的随身物品纳入保护框；
5. 保持原有 padding、非对称 padding 倍率和候选长宽比，安全扩展后裁剪。

debug/process_fail 文件夹：
origin        检测/裁剪失败的原图副本
no_person     YOLO 没有检测到有效人物
bad_aspect    允许的长宽比都无法安全容纳保护框
invalid_crop  最终裁剪框非法
other_fail    其他未归类失败
'''
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_ASPECT_RATIOS = "1:1,3:2,2:3,4:3,3:4,5:4,4:5,16:9,9:16"
DEFAULT_PROTECT_OBJECTS = "backpack,umbrella,handbag,tie,suitcase"


@dataclass
class Detection:
    name: str
    cls_id: int
    conf: float
    box: List[float]
    mask_box: Optional[List[float]] = None

    @property
    def safe_box(self) -> List[float]:
        """优先把检测框和 segmentation mask 外接框合并，避免 mask 或 box 其中一个偏紧。"""
        if self.mask_box is None:
            return self.box
        return union_boxes([self.box, self.mask_box])


def clamp(v, low, high):
    return max(low, min(high, v))


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
    """
    交集 / 较小框面积。
    这个指标专门处理“同一个人被识别成一个大框 + 一个半身/局部小框”的情况。
    """
    inter = intersection_area(a, b)
    min_area = min(box_area(a), box_area(b))
    if min_area <= 0:
        return 0.0
    return inter / min_area


def are_duplicate_person_boxes(a, b, iou_thres=0.55, cover_thres=0.80):
    """
    判断两个 person 框是不是同一个人的重复检测。

    - IoU 高：两个框整体高度重合。
    - cover 高：一个框大面积被另一个框包住，典型就是全身框 + 半身框。
    """
    return (
        box_iou(a, b) >= iou_thres
        or box_intersection_over_min_area(a, b) >= cover_thres
    )


def merge_detection_cluster(cluster: Sequence[Detection]) -> Detection:
    """
    把同一个人的重复检测合并成一个唯一人物。
    这里用 union，而不是只保留最高置信度框，因为最高置信度框有时会是局部框。
    """
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


def deduplicate_person_detections(
    person_dets: Sequence[Detection],
    iou_thres=0.55,
    cover_thres=0.80,
) -> List[Detection]:
    """
    人物去重：处理同一个人被识别成全身框、半身框或局部框的情况。

    返回合并后的唯一人物列表。单人业务中，后续会从这里选择面积最大的有效人物作为主体。
    """
    clusters: List[List[Detection]] = []

    # 面积优先，避免先拿到局部框导致聚类不稳。
    sorted_dets = sorted(
        person_dets,
        key=lambda d: (box_area(d.safe_box), d.conf),
        reverse=True,
    )

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




def select_main_person_detection(
    person_dets: Sequence[Detection],
) -> Tuple[bool, Optional[Detection], str]:
    """
    从去重后的 person 检测中选择唯一主体。

    单人输入场景下，面积最大的有效人物框最稳定：
    - 能过滤偶发的背景小人；
    - 若同一个人出现重复框，前面的去重会先合并；
    - 面积相同或接近时，以置信度作为次级排序条件。
    """
    if not person_dets:
        return False, None, "no_person"

    valid_dets = [det for det in person_dets if box_area(det.safe_box) > 0]
    if not valid_dets:
        return False, None, "invalid_person_box"

    main_det = max(
        valid_dets,
        key=lambda det: (box_area(det.safe_box), det.conf),
    )
    return True, main_det, "ok"


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def boxes_intersect(a, b):
    return intersection_area(a, b) > 0


def union_boxes(boxes: Sequence[Sequence[float]]) -> List[float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def parse_aspect_ratios(text: str) -> List[Tuple[int, int]]:
    ratios = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid aspect ratio '{part}', expected format like 3:2")
        w, h = part.split(":", 1)
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid aspect ratio '{part}', width/height must be positive")
        ratios.append((w, h))
    if not ratios:
        raise ValueError("No valid aspect ratios provided")
    return ratios


def parse_name_set(text: str) -> set:
    return {x.strip().lower() for x in text.split(",") if x.strip()}


def expand_box_asymmetric(
    box,
    padding,
    img_w,
    img_h,
    side_multiplier=1.0,
    top_multiplier=1.0,
    bottom_multiplier=1.0,
):
    """
    非对称扩框：为了避免衣服下摆、包、腿脚被切，通常 bottom/side 可以更大。
    padding=0.25 表示基础扩展比例为 bbox 宽/高的 25%。
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    side_pad = bw * padding * side_multiplier
    top_pad = bh * padding * top_multiplier
    bottom_pad = bh * padding * bottom_multiplier

    x1 -= side_pad
    x2 += side_pad
    y1 -= top_pad
    y2 += bottom_pad

    return [
        clamp(x1, 0, img_w),
        clamp(y1, 0, img_h),
        clamp(x2, 0, img_w),
        clamp(y2, 0, img_h),
    ]


def is_ratio_feasible_for_box(box, target_aspect, img_w, img_h):
    """判断在不裁掉 box 的前提下，是否能扩展成指定比例并放进原图。"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    current_aspect = w / h
    if current_aspect > target_aspect:
        required_w = w
        required_h = w / target_aspect
    else:
        required_h = h
        required_w = h * target_aspect

    return required_w <= img_w + 1e-6 and required_h <= img_h + 1e-6


def choose_closest_feasible_aspect(box, allowed_ratios, img_w, img_h):
    """
    从候选比例里选最接近 bbox 的比例；如果某个比例会导致必须缩小 box，则跳过。
    这样可以保证比例适配阶段不会切掉衣服、包或身体边缘。
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None, "free"

    current_aspect = w / h
    best = None
    best_diff = float("inf")

    for rw, rh in allowed_ratios:
        aspect = rw / rh
        if not is_ratio_feasible_for_box(box, aspect, img_w, img_h):
            continue
        # log 差值对横竖比例更公平，例如 2:1 和 1:2 到 1:1 的距离对称。
        diff = abs(np.log(current_aspect / aspect))
        if diff < best_diff:
            best_diff = diff
            best = (aspect, f"{rw}:{rh}")

    if best is None:
        return None, "free_no_feasible_ratio"
    return best


def _place_interval_containing(start, end, length, limit):
    """在 [0, limit] 内放置一个长度为 length 的区间，并确保包含 [start, end]。"""
    length = min(length, limit)
    center = (start + end) / 2
    desired = center - length / 2

    min_left = max(0.0, end - length)
    max_left = min(start, limit - length)

    if min_left > max_left:
        # 理论上只有比例不可行或浮点误差才会发生；退化为尽量居中并限制边界。
        return clamp(desired, 0.0, max(0.0, limit - length))

    return clamp(desired, min_left, max_left)


def adjust_box_to_aspect_safe(box, target_aspect, img_w, img_h):
    """
    安全比例适配：只允许扩展，不允许为了满足比例而缩小原始保护框。
    返回的 crop box 一定尽量包含输入 box。
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return [int(round(v)) for v in box]

    current_aspect = w / h
    if current_aspect > target_aspect:
        new_w = w
        new_h = w / target_aspect
    else:
        new_h = h
        new_w = h * target_aspect

    if new_w > img_w + 1e-6 or new_h > img_h + 1e-6:
        # 不可行时直接返回原保护框，绝不缩小。
        return [
            int(round(clamp(x1, 0, img_w))),
            int(round(clamp(y1, 0, img_h))),
            int(round(clamp(x2, 0, img_w))),
            int(round(clamp(y2, 0, img_h))),
        ]

    left = _place_interval_containing(x1, x2, new_w, img_w)
    top = _place_interval_containing(y1, y2, new_h, img_h)
    right = left + new_w
    bottom = top + new_h

    return [
        int(round(clamp(left, 0, img_w))),
        int(round(clamp(top, 0, img_h))),
        int(round(clamp(right, 0, img_w))),
        int(round(clamp(bottom, 0, img_h))),
    ]


def choose_crop_with_padding_fallback(
    base_protect_box,
    initial_padding,
    allowed_ratios,
    img_w,
    img_h,
    side_multiplier=1.0,
    top_multiplier=1.0,
    bottom_multiplier=1.0,
    padding_steps=20,
):
    """
    在当前 padding 无法适配任何允许长宽比时，逐步缩减 padding 进行兜底。

    规则：
    1. 从 initial_padding 开始尝试；如果可行，直接使用当前 padding。
    2. 如果不可行，按线性步长逐渐减小到 0。
    3. 每次只缩小 base_protect_box 外面的背景留白，不缩小 base_protect_box 本身，
       因此不会裁掉已确认的人物/保护物体框。
    4. 每个候选 padding 下，仍然选择与当前保护框最接近的可行输出比例。
    """
    padding_steps = max(1, int(padding_steps))
    initial_padding = max(0.0, float(initial_padding))

    if initial_padding <= 0:
        padding_candidates = [0.0]
    else:
        padding_candidates = np.linspace(initial_padding, 0.0, padding_steps + 1).tolist()

    best_attempt = None
    for candidate_padding in padding_candidates:
        candidate_box = expand_box_asymmetric(
            base_protect_box,
            padding=candidate_padding,
            img_w=img_w,
            img_h=img_h,
            side_multiplier=side_multiplier,
            top_multiplier=top_multiplier,
            bottom_multiplier=bottom_multiplier,
        )
        target_aspect, ratio_name = choose_closest_feasible_aspect(
            candidate_box, allowed_ratios, img_w, img_h
        )

        # 记录最接近的尝试，便于失败 debug 图展示最后一次保护框。
        best_attempt = (candidate_box, ratio_name, candidate_padding)

        if target_aspect is None:
            continue

        crop_box = adjust_box_to_aspect_safe(candidate_box, target_aspect, img_w, img_h)
        return crop_box, candidate_box, ratio_name, candidate_padding, True

    if best_attempt is not None:
        candidate_box, ratio_name, candidate_padding = best_attempt
        return None, candidate_box, ratio_name, candidate_padding, False

    return None, base_protect_box, "free_no_feasible_ratio", 0.0, False


def polygon_to_box(poly):
    if poly is None or len(poly) == 0:
        return None
    arr = np.asarray(poly)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    xs = arr[:, 0]
    ys = arr[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def get_detections(model, image_rgb, conf_thres) -> List[Detection]:
    """
    返回 YOLO 检测结果。
    如果使用 yolov8n-seg.pt / yolov8s-seg.pt，会额外读取 mask 外接框。
    如果使用普通 yolov8n.pt，则 mask_box 为 None，自动退回 bbox。
    """
    results = model.predict(
        source=image_rgb,
        conf=conf_thres,
        verbose=False,
        # device="intel:gpu",
    )
    if not results:
        return []

    result = results[0]
    if result.boxes is None:
        return []

    names = result.names or {}
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    masks_xy = None
    if getattr(result, "masks", None) is not None and result.masks is not None:
        masks_xy = result.masks.xy

    detections = []
    for i, (b, c, s) in enumerate(zip(xyxy, cls, conf)):
        cls_id = int(c)
        name = str(names.get(cls_id, cls_id)).lower()
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]

        mask_box = None
        if masks_xy is not None and i < len(masks_xy):
            mask_box = polygon_to_box(masks_xy[i])

        detections.append(
            Detection(
                name=name,
                cls_id=cls_id,
                conf=float(s),
                box=[x1, y1, x2, y2],
                mask_box=mask_box,
            )
        )
    return detections


def select_protected_objects(detections, selected_person_boxes, protect_names, object_margin, img_w, img_h):
    """
    选中靠近人物的随身物品，例如 backpack / handbag / umbrella / suitcase。
    关联逻辑：物体框与人物区域相交，或物体中心落在人物 union 扩展区域内。
    """
    if not protect_names or not selected_person_boxes:
        return []

    person_union = union_boxes(selected_person_boxes)
    associate_area = expand_box_asymmetric(
        person_union,
        padding=object_margin,
        img_w=img_w,
        img_h=img_h,
        side_multiplier=1.0,
        top_multiplier=0.6,
        bottom_multiplier=1.2,
    )

    objects = []
    for det in detections:
        if det.name not in protect_names:
            continue
        obj_box = det.safe_box
        if boxes_intersect(obj_box, person_union) or point_in_box(box_center(obj_box), associate_area):
            objects.append(det)
    return objects


def save_image(img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(out_path, quality=95, subsampling=0)
    else:
        img.save(out_path)


def save_debug_image(img, out_path, max_side=1600, quality=75):
    """
    保存压缩后的 debug 画框图。

    只影响 debug/process_ok 和 debug/process_fail 里的标注图，不影响 output_crop 和失败原图 origin。
    max_side <= 0 表示不缩放；quality 只对 jpg/jpeg 生效。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    debug_img = img
    if max_side is not None and max_side > 0:
        w, h = debug_img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            debug_img = debug_img.resize(new_size, Image.Resampling.LANCZOS)

    suffix = out_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        out_path = out_path.with_suffix(".jpg")
        suffix = ".jpg"

    if suffix in {".jpg", ".jpeg"}:
        debug_img.save(out_path, quality=int(quality), optimize=True)
    elif suffix == ".webp":
        debug_img.save(out_path, quality=int(quality), method=6)
    else:
        debug_img.save(out_path, optimize=True)


def draw_debug_image(
    img,
    debug_path,
    raw_person_dets,
    selected_person_dets=None,
    object_dets=None,
    protect_box=None,
    crop_box=None,
    ratio_name="free",
    success=False,
    reason="",
    debug_max_side=1600,
    debug_quality=75,
):
    """
    调试图：先把原图缩放到 debug 尺寸，再按同一比例缩放坐标并画框/文字。

    这样不会出现“先在大图上画字，再缩小保存”导致的文字和置信度模糊问题。

    - 灰色：YOLO 检出的所有 person
    - 绿色：确认后的唯一主体人物
    - 蓝色：被纳入保护的随身物品
    - 橙色：保护框
    - 绿色粗框：最终裁切框
    """
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    src_w, src_h = img.size
    scale = 1.0
    if debug_max_side is not None and debug_max_side > 0:
        longest = max(src_w, src_h)
        if longest > debug_max_side:
            scale = debug_max_side / float(longest)

    if scale < 1.0:
        canvas_size = (
            max(1, int(round(src_w * scale))),
            max(1, int(round(src_h * scale))),
        )
        canvas = img.resize(canvas_size, Image.Resampling.LANCZOS)
    else:
        canvas = img.copy()

    draw = ImageDraw.Draw(canvas)
    canvas_w, canvas_h = canvas.size

    selected_person_dets = selected_person_dets or []
    object_dets = object_dets or []
    selected_ids = {id(d) for d in selected_person_dets}

    # 在缩放后的图上原生画字，避免文字被 resize 模糊。
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
        # 给文字加一层浅底，压缩后也更容易看清。
        try:
            bbox = draw.textbbox((x, y), text, font=font)
            pad = 3
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                fill="white",
            )
        except Exception:
            pass
        draw.text((x, y), text, fill=color, font=font)

    status = "PROCESS_OK" if success else "PROCESS_FAIL"
    status_color = "green" if success else "red"
    label_text = status if not reason else f"{status}: {reason}"
    label(label_text[:160], (16 / max(scale, 1e-9), 16 / max(scale, 1e-9)), status_color)

    # 先画所有 YOLO person 原始框，方便排查是否漏检主体。
    for det in raw_person_dets:
        color = "green" if id(det) in selected_ids else "gray"
        width = 5 if id(det) in selected_ids else 3
        rect(det.box, color, width)
        label(f"person {det.conf:.2f}", (det.box[0], max(0, det.box[1] - 14)), color)
        if det.mask_box is not None:
            rect(det.mask_box, "magenta" if id(det) in selected_ids else "purple", 2)

    # 去重合并后的主体框可能不是 raw det 本身，额外画 safe_box。
    for idx, det in enumerate(selected_person_dets, start=1):
        rect(det.safe_box, "lime", 5)
        label(f"MAIN {det.conf:.2f}", (det.safe_box[0], max(0, det.safe_box[1] - 30)), "lime")

    for det in object_dets:
        rect(det.safe_box, "blue", 3)
        label(f"{det.name} {det.conf:.2f}", (det.safe_box[0], max(0, det.safe_box[1] - 14)), "blue")

    if protect_box is not None:
        rect(protect_box, "orange", 4)
        label("protect box", (protect_box[0], max(0, protect_box[1] - 16)), "orange")

    if crop_box is not None:
        rect(crop_box, "green", 5)
        label(f"final crop {ratio_name}", (crop_box[0], max(0, crop_box[1] - 18)), "green")

    if debug_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        debug_path = debug_path.with_suffix(".jpg")

    # canvas 已经是压缩后的 debug 尺寸，这里只负责保存，不再二次缩放。
    save_debug_image(canvas, debug_path, max_side=0, quality=debug_quality)



def failure_reason_to_folder(reason: str) -> str:
    """把详细失败原因映射成简短、稳定的 debug 子目录名。"""
    reason = str(reason or "").lower()

    if "no_feasible_allowed_aspect_ratio" in reason:
        return "bad_aspect"
    if "invalid_crop" in reason:
        return "invalid_crop"
    if "no_person" in reason or ("raw0" in reason and "unique0" in reason):
        return "no_person"
    return "other_fail"


def build_debug_fail_path(debug_fail_root, debug_rel_path, reason):
    """构造 debug/process_fail/<问题类型>/<原相对路径>.jpg。"""
    if debug_fail_root is None or debug_rel_path is None:
        return None
    return (Path(debug_fail_root) / failure_reason_to_folder(reason) / debug_rel_path).with_suffix(".jpg")


def build_debug_fail_origin_path(debug_fail_root, rel_path):
    """构造 debug/process_fail/origin/<原相对路径>，用于保存/移动检测失败原图。"""
    if debug_fail_root is None or rel_path is None:
        return None
    return Path(debug_fail_root) / "origin" / rel_path


def move_failed_origin_image(img_path, origin_path):
    """把检测失败的原图复制到 debug/process_fail/origin 下，保留原始相对路径。

    如果目标文件已存在，为避免覆盖，会自动追加 _1、_2 等后缀。
    """
    if origin_path is None:
        return None

    img_path = Path(img_path)
    origin_path = Path(origin_path)

    if not img_path.exists():
        return None

    origin_path.parent.mkdir(parents=True, exist_ok=True)

    final_path = origin_path
    if final_path.exists():
        stem = final_path.stem
        suffix = final_path.suffix
        parent = final_path.parent
        idx = 1
        while final_path.exists():
            final_path = parent / f"{stem}_{idx}{suffix}"
            idx += 1

    shutil.copy2(str(img_path), str(final_path))
    return final_path

def process_one_image(
    img_path,
    out_path,
    model,
    conf_thres,
    padding,
    adjust_aspect,
    allowed_ratios,
    fallback,
    protect_objects,
    object_margin,
    side_padding_multiplier,
    top_padding_multiplier,
    bottom_padding_multiplier,
    dedupe_people=True,
    person_duplicate_iou=0.55,
    person_duplicate_cover=0.80,
    debug_ok_path=None,
    debug_fail_path=None,
    debug_fail_root=None,
    debug_rel_path=None,
    debug_fail_origin_path=None,
    debug_max_side=1600,
    debug_quality=75,
):
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    img_w, img_h = img.size
    image_rgb = np.array(img)

    detections = get_detections(model, image_rgb, conf_thres)
    raw_person_dets = [d for d in detections if d.name == "person"]

    if dedupe_people:
        person_dets = deduplicate_person_detections(
            raw_person_dets,
            iou_thres=person_duplicate_iou,
            cover_thres=person_duplicate_cover,
        )
    else:
        person_dets = list(raw_person_dets)

    raw_person_count = len(raw_person_dets)
    unique_person_count = len(person_dets)

    main_ok, selected_person_det, main_reason = select_main_person_detection(person_dets)
    selected_person_dets = [selected_person_det] if selected_person_det is not None else []

    if not main_ok:
        reason = f"{main_reason}_raw{raw_person_count}_unique{unique_person_count}"
        current_debug_fail_path = build_debug_fail_path(debug_fail_root, debug_rel_path, reason) or debug_fail_path
        if current_debug_fail_path is not None:
            draw_debug_image(
                img=img,
                debug_path=current_debug_fail_path,
                raw_person_dets=raw_person_dets,
                selected_person_dets=selected_person_dets,
                success=False,
                reason=reason,
                debug_max_side=debug_max_side,
                debug_quality=debug_quality,
            )
        if fallback == "error":
            raise RuntimeError(f"Failed to confirm the main person: {img_path}, {reason}")
        move_failed_origin_image(img_path, debug_fail_origin_path)
        return f"failed_{reason}"

    selected_person_boxes = [d.safe_box for d in selected_person_dets]

    object_dets = select_protected_objects(
        detections=detections,
        selected_person_boxes=selected_person_boxes,
        protect_names=protect_objects,
        object_margin=object_margin,
        img_w=img_w,
        img_h=img_h,
    )

    # 基础保护框 = 单个主体人物安全框 + 与其关联的随身物品框。
    # 后续自适应缩减 padding 时，只会减少这个基础保护框外面的背景留白，
    # 不会缩小基础保护框本身，因此不会破坏人物完整性。
    all_protected_boxes = selected_person_boxes + [d.safe_box for d in object_dets]
    base_protect_box = union_boxes(all_protected_boxes)

    ratio_name = "free"
    effective_padding = padding

    if adjust_aspect:
        (
            crop_box,
            protect_box,
            ratio_name,
            effective_padding,
            crop_ok,
        ) = choose_crop_with_padding_fallback(
            base_protect_box=base_protect_box,
            initial_padding=padding,
            allowed_ratios=allowed_ratios,
            img_w=img_w,
            img_h=img_h,
            side_multiplier=side_padding_multiplier,
            top_multiplier=top_padding_multiplier,
            bottom_multiplier=bottom_padding_multiplier,
            padding_steps=20,
        )

        if not crop_ok:
            reason = "no_feasible_allowed_aspect_ratio"
            current_debug_fail_path = build_debug_fail_path(debug_fail_root, debug_rel_path, reason) or debug_fail_path
            if current_debug_fail_path is not None:
                draw_debug_image(
                    img=img,
                    debug_path=current_debug_fail_path,
                    raw_person_dets=raw_person_dets,
                    selected_person_dets=selected_person_dets,
                    object_dets=object_dets,
                    protect_box=protect_box,
                    crop_box=None,
                    ratio_name=ratio_name,
                    success=False,
                    reason=reason,
                    debug_max_side=debug_max_side,
                    debug_quality=debug_quality,
                )
            if fallback == "error":
                raise RuntimeError(f"No feasible allowed aspect ratio for image: {img_path}")
            move_failed_origin_image(img_path, debug_fail_origin_path)
            return f"failed_{reason}"
    else:
        protect_box = expand_box_asymmetric(
            base_protect_box,
            padding=padding,
            img_w=img_w,
            img_h=img_h,
            side_multiplier=side_padding_multiplier,
            top_multiplier=top_padding_multiplier,
            bottom_multiplier=bottom_padding_multiplier,
        )
        crop_box = [int(round(v)) for v in protect_box]

    x1, y1, x2, y2 = crop_box

    if x2 <= x1 or y2 <= y1:
        reason = "invalid_crop"
        current_debug_fail_path = build_debug_fail_path(debug_fail_root, debug_rel_path, reason) or debug_fail_path
        if current_debug_fail_path is not None:
            draw_debug_image(
                img=img,
                debug_path=current_debug_fail_path,
                raw_person_dets=raw_person_dets,
                selected_person_dets=selected_person_dets,
                object_dets=object_dets,
                protect_box=protect_box,
                crop_box=crop_box,
                ratio_name=ratio_name,
                success=False,
                reason=reason,
                debug_max_side=debug_max_side,
                debug_quality=debug_quality,
            )
        if fallback == "error":
            raise RuntimeError(f"Invalid crop box for image: {img_path}")
        move_failed_origin_image(img_path, debug_fail_origin_path)
        return f"failed_{reason}"

    cropped = img.crop((x1, y1, x2, y2))
    save_image(cropped, out_path)

    if debug_ok_path is not None:
        draw_debug_image(
            img=img,
            debug_path=debug_ok_path,
            raw_person_dets=raw_person_dets,
            selected_person_dets=selected_person_dets,
            object_dets=object_dets,
            protect_box=protect_box,
            crop_box=crop_box,
            ratio_name=ratio_name,
            success=True,
            reason=f"ok padding={effective_padding:.4f}",
            debug_max_side=debug_max_side,
            debug_quality=debug_quality,
        )

    has_masks = any(d.mask_box is not None for d in selected_person_dets)
    mode = "mask" if has_masks else "bbox"
    object_suffix = f"_obj{len(object_dets)}" if object_dets else ""
    padding_suffix = f"_pad{effective_padding:.4f}" if adjust_aspect and abs(effective_padding - padding) > 1e-9 else ""
    return f"cropped_{ratio_name}_{mode}{object_suffix}{padding_suffix}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True, help="输入图片文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出根目录。脚本会在里面自动创建 output_crop 和 debug/process_ok、debug/process_fail")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO 模型路径或模型名。建议使用 yolov8n-seg.pt / yolov8s-seg.pt，以便用 person mask 保护衣服轮廓。普通 yolov8n.pt 也能用，但只会用 bbox。",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO 检测置信度阈值")
    parser.add_argument("--padding", type=float, default=0.25, help="基础 bbox 外扩比例")
    parser.add_argument(
        "--disable_person_dedup",
        action="store_true",
        help="关闭 person 重复检测去重。单人照片通常建议保持开启。",
    )
    parser.add_argument(
        "--person_duplicate_iou",
        type=float,
        default=0.55,
        help="两个 person 框 IoU 大于该值时，视为同一个人的重复检测。",
    )
    parser.add_argument(
        "--person_duplicate_cover",
        type=float,
        default=0.80,
        help="交集/较小框面积大于该值时，视为同一个人的重复检测。用于处理大框套小框。",
    )

    parser.add_argument(
        "--aspect_ratios",
        type=str,
        default=DEFAULT_ASPECT_RATIOS,
        help="允许的输出比例，用英文逗号分隔，例如 1:1,3:2,2:3,4:3,3:4,16:9,9:16",
    )
    parser.add_argument(
        "--no_adjust_aspect",
        "--no_keep_aspect",
        dest="no_adjust_aspect",
        action="store_true",
        help="不调整到固定比例，直接按保护框裁剪。兼容旧参数 --no_keep_aspect。",
    )

    parser.add_argument(
        "--protect_objects",
        type=str,
        default=DEFAULT_PROTECT_OBJECTS,
        help="需要和人物一起保护的 COCO 物体类别，用英文逗号分隔。默认保护 backpack,umbrella,handbag,tie,suitcase。",
    )
    parser.add_argument(
        "--no_protect_objects",
        action="store_true",
        help="关闭随身物品保护，只使用 person 框/mask。",
    )
    parser.add_argument(
        "--object_margin",
        type=float,
        default=0.35,
        help="判断随身物品是否属于人物附近的扩展范围。越大越容易把附近包/伞纳入保护。",
    )
    parser.add_argument("--side_padding_multiplier", type=float, default=1.35, help="左右 padding 倍率，防止衣袖/包边被裁")
    parser.add_argument("--top_padding_multiplier", type=float, default=1.0, help="顶部 padding 倍率")
    parser.add_argument("--bottom_padding_multiplier", type=float, default=1.8, help="底部 padding 倍率，防止衣服下摆/腿脚被裁")

    parser.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="可选：debug 根目录。默认使用 output_dir/debug，并自动分 process_ok / process_fail/<问题类型>。",
    )
    parser.add_argument(
        "--debug_max_side",
        type=int,
        default=1600,
        help="debug 画框图压缩后的最长边像素。设为 0 表示不缩放。默认 1600。",
    )
    parser.add_argument(
        "--debug_quality",
        type=int,
        default=75,
        help="debug 画框图 JPG/WebP 保存质量，1-95。默认 75。",
    )
    parser.add_argument(
        "--fallback",
        type=str,
        default="skip",
        choices=["copy", "skip", "error"],
        help="失败时的处理。当前需求下 copy/skip 都不会输出 crop；error 会直接报错。默认 skip。",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_crop_dir = output_root / "output_crop"
    debug_root = Path(args.debug_dir) if args.debug_dir else output_root / "debug"
    debug_ok_dir = debug_root / "process_ok"
    debug_fail_dir = debug_root / "process_fail"
    debug_fail_origin_dir = debug_fail_dir / "origin"

    output_crop_dir.mkdir(parents=True, exist_ok=True)
    debug_ok_dir.mkdir(parents=True, exist_ok=True)
    debug_fail_dir.mkdir(parents=True, exist_ok=True)
    debug_fail_origin_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    allowed_ratios = parse_aspect_ratios(args.aspect_ratios)
    protect_objects = set() if args.no_protect_objects else parse_name_set(args.protect_objects)

    model = YOLO(args.model)

    stats = {}
    for img_path in tqdm(image_paths, desc="Cropping images"):
        rel_path = img_path.relative_to(input_dir)
        out_path = output_crop_dir / rel_path
        debug_ok_path = (debug_ok_dir / rel_path).with_suffix(".jpg")
        debug_fail_rel_path = rel_path.with_suffix(".jpg")
        debug_fail_origin_path = build_debug_fail_origin_path(debug_fail_dir, rel_path)
        # 兼容兜底路径；实际失败图会优先进入 debug/process_fail/<问题类型>/<相对路径>.jpg
        debug_fail_path = (debug_fail_dir / "other_fail" / debug_fail_rel_path).with_suffix(".jpg")

        try:
            status = process_one_image(
                img_path=img_path,
                out_path=out_path,
                model=model,
                conf_thres=args.conf,
                padding=args.padding,
                adjust_aspect=not args.no_adjust_aspect,
                allowed_ratios=allowed_ratios,
                fallback=args.fallback,
                protect_objects=protect_objects,
                object_margin=args.object_margin,
                side_padding_multiplier=args.side_padding_multiplier,
                top_padding_multiplier=args.top_padding_multiplier,
                bottom_padding_multiplier=args.bottom_padding_multiplier,
                dedupe_people=not args.disable_person_dedup,
                person_duplicate_iou=args.person_duplicate_iou,
                person_duplicate_cover=args.person_duplicate_cover,
                debug_ok_path=debug_ok_path,
                debug_fail_path=debug_fail_path,
                debug_fail_root=debug_fail_dir,
                debug_rel_path=debug_fail_rel_path,
                debug_fail_origin_path=debug_fail_origin_path,
                debug_max_side=args.debug_max_side,
                debug_quality=args.debug_quality,
            )
        except Exception as e:
            status = "error"
            print(f"[ERROR] {img_path}: {e}")

        stats[status] = stats.get(status, 0) + 1

    print("\nDone.")
    print("Debug fail folders:")
    print("  origin: original images copied here when processing fails")
    print("  no_person: YOLO did not detect a valid person")
    print("  bad_aspect: no allowed aspect ratio can safely contain the main-person protect box")
    print("  invalid_crop: crop box is invalid")
    print("  other_fail: uncategorized failure")
    print("Stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()



'''
PowerShell example（单人照片）:

python ./process/04_image_crop_seg_single_person.py `
  --input_dir images\20260731_single\images_res `
  --output_dir images\20260731_single\images_res_crop `
  --model yolo11s-seg_openvino_model-640 `
  --conf 0.1 `
  --padding 0.2

常用调整：
- 没检测到人物：降低 --conf，例如 0.25 -> 0.15。
- 底部留白过多：降低 --bottom_padding_multiplier，例如 1.8 -> 1.3。
- 某些比例无法安全裁剪：降低 --padding，或缩小非对称 padding 倍率。
- 不需要固定长宽比：添加 --no_adjust_aspect。
'''
