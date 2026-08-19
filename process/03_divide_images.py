import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(input_dir: Path):
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def detect_person_boxes(model, image_path: Path, conf: float):
    result = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False
    )[0]

    persons = []

    if result.boxes is None:
        return persons

    for b in result.boxes:
        cls_id = int(b.cls.item())
        score = float(b.conf.item())

        # COCO 里 person 类别 id 是 0
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()

        persons.append({
            "bbox": [
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ],
            "conf": score,
        })

    return persons


def count_main_persons(
    persons,
    min_area_ratio=0.03,
    top_area_keep_ratio=0.35
):
    """
    只统计主人物，尽量忽略背景小人。

    min_area_ratio:
        人物框面积 / 图片面积，小于这个比例认为是背景小人。

    top_area_keep_ratio:
        相对于最大人物框面积，太小的人认为是背景小人。
        例如 0.35 表示小于最大人物 35% 面积的人会被忽略。
    """

    if not persons:
        return 0

    areas = [p["area"] for p in persons]
    max_area = max(areas)

    main_persons = []

    for p in persons:
        if p["area_ratio"] < min_area_ratio:
            continue

        if p["area"] < max_area * top_area_keep_ratio:
            continue

        main_persons.append(p)

    return len(main_persons)


def get_folder_id(image_path: Path, source_dir: Path):
    """
    从次文件夹名称中提取 id。

    假设结构为：
    source_dir/
        12345_说明/
            xxx.jpg

    返回 12345
    """

    relative_path = image_path.relative_to(source_dir)

    # 第一级目录就是次文件夹
    sub_folder_name = relative_path.parts[0]

    # 从 "12345_说明" 取 "12345"
    folder_id = sub_folder_name.split("_")[0]

    return folder_id


def get_unique_target_path(target_dir: Path, filename: str):
    """
    防止极端情况下重名覆盖。
    正常情况下 id_原图名 已经够用。
    """

    target_path = target_dir / filename

    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix

    counter = 1
    while True:
        new_path = target_dir / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def main():
    # =========================
    # 你只需要改这里
    # =========================

    source_dir = Path(r"images\single\20260812_single\images")
    output_dir = Path(r"images\single\20260812_single\images_res")

    model_path = "yolo11n.pt"
    conf = 0.35

    # 控制“背景小人”过滤强度
    min_area_ratio = 0.03
    top_area_keep_ratio = 0.35

    # =========================

    less_dir = output_dir / "less_than_2"
    equal_dir = output_dir / "equal_2"
    more_dir = output_dir / "more_than_2"

    less_dir.mkdir(parents=True, exist_ok=True)
    equal_dir.mkdir(parents=True, exist_ok=True)
    more_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    image_paths = list_images(source_dir)

    count_less = 0
    count_equal = 0
    count_more = 0

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            persons = detect_person_boxes(
                model=model,
                image_path=image_path,
                conf=conf
            )

            # 读取图片尺寸，用于计算人物面积占比
            from PIL import Image
            with Image.open(image_path) as img:
                image_w, image_h = img.size

            image_area = image_w * image_h

            for p in persons:
                area = box_area(p["bbox"])
                p["area"] = area
                p["area_ratio"] = area / image_area

            main_person_count = count_main_persons(
                persons,
                min_area_ratio=min_area_ratio,
                top_area_keep_ratio=top_area_keep_ratio
            )

            if main_person_count < 2:
                target_dir = less_dir
                count_less += 1
            elif main_person_count == 2:
                target_dir = equal_dir
                count_equal += 1
            else:
                target_dir = more_dir
                count_more += 1

            folder_id = get_folder_id(image_path, source_dir)

            new_name = f"{folder_id}_{image_path.name}"

            target_path = get_unique_target_path(target_dir, new_name)

            # import gc
            # gc.collect()

            shutil.move(str(image_path), str(target_path))

        except Exception as e:
            print(f"[ERROR] 处理失败: {image_path}")
            print(e)

    print("完成")
    print(f"小于2人: {count_less}")
    print(f"等于2人: {count_equal}")
    print(f"大于2人: {count_more}")


if __name__ == "__main__":
    main()
