import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SRC = Path("car_damage_detection/data/car-damages-forked")
OUT = Path("car_damage_detection/data/car-damages-ready")

CLASS_NAMES = ["Minor Damage (Dent)", "Minor Damage (Scratch)", "No Damage", "Severe Damage"]

NAME_MAP = {
    "Minor Damage -Dent-": "Minor Damage (Dent)",
    "Minor Damage -Scratch-": "Minor Damage (Scratch)",
    "No Damage": "No Damage",
    "Severe Damage": "Severe Damage",
}

def polygons_to_mask(polygons, w, h):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            draw.polygon(poly, fill=255)
    return mask

for split in ["train", "valid", "test"]:
    src_split = SRC / split
    json_path = src_split / "_annotations.coco.json"
    if not json_path.exists():
        continue

    with open(json_path) as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    cat_map = {}
    for cat in data["categories"]:
        mapped = NAME_MAP.get(cat["name"])
        if mapped is not None:
            cat_map[cat["id"]] = class_to_idx[mapped]

    anns_by_img = {}
    for ann in data["annotations"]:
        if ann["category_id"] not in cat_map:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    inst_dir = OUT / "instance" / split
    inst_labels_dir = inst_dir / "labels"
    inst_images_dir = inst_dir / "images"
    inst_labels_dir.mkdir(parents=True, exist_ok=True)
    inst_images_dir.mkdir(parents=True, exist_ok=True)

    sem_dir = OUT / "semantic" / split
    sem_images_dir = sem_dir / "images"
    sem_masks_dir = sem_dir / "masks"
    sem_images_dir.mkdir(parents=True, exist_ok=True)
    sem_masks_dir.mkdir(parents=True, exist_ok=True)

    for img_id, anns in anns_by_img.items():
        info = img_info[img_id]
        w, h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem

        src_img = src_split / info["file_name"]
        if not src_img.exists():
            continue

        shutil.copy2(src_img, inst_images_dir / info["file_name"])
        shutil.copy2(src_img, sem_images_dir / info["file_name"])

        lines = []
        for ann in anns:
            segs = ann.get("segmentation")
            if not segs or not segs[0] or len(segs[0]) < 6:
                continue
            poly = segs[0]
            norm = [str(poly[i] / w if i % 2 == 0 else poly[i] / h) for i in range(len(poly))]
            cls_id = cat_map[ann["category_id"]]
            lines.append(f"{cls_id} " + " ".join(norm))

        if lines:
            (inst_labels_dir / f"{stem}.txt").write_text("\n".join(lines))

        sem_mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(sem_mask)
        for ann in anns:
            segs = ann.get("segmentation")
            if not segs or not segs[0] or len(segs[0]) < 6:
                continue
            cls_id = cat_map[ann["category_id"]] + 1
            coords = [(segs[0][i], segs[0][i+1]) for i in range(0, len(segs[0]), 2)]
            draw.polygon(coords, fill=cls_id)
        sem_mask.save(sem_masks_dir / f"{stem}.png")

inst_yaml = f"""train: {OUT}/instance/train/images
val: {OUT}/instance/valid/images
test: {OUT}/instance/test/images
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
(OUT / "instance" / "data.yaml").write_text(inst_yaml)

print("Done. Prepared datasets:")
for split in ["train", "valid", "test"]:
    inst = len(list((OUT / "instance" / split / "images").glob("*")))
    sem = len(list((OUT / "semantic" / split / "images").glob("*")))
    print(f"  {split}: instance={inst} images, semantic={sem} images")
