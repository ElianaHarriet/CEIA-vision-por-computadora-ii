from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.axes import Axes
from PIL import Image

CLASS_NAMES = {0: "Dent", 1: "Scratch", 2: "No Damage", 3: "Severe Damage"}


def parse_yolo_segmentation_line(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return class_id, points


def compute_polygon_stats(
    points: List[Tuple[float, float]],
) -> dict:
    if not points:
        return {"bbox_area": 0.0, "bbox_width": 0.0, "bbox_height": 0.0, "aspect_ratio": 0.0}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    area = w * h
    ar = w / h if h > 0 else 0.0
    return {"bbox_area": area, "bbox_width": w, "bbox_height": h, "aspect_ratio": ar}


def load_instance_labels(
    root_dir: Path, splits: List[str] | None = None
) -> pd.DataFrame:
    if splits is None:
        splits = ["train", "valid", "test"]

    records = []
    for split in splits:
        label_dir = root_dir / split / "labels"
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_id = label_path.stem
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id, points = parse_yolo_segmentation_line(line)
                    stats = compute_polygon_stats(points)
                    records.append(
                        {
                            "image_id": image_id,
                            "class_id": class_id,
                            "class_name": CLASS_NAMES.get(class_id, f"Class_{class_id}"),
                            "polygon_points": points,
                            "split": split,
                            **stats,
                        }
                    )
    return pd.DataFrame(records)


def plot_image_with_polygons(
    ax: Axes,
    image_path: Path,
    image_labels: pd.DataFrame,
    class_colors: dict[int, str] | None = None,
) -> None:
    if class_colors is None:
        class_colors = {0: "red", 1: "blue", 2: "green", 3: "orange"}

    img = Image.open(image_path)
    ax.imshow(img)
    h, w = img.size[1], img.size[0]

    for _, row in image_labels.iterrows():
        pts_norm = row["polygon_points"]
        pts_abs = [(x * w, y * h) for x, y in pts_norm]
        polygon = patches.Polygon(
            pts_abs,
            closed=True,
            linewidth=2,
            edgecolor=class_colors.get(row["class_id"], "white"),
            facecolor="none",
        )
        ax.add_patch(polygon)
        cx = sum(p[0] for p in pts_abs) / len(pts_abs)
        cy = sum(p[1] for p in pts_abs) / len(pts_abs)
        ax.text(
            cx,
            cy,
            row["class_name"],
            fontsize=8,
            color="white",
            bbox=dict(facecolor=class_colors.get(row["class_id"], "black"), alpha=0.6, pad=1),
            ha="center",
            va="center",
        )

    ax.axis("off")
