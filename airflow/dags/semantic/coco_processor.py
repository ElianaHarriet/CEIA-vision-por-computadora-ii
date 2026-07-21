"""
COCO format processor for semantic segmentation.
Single Responsibility: Process COCO annotations.
"""
import json
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageDraw


class CocoAnnotationProcessor:
    """Process COCO format annotations."""

    def __init__(self, json_path: Path, class_names: List[str], name_map: Dict[str, str]):
        """Initialize processor with COCO data."""
        self._json_path = json_path
        self._class_names = class_names
        self._name_map = name_map
        self._data = self._load_json()

    def get_images_info(self) -> Dict:
        """Get images information from COCO."""
        return {img["id"]: img for img in self._data["images"]}

    def get_category_mapping(self) -> Dict[int, int]:
        """Map COCO categories to class indices."""
        class_to_idx = {name: i for i, name in enumerate(self._class_names)}
        cat_map = {}
        for cat in self._data["categories"]:
            mapped = self._name_map.get(cat["name"])
            if mapped:
                cat_map[cat["id"]] = class_to_idx[mapped]
        return cat_map

    def group_annotations_by_image(self, cat_map: Dict[int, int]) -> Dict:
        """Group annotations by image ID."""
        anns_by_img = {}
        for ann in self._data["annotations"]:
            if ann["category_id"] not in cat_map:
                continue
            img_id = ann["image_id"]
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)
        return anns_by_img

    def _load_json(self) -> Dict:
        """Load COCO JSON file."""
        with open(self._json_path, encoding='utf-8') as f:
            return json.load(f)


class YoloLabelGenerator:
    """Generate YOLO format labels."""

    @staticmethod
    def create_label_line(annotation: Dict, cat_map: Dict, width: int, height: int) -> str:
        """Create single YOLO label line."""
        segs = annotation.get("segmentation")
        if not segs or not segs[0] or len(segs[0]) < 6:
            return ""
        
        poly = segs[0]
        normalized = YoloLabelGenerator._normalize_polygon(poly, width, height)
        cls_id = cat_map[annotation["category_id"]]
        return f"{cls_id} {normalized}"

    @staticmethod
    def _normalize_polygon(polygon: List[float], width: int, height: int) -> str:
        """Normalize polygon coordinates."""
        normalized = []
        for i in range(len(polygon)):
            value = polygon[i] / width if i % 2 == 0 else polygon[i] / height
            normalized.append(str(value))
        return " ".join(normalized)


class SemanticMaskGenerator:
    """Generate semantic segmentation masks."""

    @staticmethod
    def create_mask(annotations: List[Dict], cat_map: Dict, width: int, height: int) -> Image:
        """Create semantic mask from annotations."""
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        for ann in annotations:
            SemanticMaskGenerator._draw_annotation(ann, cat_map, draw)
        
        return mask

    @staticmethod
    def _draw_annotation(ann: Dict, cat_map: Dict, draw: ImageDraw) -> None:
        """Draw single annotation on mask."""
        segs = ann.get("segmentation")
        if not segs or not segs[0] or len(segs[0]) < 6:
            return
        
        cls_id = cat_map[ann["category_id"]] + 1
        coords = [(segs[0][i], segs[0][i+1]) for i in range(0, len(segs[0]), 2)]
        draw.polygon(coords, fill=cls_id)
