"""
Dataset preparation orchestrator for semantic segmentation.
Single Responsibility: Orchestrate dataset preparation workflow.
"""
from pathlib import Path
from typing import List, Dict

import sys
dag_path = Path(__file__).parent.parent
if str(dag_path) not in sys.path:
    sys.path.insert(0, str(dag_path))

from semantic.coco_processor import CocoAnnotationProcessor, YoloLabelGenerator, SemanticMaskGenerator
from core.file_operations import FileSystemOperations


class SemanticDatasetPreparer:
    """Prepare semantic segmentation datasets."""

    def __init__(self, raw_path: Path, ready_path: Path, class_names: List[str], name_map: Dict):
        """Initialize preparer with paths and configuration."""
        self._raw_path = raw_path
        self._ready_path = ready_path
        self._class_names = class_names
        self._name_map = name_map
        self._fs = FileSystemOperations()

    def prepare_split(self, split: str) -> int:
        """Prepare single split of dataset."""
        json_path = self._raw_path / split / "_annotations.coco.json"
        if not json_path.exists():
            return 0

        processor = CocoAnnotationProcessor(json_path, self._class_names, self._name_map)
        return self._process_split(split, processor)

    def create_data_yaml(self) -> None:
        """Create data.yaml for YOLOv8."""
        content = self._generate_yaml_content()
        yaml_path = self._ready_path / "instance" / "data.yaml"
        yaml_path.write_text(content)

    def _process_split(self, split: str, processor: CocoAnnotationProcessor) -> int:
        """Process split using COCO processor."""
        img_info = processor.get_images_info()
        cat_map = processor.get_category_mapping()
        anns_by_img = processor.group_annotations_by_image(cat_map)

        self._create_output_dirs(split)
        return self._process_images(split, img_info, anns_by_img, cat_map)

    def _create_output_dirs(self, split: str) -> None:
        """Create output directory structure."""
        for format_type in ["instance", "semantic"]:
            for subdir in self._get_subdirs(format_type):
                path = self._ready_path / format_type / split / subdir
                self._fs.ensure_directory(path)

    def _get_subdirs(self, format_type: str) -> List[str]:
        """Get subdirectories for format type."""
        if format_type == "instance":
            return ["images", "labels"]
        return ["images", "masks"]

    def _process_images(self, split: str, img_info: Dict, anns_by_img: Dict, cat_map: Dict) -> int:
        """Process all images in split."""
        count = 0
        for img_id, anns in anns_by_img.items():
            if self._process_single_image(split, img_id, img_info, anns, cat_map):
                count += 1
        return count

    def _process_single_image(self, split: str, img_id: int, img_info: Dict, anns: List, cat_map: Dict) -> bool:
        """Process single image and its annotations."""
        info = img_info[img_id]
        src_img = self._raw_path / split / info["file_name"]
        if not src_img.exists():
            return False

        self._copy_images(src_img, split, info["file_name"])
        self._create_instance_labels(info, anns, cat_map, split)
        self._create_semantic_mask(info, anns, cat_map, split)
        return True

    def _copy_images(self, src_img: Path, split: str, filename: str) -> None:
        """Copy image to both output directories."""
        inst_dst = self._ready_path / "instance" / split / "images" / filename
        sem_dst = self._ready_path / "semantic" / split / "images" / filename
        self._fs.copy_file(src_img, inst_dst)
        self._fs.copy_file(src_img, sem_dst)

    def _create_instance_labels(self, info: Dict, anns: List, cat_map: Dict, split: str) -> None:
        """Create YOLO format labels."""
        lines = []
        for ann in anns:
            line = YoloLabelGenerator.create_label_line(ann, cat_map, info["width"], info["height"])
            if line:
                lines.append(line)
        
        if lines:
            stem = Path(info["file_name"]).stem
            label_path = self._ready_path / "instance" / split / "labels" / f"{stem}.txt"
            label_path.write_text("\n".join(lines))

    def _create_semantic_mask(self, info: Dict, anns: List, cat_map: Dict, split: str) -> None:
        """Create semantic segmentation mask."""
        mask = SemanticMaskGenerator.create_mask(anns, cat_map, info["width"], info["height"])
        stem = Path(info["file_name"]).stem
        mask_path = self._ready_path / "semantic" / split / "masks" / f"{stem}.png"
        mask.save(mask_path)

    def _generate_yaml_content(self) -> str:
        """Generate YAML content for YOLOv8."""
        train_path = self._ready_path / "instance" / "train" / "images"
        val_path = self._ready_path / "instance" / "valid" / "images"
        test_path = self._ready_path / "instance" / "test" / "images"
        
        return f"""train: {train_path}
val: {val_path}
test: {test_path}
nc: {len(self._class_names)}
names: {self._class_names}
"""
