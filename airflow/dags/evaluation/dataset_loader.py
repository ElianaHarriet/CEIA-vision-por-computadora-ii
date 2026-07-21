"""Dataset loader for evaluation."""
from pathlib import Path
import numpy as np
from PIL import Image


class TestDatasetLoader:
    """Loader for test dataset."""

    def __init__(self, instance_path: str, semantic_path: str):
        """Initialize loader."""
        self.instance_path = Path(instance_path)
        self.semantic_path = Path(semantic_path)

    def load_test_images(self):
        """Load test images."""
        print("Loading test images...")
        instance_imgs = self._get_instance_images()
        semantic_imgs = self._get_semantic_images()
        common_imgs = self._find_common_images(
            instance_imgs,
            semantic_imgs
        )
        print(f"✓ Found {len(common_imgs)} test images")
        return common_imgs

    def _get_instance_images(self):
        """Get instance test images."""
        path = self.instance_path / 'test' / 'images'
        return self._list_images(path)

    def _get_semantic_images(self):
        """Get semantic test images."""
        path = self.semantic_path / 'test' / 'images'
        return self._list_images(path)

    def _list_images(self, path: Path):
        """List image files."""
        jpg_files = list(path.glob('*.jpg'))
        png_files = list(path.glob('*.png'))
        files = jpg_files + png_files
        return {f.stem: str(f) for f in files}

    def _find_common_images(self, dict1: dict, dict2: dict):
        """Find common images between datasets."""
        common_keys = set(dict1.keys()) & set(dict2.keys())
        return {k: dict1[k] for k in sorted(common_keys)}

    def load_ground_truth_masks(self, image_names: list):
        """Load ground truth masks."""
        print("Loading ground truth masks...")
        masks = {}
        mask_dir = self.semantic_path / 'test' / 'masks'
        for name in image_names:
            mask_path = mask_dir / f"{name}.png"
            if mask_path.exists():
                masks[name] = np.array(Image.open(mask_path))
        print(f"✓ Loaded {len(masks)} ground truth masks")
        return masks
