"""Data validation for training."""
from pathlib import Path


class InstanceDataValidator:
    """Validator for instance segmentation data."""

    def __init__(self, data_path: str):
        """Initialize validator."""
        self.data_path = Path(data_path)

    def validate(self):
        """Validate instance data structure."""
        print(f"Validating: {self.data_path}")
        self._check_splits()
        self._check_yaml()
        print("✓ All instance data validated")
        return str(self.data_path)

    def _check_splits(self):
        """Check train/valid/test splits."""
        splits = ['train', 'valid', 'test']
        for split in splits:
            self._validate_split(split)

    def _validate_split(self, split: str):
        """Validate a single split."""
        split_path = self.data_path / split
        self._ensure_exists(split_path)
        self._validate_split_structure(split_path, split)

    def _validate_split_structure(self, path: Path, name: str):
        """Validate split has images and labels."""
        img_path = path / 'images'
        lbl_path = path / 'labels'
        self._ensure_exists(img_path)
        self._ensure_exists(lbl_path)
        self._count_files(img_path, lbl_path, name)

    def _count_files(self, img_path: Path, lbl_path: Path, split: str):
        """Count and display files."""
        imgs = len(list(img_path.glob('*.jpg')))
        imgs += len(list(img_path.glob('*.png')))
        lbls = len(list(lbl_path.glob('*.txt')))
        print(f"✓ {split}: {imgs} images, {lbls} labels")

    def _check_yaml(self):
        """Check data.yaml exists."""
        yaml_path = self.data_path / 'data.yaml'
        self._ensure_exists(yaml_path)
        print(f"✓ data.yaml found: {yaml_path}")

    def _ensure_exists(self, path: Path):
        """Ensure path exists."""
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")


class SemanticDataValidator:
    """Validator for semantic segmentation data."""

    def __init__(self, data_path: str):
        """Initialize validator."""
        self.data_path = Path(data_path)

    def validate(self):
        """Validate semantic data structure."""
        print(f"Validating: {self.data_path}")
        self._check_splits()
        print("✓ All semantic data validated")
        return str(self.data_path)

    def _check_splits(self):
        """Check train/valid/test splits."""
        splits = ['train', 'valid', 'test']
        for split in splits:
            self._validate_split(split)

    def _validate_split(self, split: str):
        """Validate a single split."""
        split_path = self.data_path / split
        self._ensure_exists(split_path)
        self._validate_split_structure(split_path, split)

    def _validate_split_structure(self, path: Path, name: str):
        """Validate split has images and masks."""
        img_path = path / 'images'
        msk_path = path / 'masks'
        self._ensure_exists(img_path)
        self._ensure_exists(msk_path)
        self._count_and_verify(img_path, msk_path, name)

    def _count_and_verify(self, img_path: Path, msk_path: Path, split: str):
        """Count files and verify matching counts."""
        imgs = self._count_images(img_path)
        msks = len(list(msk_path.glob('*.png')))
        self._verify_counts(imgs, msks, split)
        print(f"✓ {split}: {imgs} images, {msks} masks")

    def _count_images(self, path: Path):
        """Count image files."""
        count = len(list(path.glob('*.jpg')))
        count += len(list(path.glob('*.png')))
        return count

    def _verify_counts(self, imgs: int, msks: int, split: str):
        """Verify image and mask counts match."""
        if imgs != msks:
            msg = f"{split}: images ({imgs}) != masks ({msks})"
            raise ValueError(msg)

    def _ensure_exists(self, path: Path):
        """Ensure path exists."""
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
