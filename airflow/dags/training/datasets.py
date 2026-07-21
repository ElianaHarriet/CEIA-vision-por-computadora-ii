"""Datasets for semantic segmentation."""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SemanticSegmentationDataset(Dataset):
    """Dataset for semantic segmentation."""

    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        """Initialize dataset."""
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self._load_file_paths()

    def _load_file_paths(self):
        """Load image and mask paths."""
        self.images = self._get_sorted_files(self.images_dir)
        self.masks = self._get_sorted_masks()
        self._verify_counts()

    def _get_sorted_files(self, path: Path):
        """Get sorted image files."""
        jpg_files = list(path.glob('*.jpg'))
        png_files = list(path.glob('*.png'))
        return sorted(jpg_files + png_files)

    def _get_sorted_masks(self):
        """Get sorted mask files."""
        return sorted(list(self.masks_dir.glob('*.png')))

    def _verify_counts(self):
        """Verify image and mask counts match."""
        assert len(self.images) == len(self.masks), "Mismatch"

    def __len__(self):
        """Get dataset size."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get single sample."""
        image = self._load_image(idx)
        mask = self._load_mask(idx)
        return self._apply_transform(image, mask)

    def _load_image(self, idx: int):
        """Load image as RGB array."""
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        return np.array(image)

    def _load_mask(self, idx: int):
        """Load mask as array."""
        mask_path = self.masks[idx]
        return np.array(Image.open(mask_path))

    def _apply_transform(self, image, mask):
        """Apply transformations."""
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], augmented['mask'].long()
        return image, mask


class TransformFactory:
    """Factory for creating transforms."""

    @staticmethod
    def create_train_transform(img_size: tuple):
        """Create training transforms."""
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def create_val_transform(img_size: tuple):
        """Create validation transforms."""
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
