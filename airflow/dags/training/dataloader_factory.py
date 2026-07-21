"""DataLoader factory for semantic segmentation."""
from pathlib import Path
from torch.utils.data import DataLoader
from training.datasets import SemanticSegmentationDataset
from training.datasets import TransformFactory


class DataLoaderFactory:
    """Factory for creating DataLoaders."""

    def __init__(self, data_path: str, img_size: tuple, batch_size: int):
        """Initialize factory."""
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.batch_size = batch_size

    def create_loaders(self):
        """Create train, valid, test loaders."""
        train_loader = self._create_train_loader()
        valid_loader = self._create_valid_loader()
        test_loader = self._create_test_loader()
        self._print_info(train_loader, valid_loader, test_loader)
        return train_loader, valid_loader, test_loader

    def _create_train_loader(self):
        """Create training DataLoader."""
        dataset = self._create_train_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def _create_valid_loader(self):
        """Create validation DataLoader."""
        dataset = self._create_valid_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def _create_test_loader(self):
        """Create test DataLoader."""
        dataset = self._create_test_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def _create_train_dataset(self):
        """Create training dataset."""
        transform = TransformFactory.create_train_transform(self.img_size)
        return self._create_dataset('train', transform)

    def _create_valid_dataset(self):
        """Create validation dataset."""
        transform = TransformFactory.create_val_transform(self.img_size)
        return self._create_dataset('valid', transform)

    def _create_test_dataset(self):
        """Create test dataset."""
        transform = TransformFactory.create_val_transform(self.img_size)
        return self._create_dataset('test', transform)

    def _create_dataset(self, split: str, transform):
        """Create dataset for split."""
        images_dir = self.data_path / split / 'images'
        masks_dir = self.data_path / split / 'masks'
        return SemanticSegmentationDataset(
            str(images_dir),
            str(masks_dir),
            transform
        )

    def _print_info(self, train, valid, test):
        """Print loader information."""
        print("✓ DataLoaders created:")
        print(f"  Train: {len(train.dataset)} images")
        print(f"  Valid: {len(valid.dataset)} images")
        print(f"  Test: {len(test.dataset)} images")
