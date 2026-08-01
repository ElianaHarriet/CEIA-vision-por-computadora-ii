"""Predictors for evaluation."""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YOLOPredictor:
    """Predictor for YOLO models."""

    def __init__(self, model):
        """Initialize predictor."""
        self.model = model

    def predict(self, image_paths: dict):
        """Predict on images."""
        print("Predicting with YOLO...")
        predictions = {}
        for name, path in tqdm(image_paths.items()):
            pred = self._predict_single(path)
            predictions[name] = pred
        print(f"✓ Predicted {len(predictions)} images")
        return predictions

    def _predict_single(self, image_path: str):
        """Predict single image."""
        results = self.model.predict(
            image_path, verbose=False, retina_masks=True
        )
        return self._extract_masks(results[0])

    def _extract_masks(self, result):
        """Extract masks from result."""
        if result.masks is None:
            return {
                'masks': [],
                'boxes': [],
                'confidences': [],
                'classes': []
            }
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        return {
            'masks': masks,
            'boxes': boxes,
            'confidences': confs,
            'classes': classes
        }


class UNetPredictor:
    """Predictor for U-Net models."""

    def __init__(self, model, device, img_size: tuple):
        """Initialize predictor."""
        self.model = model
        self.device = device
        self.img_size = img_size
        self.transform = self._create_transform()

    def _create_transform(self):
        """Create preprocessing transform."""
        return A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    def predict(self, image_paths: dict):
        """Predict on images."""
        print("Predicting with U-Net...")
        predictions = {}
        with torch.no_grad():
            for name, path in tqdm(image_paths.items()):
                pred = self._predict_single(path)
                predictions[name] = pred
        print(f"✓ Predicted {len(predictions)} images")
        return predictions

    def _predict_single(self, image_path: str):
        """Predict single image."""
        image = self._load_image(image_path)
        tensor = self._preprocess(image)
        output = self.model(tensor)
        mask = self._postprocess(output, image.shape[:2])
        return mask

    def _load_image(self, path: str):
        """Load image as numpy array."""
        return np.array(Image.open(path).convert('RGB'))

    def _preprocess(self, image):
        """Preprocess image."""
        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess(self, output, original_size):
        """Postprocess output to mask."""
        pred = torch.argmax(output, dim=1)
        mask = pred.cpu().numpy()[0]
        mask = self._resize_mask(mask, original_size)
        return mask

    def _resize_mask(self, mask, target_size):
        """Resize mask to original size."""
        from PIL import Image
        mask_img = Image.fromarray(mask.astype(np.uint8))
        resized = mask_img.resize(
            (target_size[1], target_size[0]),
            Image.NEAREST
        )
        return np.array(resized)
