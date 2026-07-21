"""Mask flattening for instance segmentation."""
import numpy as np


class MaskFlattener:
    """Flattens instance masks to semantic format."""

    def __init__(self, strategy='or'):
        """Initialize flattener."""
        self.strategy = strategy

    def flatten_predictions(self, predictions: dict):
        """Flatten all predictions."""
        print(f"Flattening masks (strategy: {self.strategy})...")
        flattened = {}
        for name, pred in predictions.items():
            flat_mask = self._flatten_single(pred)
            flattened[name] = flat_mask
        print(f"✓ Flattened {len(flattened)} predictions")
        return flattened

    def _flatten_single(self, prediction: dict):
        """Flatten single prediction."""
        masks = prediction['masks']
        if len(masks) == 0:
            return None
        if self.strategy == 'or':
            return self._flatten_or(masks)
        if self.strategy == 'confidence':
            confs = prediction['confidences']
            return self._flatten_confidence(masks, confs)
        if self.strategy == 'last':
            return self._flatten_last(masks)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _flatten_or(self, masks):
        """Flatten using OR logic."""
        if len(masks) == 0:
            return None
        result = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            result = np.logical_or(result, mask > 0.5)
        return result.astype(np.uint8)

    def _flatten_confidence(self, masks, confidences):
        """Flatten prioritizing high confidence."""
        if len(masks) == 0:
            return None
        result = np.zeros_like(masks[0], dtype=np.float32)
        conf_map = np.zeros_like(masks[0], dtype=np.float32)
        for mask, conf in zip(masks, confidences):
            mask_bool = mask > 0.5
            update_idx = (mask_bool) & (conf > conf_map)
            result[update_idx] = 1
            conf_map[update_idx] = conf
        return result.astype(np.uint8)

    def _flatten_last(self, masks):
        """Flatten with last instance winning."""
        if len(masks) == 0:
            return None
        result = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            mask_bool = mask > 0.5
            result[mask_bool] = 1
        return result
