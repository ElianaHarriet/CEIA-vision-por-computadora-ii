"""Mask flattening for instance segmentation."""
import numpy as np


class MaskFlattener:
    """Flattens instance masks to semantic format.

    The ``class_aware`` strategy produces a multiclass mask (0-4) so
    instance predictions can be compared against the semantic ground
    truth in the same label space. Overlapping instances are resolved
    by confidence: the highest-confidence instance wins each pixel.
    """

    def __init__(self, strategy='or', class_map=None):
        """Initialize flattener."""
        self.strategy = strategy
        self.class_map = class_map

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
        if self.strategy == 'class_aware':
            confs = prediction['confidences']
            classes = prediction.get('classes', [])
            return self._flatten_class_aware(masks, confs, classes)
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

    def _flatten_class_aware(self, masks, confidences, classes):
        """Flatten into a multiclass mask preserving class information.

        Each instance is painted with ``class_map[cls]`` (default:
        semantic value = YOLO class id + 1). Overlaps are resolved by
        confidence, so the highest-confidence instance wins.
        """
        if len(masks) == 0:
            return None
        class_map = self.class_map or {i: i + 1 for i in range(4)}
        result = np.zeros_like(masks[0], dtype=np.float32)
        conf_map = np.zeros_like(masks[0], dtype=np.float32)
        for mask, conf, cls in zip(masks, confidences, classes):
            cls_value = class_map.get(int(cls), 1)
            mask_bool = mask > 0.5
            update_idx = (mask_bool) & (conf > conf_map)
            result[update_idx] = cls_value
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
