"""Metrics calculator for evaluation."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class MetricsCalculator:
    """Calculator for segmentation metrics."""

    def __init__(self, num_classes: int):
        """Initialize calculator."""
        self.num_classes = num_classes

    def calculate_all_metrics(self, pred_dict, gt_dict):
        """Calculate all metrics."""
        print("Calculating metrics...")
        per_image = self._calculate_per_image(pred_dict, gt_dict)
        aggregated = self._aggregate_metrics(per_image)
        print("✓ Metrics calculated")
        return {
            'per_image': per_image,
            'aggregated': aggregated
        }

    def _calculate_per_image(self, pred_dict, gt_dict):
        """Calculate metrics per image.

        Images without detections (pred is None) are skipped, so the
        per-image and aggregated metrics are computed over the subset of
        images where the model produced a prediction. This means the
        instance and semantic evaluation sets may differ in size.
        """
        results = {}
        for name in pred_dict.keys():
            if name not in gt_dict:
                continue
            pred = pred_dict[name]
            gt = gt_dict[name]
            if pred is None:
                continue
            metrics = self._calculate_single(pred, gt)
            results[name] = metrics
        return results

    def _calculate_single(self, pred, gt):
        """Calculate metrics for single image."""
        pred = self._sanitize_mask(pred)
        gt = self._sanitize_mask(gt)
        return {
            'iou_global': self._calculate_iou(pred, gt),
            'iou_per_class': self._calculate_iou_per_class(pred, gt),
            'confusion': self._calculate_confusion(pred, gt),
            'pixel_accuracy': self._calculate_pixel_acc(pred, gt),
            'precision': self._calculate_precision(pred, gt),
            'recall': self._calculate_recall(pred, gt),
            'f1_score': self._calculate_f1(pred, gt),
            'binary_precision': self._binary_precision(pred, gt),
            'binary_recall': self._binary_recall(pred, gt),
            'binary_f1': self._binary_f1(pred, gt),
            'area_error': self._calculate_area_error(pred, gt),
            'relative_area_error': self._calculate_relative_area_error(pred, gt),
        }

    def _sanitize_mask(self, mask):
        """Normalize a mask so downstream metrics are well-defined.

        - NaN pixels are treated as background (0).
        - Values outside [0, num_classes-1] are clipped, so a stray
          value (e.g. 255) is not silently counted as damage.
        """
        mask = np.asarray(mask, dtype=np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        mask = np.clip(mask, 0, self.num_classes - 1)
        return mask.astype(np.uint8)

    def _calculate_confusion(self, pred, gt):
        """Per-class confusion counts: [tp, fp, fn]."""
        conf = []
        for cls in range(self.num_classes):
            pred_cls = pred == cls
            gt_cls = gt == cls
            conf.append({
                'tp': int(np.logical_and(pred_cls, gt_cls).sum()),
                'fp': int(np.logical_and(pred_cls, ~gt_cls).sum()),
                'fn': int(np.logical_and(~pred_cls, gt_cls).sum()),
            })
        return conf

    def _binary_components(self, pred, gt):
        """True/false positives and negatives on the binary damage mask."""
        pred_bin = self._damage_mask(pred)
        gt_bin = self._damage_mask(gt)
        tp = int(np.logical_and(pred_bin, gt_bin).sum())
        fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
        fn = int(np.logical_and(~pred_bin, gt_bin).sum())
        return tp, fp, fn

    def _binary_precision(self, pred, gt):
        """Precision on the binary damage mask (Opción B)."""
        tp, fp, _ = self._binary_components(pred, gt)
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def _binary_recall(self, pred, gt):
        """Recall on the binary damage mask (Opción B)."""
        tp, _, fn = self._binary_components(pred, gt)
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def _binary_f1(self, pred, gt):
        """F1 on the binary damage mask (Opción B)."""
        p = self._binary_precision(pred, gt)
        r = self._binary_recall(pred, gt)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @staticmethod
    def _damage_mask(mask):
        """Binarize to damage pixels per the binary comparison option.

        In the shared 5-class label space (0=Background, 1=Dent,
        2=Scratch, 3=No Damage, 4=Severe) only classes 1, 2, 4 count as
        damage; No Damage (3) is treated as background.
        """
        return (mask >= 1) & (mask != 3)

    def _calculate_iou(self, pred, gt):
        """Calculate global IoU on the binary damage mask.

        When neither GT nor prediction contain damage (union == 0) the
        image is a correct "no damage" case, so its IoU is 1.0 rather than
        0.0 — otherwise it would deflate the mean with a spurious miss.
        """
        pred_bin = self._damage_mask(pred)
        gt_bin = self._damage_mask(gt)
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)

    def _calculate_iou_per_class(self, pred, gt):
        """Calculate IoU per class.

        A class absent from both prediction and ground truth (union == 0)
        scores 1.0, since neither side claims it; only a predicted class
        that does not exist in GT (false positive) scores 0.0.
        """
        ious = []
        for cls in range(self.num_classes):
            pred_cls = pred == cls
            gt_cls = gt == cls
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            iou = 1.0 if union == 0 else intersection / union
            ious.append(float(iou))
        return ious

    def _calculate_pixel_acc(self, pred, gt):
        """Calculate pixel accuracy."""
        correct = (pred == gt).sum()
        total = gt.size
        return float(correct / total)

    def _calculate_precision(self, pred, gt):
        """Calculate precision."""
        metrics = self._get_sklearn_metrics(pred, gt)
        return float(np.mean(metrics[0]))

    def _calculate_recall(self, pred, gt):
        """Calculate recall."""
        metrics = self._get_sklearn_metrics(pred, gt)
        return float(np.mean(metrics[1]))

    def _calculate_f1(self, pred, gt):
        """Calculate F1 score."""
        metrics = self._get_sklearn_metrics(pred, gt)
        return float(np.mean(metrics[2]))

    def _get_sklearn_metrics(self, pred, gt):
        """Get sklearn precision, recall, f1."""
        labels = list(range(self.num_classes))
        return precision_recall_fscore_support(
            gt.flatten(),
            pred.flatten(),
            labels=labels,
            average=None,
            zero_division=0
        )

    def _calculate_area_error(self, pred, gt):
        """Calculate absolute damage-area error in pixels."""
        pred_area = self._damage_mask(pred).sum()
        gt_area = self._damage_mask(gt).sum()
        return float(abs(pred_area - gt_area))

    def _calculate_relative_area_error(self, pred, gt):
        """Calculate relative area error |A_pred - A_gt| / A_gt."""
        pred_area = self._damage_mask(pred).sum()
        gt_area = self._damage_mask(gt).sum()
        if gt_area == 0:
            return 0.0
        return float(abs(pred_area - gt_area) / gt_area)

    def _aggregate_metrics(self, per_image: dict):
        """Aggregate per-image metrics."""
        if not per_image:
            return {}
        all_ious = [m['iou_global'] for m in per_image.values()]
        all_pa = [m['pixel_accuracy'] for m in per_image.values()]
        all_prec = [m['precision'] for m in per_image.values()]
        all_rec = [m['recall'] for m in per_image.values()]
        all_f1 = [m['f1_score'] for m in per_image.values()]
        all_bin_prec = [m['binary_precision'] for m in per_image.values()]
        all_bin_rec = [m['binary_recall'] for m in per_image.values()]
        all_bin_f1 = [m['binary_f1'] for m in per_image.values()]
        all_area_err = [m['area_error'] for m in per_image.values()]
        all_rel_area_err = [m['relative_area_error'] for m in per_image.values()]
        return {
            'mean_iou': float(np.mean(all_ious)),
            'std_iou': float(np.std(all_ious)),
            'mean_pixel_accuracy': float(np.mean(all_pa)),
            'mean_precision': float(np.mean(all_prec)),
            'mean_recall': float(np.mean(all_rec)),
            'mean_f1_score': float(np.mean(all_f1)),
            'mean_binary_precision': float(np.mean(all_bin_prec)),
            'mean_binary_recall': float(np.mean(all_bin_rec)),
            'mean_binary_f1': float(np.mean(all_bin_f1)),
            'mean_area_error': float(np.mean(all_area_err)),
            'mean_relative_area_error': float(np.mean(all_rel_area_err)),
            'per_class_iou': self._aggregate_per_class_iou(per_image),
            'n_images': len(per_image),
        }

    def _aggregate_per_class_iou(self, per_image: dict):
        """Micro-averaged IoU per class: sum(TP) / sum(TP+FP+FN).

        Accumulates the confusion counts over all images so a class present
        in only a few images contributes proportionally, instead of being
        diluted by a mean over images where it never appears.
        """
        tp = [0] * self.num_classes
        fp = [0] * self.num_classes
        fn = [0] * self.num_classes
        present = [0] * self.num_classes
        for m in per_image.values():
            for cls, conf in enumerate(m['confusion']):
                tp[cls] += conf['tp']
                fp[cls] += conf['fp']
                fn[cls] += conf['fn']
                if conf['tp'] + conf['fn'] > 0 or conf['fp'] > 0:
                    present[cls] += 1
        ious = {}
        for cls in range(self.num_classes):
            denom = tp[cls] + fp[cls] + fn[cls]
            # A class absent from every image scores 1.0, mirroring the
            # per-image convention in _calculate_iou_per_class (a class no
            # side claims is a correct "not present" prediction, not a miss).
            ious[str(cls)] = float(tp[cls] / denom) if denom > 0 else 1.0
        return {
            'iou': ious,
            'n_images_with_class': present,
        }


class ComparisonCalculator:
    """Calculator for model comparison."""

    def compare_metrics(self, metrics_a: dict, metrics_b: dict):
        """Compare metrics between models."""
        print("Comparing metrics...")
        agg_a = metrics_a['aggregated']
        agg_b = metrics_b['aggregated']
        comparison = {}
        for key in agg_a.keys():
            if key.startswith('mean_'):
                val_a = agg_a[key]
                val_b = agg_b[key]
                comparison[key] = {
                    'model_a': val_a,
                    'model_b': val_b,
                    'difference': val_a - val_b,
                    'relative_diff': ((val_a - val_b) / val_b * 100)
                    if val_b != 0 else 0
                }
        per_class_a = agg_a.get('per_class_iou', {}).get('iou', {})
        per_class_b = agg_b.get('per_class_iou', {}).get('iou', {})
        comparison['per_class_iou'] = {
            str(k): {
                'model_a': per_class_a.get(str(k), 0.0),
                'model_b': per_class_b.get(str(k), 0.0),
                'difference': per_class_a.get(str(k), 0.0)
                - per_class_b.get(str(k), 0.0),
            }
            for k in set(per_class_a) | set(per_class_b)
        }
        print("✓ Comparison calculated")
        return comparison
