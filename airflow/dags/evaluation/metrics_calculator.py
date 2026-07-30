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
        """Calculate metrics per image."""
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
        return {
            'iou_global': self._calculate_iou(pred, gt),
            'iou_per_class': self._calculate_iou_per_class(pred, gt),
            'pixel_accuracy': self._calculate_pixel_acc(pred, gt),
            'precision': self._calculate_precision(pred, gt),
            'recall': self._calculate_recall(pred, gt),
            'f1_score': self._calculate_f1(pred, gt),
        }

    def _calculate_iou(self, pred, gt):
        """Calculate global IoU."""
        pred_bin = pred > 0
        gt_bin = gt > 0
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        return float(intersection / union) if union > 0 else 0.0

    def _calculate_iou_per_class(self, pred, gt):
        """Calculate IoU per class."""
        ious = []
        for cls in range(self.num_classes):
            pred_cls = pred == cls
            gt_cls = gt == cls
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            iou = intersection / union if union > 0 else 0.0
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

    def _aggregate_metrics(self, per_image: dict):
        """Aggregate per-image metrics."""
        if not per_image:
            return {}
        all_ious = [m['iou_global'] for m in per_image.values()]
        all_pa = [m['pixel_accuracy'] for m in per_image.values()]
        all_prec = [m['precision'] for m in per_image.values()]
        all_rec = [m['recall'] for m in per_image.values()]
        all_f1 = [m['f1_score'] for m in per_image.values()]
        return {
            'mean_iou': float(np.mean(all_ious)),
            'std_iou': float(np.std(all_ious)),
            'mean_pixel_accuracy': float(np.mean(all_pa)),
            'mean_precision': float(np.mean(all_prec)),
            'mean_recall': float(np.mean(all_rec)),
            'mean_f1_score': float(np.mean(all_f1)),
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
        print("✓ Comparison calculated")
        return comparison
