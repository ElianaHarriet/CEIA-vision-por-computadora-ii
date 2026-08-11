"""Tests for the metrics calculator.

These tests cover the scientific fixes applied to the evaluation:
- IoU = 1.0 for images without damage (no spurious zeros).
- Binary precision/recall/F1 computed on a consistent binary damage mask.
- Per-class IoU micro-averaged over the whole test set.
"""
import numpy as np
import pytest

from evaluation.metrics_calculator import MetricsCalculator, ComparisonCalculator


def _single_per_image(gt, pred):
    calculator = MetricsCalculator(num_classes=5)
    raw = calculator._calculate_single(pred, gt)
    return raw


@pytest.fixture
def calculator():
    return MetricsCalculator(num_classes=5)


class TestIoDNoDamage:
    def test_iou_is_one_when_no_damage_anywhere(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        pred = np.zeros((10, 10), dtype=np.uint8)
        assert calculator._calculate_iou(pred, gt) == 1.0

    def test_iou_is_one_when_only_no_damage_class_present(self, calculator):
        gt = np.full((10, 10), 3, dtype=np.uint8)
        pred = np.full((10, 10), 3, dtype=np.uint8)
        assert calculator._calculate_iou(pred, gt) == 1.0

    def test_iou_zero_when_damage_missed(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[2:4, 2:4] = 1
        pred = np.zeros((10, 10), dtype=np.uint8)
        assert calculator._calculate_iou(pred, gt) == 0.0

    def test_iou_perfect_when_damage_matches(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[2:4, 2:4] = 1
        pred = gt.copy()
        assert calculator._calculate_iou(pred, gt) == 1.0


class TestBinaryMetrics:
    def test_binary_precision(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[2:4, 2:4] = 1
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[2:4, 2:4] = 1
        pred[6:8, 6:8] = 1
        assert calculator._binary_precision(pred, gt) == 0.5

    def test_binary_recall(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[2:4, 2:4] = 1
        gt[6:8, 6:8] = 1
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[2:4, 2:4] = 1
        assert calculator._binary_recall(pred, gt) == 0.5

    def test_binary_f1(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[2:4, 2:4] = 1
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[2:4, 2:4] = 1
        assert calculator._binary_f1(pred, gt) == 1.0

    def test_binary_precision_zero_when_no_prediction(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[2:4, 2:4] = 1
        pred = np.zeros((10, 10), dtype=np.uint8)
        assert calculator._binary_precision(pred, gt) == 0.0

    def test_no_damage_class_is_background(self, calculator):
        """Class 3 (No Damage) must not count as damage in the binary mask."""
        gt = np.full((10, 10), 3, dtype=np.uint8)
        pred = np.full((10, 10), 3, dtype=np.uint8)
        tp, fp, fn = calculator._binary_components(pred, gt)
        assert (tp, fp, fn) == (0, 0, 0)


class TestPerClassIoU:
    def test_per_class_micro_average(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0:5, 0:10] = 4
        pred = np.full((10, 10), 3, dtype=np.uint8)
        pred[0:5, 0:5] = 4
        per_class = calculator._calculate_iou_per_class(pred, gt)
        # Class 4: tp=25, fp=0, fn=25 -> 0.5
        assert per_class[4] == 0.5
        # Classes never present get 1.0 (no spurious zero).
        assert per_class[1] == 1.0

    def test_absent_class_iou_is_one(self, calculator):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0:5, 0:10] = 4
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[0:5, 0:5] = 4
        per_class = calculator._calculate_iou_per_class(pred, gt)
        # Class 1 never appears in either, but background (0) does.
        assert per_class[1] == 1.0
        assert per_class[2] == 1.0
        assert per_class[3] == 1.0


def test_aggregate_per_class_iou_micro_averaged(calculator):
    per_image = {}
    for i in range(3):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0:5, 0:10] = 4
        pred = np.full((10, 10), 3, dtype=np.uint8)
        pred[0:5, 0:5] = 4
        per_image[f"img_{i}"] = {
            **{k: 0.0 for k in [
                'iou_global', 'pixel_accuracy', 'precision', 'recall',
                'f1_score', 'binary_precision', 'binary_recall', 'binary_f1',
                'area_error', 'relative_area_error', 'iou_per_class',
            ]},
            'confusion': calculator._calculate_confusion(pred, gt),
        }
    agg = calculator._aggregate_per_class_iou(per_image)
    # TP=75, FP=0, FN=75 per whole test set -> 0.5, not a simple mean.
    assert agg['iou']['4'] == 0.5
    assert agg['n_images_with_class'][4] == 3


class TestAllMetricsPipeline:
    def test_calculate_all_metrics_includes_binary(self):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[2:4, 2:4] = 1
        pred = gt.copy()
        calc = MetricsCalculator(num_classes=5)
        result = calc.calculate_all_metrics({"img": pred}, {"img": gt})
        agg = result["aggregated"]
        assert "mean_binary_precision" in agg
        assert "mean_binary_recall" in agg
        assert "mean_binary_f1" in agg
        assert agg["mean_binary_f1"] == 1.0

    def test_no_damage_image_keeps_iou_in_aggregate(self):
        gt = np.zeros((10, 10), dtype=np.uint8)
        pred = np.zeros((10, 10), dtype=np.uint8)
        calc = MetricsCalculator(num_classes=5)
        result = calc.calculate_all_metrics({"img": pred}, {"img": gt})
        assert result["aggregated"]["mean_iou"] == 1.0


class TestComparisonPerClass:
    def test_comparison_has_per_class_delta(self):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[0:5, 0:10] = 4
        pred = np.full((10, 10), 3, dtype=np.uint8)
        pred[0:5, 0:5] = 4
        calc = MetricsCalculator(num_classes=5)
        metrics_a = calc.calculate_all_metrics({"img": pred}, {"img": gt})
        metrics_b = calc.calculate_all_metrics({"img": pred}, {"img": gt})
        comp = ComparisonCalculator().compare_metrics(metrics_a, metrics_b)
        assert "per_class_iou" in comp
        assert "0" in comp["per_class_iou"]
        assert comp["per_class_iou"]["0"]["difference"] == 0.0