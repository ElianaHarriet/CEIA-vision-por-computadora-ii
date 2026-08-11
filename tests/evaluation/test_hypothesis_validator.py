"""Tests for the hypothesis validator bootstrap CIs."""
import numpy as np

from evaluation.hypothesis_validator import HypothesisValidator


def _metrics(per_image):
    return {"aggregated": {"mean_iou": np.mean([m["iou_global"] for m in per_image.values()])},
            "per_image": per_image}


def _image(iou):
    return {"iou_global": iou}


class TestBootstrapCI:
    def test_ci_reproducible_with_seed(self):
        v1 = HypothesisValidator(seed=42, n_bootstrap=500)
        v2 = HypothesisValidator(seed=42, n_bootstrap=500)
        values = [0.5, 0.6, 0.4, 0.7, 0.8]
        assert v1._bootstrap_ci(values) == v2._bootstrap_ci(values)

    def test_ci_difference_reproducible_with_seed(self):
        v1 = HypothesisValidator(seed=7, n_bootstrap=500)
        v2 = HypothesisValidator(seed=7, n_bootstrap=500)
        a = [0.5, 0.6, 0.4, 0.7]
        b = [0.4, 0.3, 0.5]
        assert v1._bootstrap_ci_difference(a, b) == v2._bootstrap_ci_difference(a, b)

    def test_ci_empty_returns_zero(self):
        v = HypothesisValidator()
        assert v._bootstrap_ci([]) == (0.0, 0.0)

    def test_ci_bounds_ordering(self):
        v = HypothesisValidator(seed=1, n_bootstrap=300)
        low, high = v._bootstrap_ci([0.2, 0.4, 0.5, 0.3, 0.6, 0.7])
        assert low <= high


class TestValidation:
    def test_hypothesis_refuted_when_instance_not_higher(self):
        inst = _metrics({"a": _image(0.5), "b": _image(0.6)})
        sem = _metrics({"a": _image(0.7), "b": _image(0.8)})
        result = HypothesisValidator(n_bootstrap=200).validate(inst, sem)
        assert result["validated"] is False

    def test_result_contains_ci_keys(self):
        inst = _metrics({"a": _image(0.8), "b": _image(0.7)})
        sem = _metrics({"a": _image(0.6), "b": _image(0.5)})
        result = HypothesisValidator(n_bootstrap=200).validate(inst, sem)
        assert "ci_iou_instance" in result
        assert "ci_iou_semantic" in result
        assert "ci_difference" in result

    def test_p_value_high_when_samples_tiny(self):
        inst = _metrics({"a": _image(0.5)})
        sem = _metrics({"a": _image(0.4)})
        result = HypothesisValidator(n_bootstrap=100).validate(inst, sem)
        assert result["p_value"] == 1.0