"""Tests for the hypothesis validator bootstrap CIs."""
import numpy as np
import pytest

from evaluation.hypothesis_validator import HypothesisValidator


def _metrics(per_image):
    return {"aggregated": {"mean_iou": np.mean([m["iou_global"] for m in per_image.values()])},
            "per_image": per_image}


def _image(iou):
    return {"iou_global": iou}


class TestBootstrapCI:
    def test_ci_reproducible_with_seed(self):
        v1 = HypothesisValidator(seed=2026, n_bootstrap=500)
        v2 = HypothesisValidator(seed=2026, n_bootstrap=500)
        values = [0.5, 0.6, 0.4, 0.7, 0.8]
        assert v1._bootstrap_ci(values) == v2._bootstrap_ci(values)

    def test_ci_difference_reproducible_with_seed(self):
        v1 = HypothesisValidator(seed=7, n_bootstrap=500)
        v2 = HypothesisValidator(seed=7, n_bootstrap=500)
        a = [0.5, 0.6, 0.4, 0.7]
        b = [0.4, 0.3, 0.5, 0.6]
        assert v1._bootstrap_ci_difference(a, b) == v2._bootstrap_ci_difference(a, b)

    def test_ci_difference_requires_equal_sizes(self):
        v = HypothesisValidator()
        with pytest.raises(ValueError):
            v._bootstrap_ci_difference([0.5, 0.6, 0.4], [0.4, 0.3])

    def test_ci_empty_returns_zero(self):
        v = HypothesisValidator()
        assert v._bootstrap_ci([]) == (0.0, 0.0)

    def test_ci_difference_empty_returns_zero(self):
        v = HypothesisValidator()
        assert v._bootstrap_ci_difference([], []) == (0.0, 0.0)

    def test_ci_n_bootstrap_zero_returns_zero(self):
        v = HypothesisValidator(n_bootstrap=0)
        assert v._bootstrap_ci([0.2, 0.4, 0.5]) == (0.0, 0.0)
        assert v._bootstrap_ci_difference([0.5, 0.6], [0.4, 0.3]) == (0.0, 0.0)

    def test_ci_bounds_ordering(self):
        v = HypothesisValidator(seed=1, n_bootstrap=300)
        low, high = v._bootstrap_ci([0.2, 0.4, 0.5, 0.3, 0.6, 0.7])
        assert low <= high


class TestPairedAlignment:
    def test_paired_ious_uses_common_subset(self):
        inst = _metrics({"a": _image(0.8), "b": _image(0.7), "x": _image(0.9)})
        sem = _metrics({"a": _image(0.6), "b": _image(0.5), "y": _image(0.1)})
        a, b = HypothesisValidator._paired_ious(inst, sem)
        # Only 'a' and 'b' are evaluated by both models, sorted by name.
        assert a == [0.8, 0.7]
        assert b == [0.6, 0.5]

    def test_paired_ious_stable_alignment(self):
        inst = _metrics({"b": _image(0.7), "a": _image(0.8)})
        sem = _metrics({"a": _image(0.6), "b": _image(0.5)})
        a, b = HypothesisValidator._paired_ious(inst, sem)
        assert a == [0.8, 0.7]
        assert b == [0.6, 0.5]

    def test_validate_reports_common_subset_size(self):
        inst = _metrics({"a": _image(0.8), "b": _image(0.7), "x": _image(0.9)})
        sem = _metrics({"a": _image(0.6), "b": _image(0.5), "y": _image(0.1)})
        result = HypothesisValidator(n_bootstrap=200).validate(inst, sem)
        assert result["n_images_common"] == 2
        assert result["n_images_instance"] == 3
        assert result["n_images_semantic"] == 3


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
        assert "n_images_common" in result

    def test_p_value_high_when_samples_tiny(self):
        inst = _metrics({"a": _image(0.5)})
        sem = _metrics({"a": _image(0.4)})
        result = HypothesisValidator(n_bootstrap=100).validate(inst, sem)
        assert result["p_value"] == 1.0

    def test_hypothesis_validated_requires_ci_excluding_zero(self):
        # Instance higher, paired t-test significant, CI of difference
        # excludes 0 -> validated. (Differences must vary: a constant
        # offset is a degenerate case and is correctly not validated.)
        inst = _metrics({"a": _image(0.95), "b": _image(0.90),
                         "c": _image(0.93), "d": _image(0.91)})
        sem = _metrics({"a": _image(0.30), "b": _image(0.20),
                        "c": _image(0.28), "d": _image(0.22)})
        result = HypothesisValidator(n_bootstrap=500).validate(inst, sem)
        assert result["validated"] is True

    def test_hypothesis_refuted_when_ci_contains_zero(self):
        # Instance higher but CI of difference contains 0 -> not validated.
        inst = _metrics({"a": _image(0.60), "b": _image(0.50),
                         "c": _image(0.55), "d": _image(0.52)})
        sem = _metrics({"a": _image(0.59), "b": _image(0.51),
                        "c": _image(0.53), "d": _image(0.52)})
        result = HypothesisValidator(n_bootstrap=500).validate(inst, sem)
        # Very small, overlapping means: p will be high and CI will
        # contain 0, so the hypothesis must be refuted.
        assert result["validated"] is False

    def test_degenerate_paired_data_does_not_validate(self):
        """Identical per-image differences yield p=nan/0 from scipy; the
        gate must not treat that as a significant result."""
        inst = _metrics({"a": _image(0.9), "b": _image(0.9),
                         "c": _image(0.9), "d": _image(0.9)})
        sem = _metrics({"a": _image(0.5), "b": _image(0.5),
                        "c": _image(0.5), "d": _image(0.5)})
        result = HypothesisValidator(n_bootstrap=500).validate(inst, sem)
        # p must be finite and the hypothesis must NOT spuriously validate.
        assert np.isfinite(result["p_value"])
        assert result["p_value"] == 1.0
        assert result["validated"] is False

    def test_insufficient_common_data_refutes_without_crash(self):
        inst = _metrics({"a": _image(0.5)})
        sem = _metrics({"a": _image(0.4), "b": _image(0.6)})
        result = HypothesisValidator().validate(inst, sem)
        assert result["validated"] is False
        assert result["p_value"] == 1.0
        assert "Insufficient common data" in result["evidence"]

    def test_no_overlap_refutes_without_crash(self):
        inst = _metrics({"a": _image(0.5), "b": _image(0.6)})
        sem = _metrics({"c": _image(0.4), "d": _image(0.6)})
        result = HypothesisValidator().validate(inst, sem)
        assert result["validated"] is False
        assert result["n_images_common"] == 0

    def test_means_are_computed_on_common_subset(self):
        inst = _metrics({"a": _image(0.8), "b": _image(0.7), "x": _image(0.9)})
        sem = _metrics({"a": _image(0.6), "b": _image(0.5), "y": _image(0.1)})
        result = HypothesisValidator(n_bootstrap=200).validate(inst, sem)
        # common = {a, b}: instance mean (0.8+0.7)/2, semantic (0.6+0.5)/2
        assert abs(result["mean_iou_instance"] - 0.75) < 1e-9
        assert abs(result["mean_iou_semantic"] - 0.55) < 1e-9
        # Full-set means kept as context.
        assert abs(result["full_mean_iou_instance"] - 0.8) < 1e-9
        assert abs(result["full_mean_iou_semantic"] - 0.4) < 1e-9