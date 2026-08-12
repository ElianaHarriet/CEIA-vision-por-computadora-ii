"""Smoke tests for the comparison visualizer."""
import matplotlib
matplotlib.use("Agg")
import numpy as np

from evaluation.visualizer import ComparisonVisualizer


def _comparison():
    return {
        "mean_iou": {"model_a": 0.63, "model_b": 0.62,
                     "difference": 0.01, "relative_diff": 1.6},
        "mean_binary_precision": {"model_a": 0.63, "model_b": 0.65,
                                  "difference": -0.02, "relative_diff": -3.0},
        "mean_binary_recall": {"model_a": 0.72, "model_b": 0.64,
                               "difference": 0.08, "relative_diff": 12.5},
        "mean_binary_f1": {"model_a": 0.63, "model_b": 0.61,
                           "difference": 0.02, "relative_diff": 3.3},
        "per_class_iou": {
            "0": {"model_a": 0.88, "model_b": 0.89, "difference": -0.01},
            "1": {"model_a": 0.40, "model_b": 0.40, "difference": 0.0},
            "2": {"model_a": 0.39, "model_b": 0.53, "difference": -0.14},
            "3": {"model_a": 0.60, "model_b": 0.55, "difference": 0.05},
            "4": {"model_a": 0.48, "model_b": 0.53, "difference": -0.05},
        },
    }


def _samples():
    return {
        "img1": {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "gt": np.zeros((8, 8), dtype=np.uint8),
            "pred_instance": np.zeros((8, 8), dtype=np.uint8),
            "pred_semantic": np.zeros((8, 8), dtype=np.uint8),
            "iou_instance": 0.5,
            "iou_semantic": 0.6,
        }
    }


class TestVisualizerSmoke:
    def test_generates_all_visualizations(self, tmp_path):
        viz = ComparisonVisualizer(str(tmp_path),
                                   class_names=["Bg", "Dent", "Scratch",
                                                "No", "Severe"])
        data = {"comparison": _comparison(), "samples": _samples()}
        viz.generate_all_visualizations(data)
        assert (tmp_path / "iou_comparison.png").exists()
        assert (tmp_path / "per_class_iou.png").exists()
        assert (tmp_path / "metrics_comparison.png").exists()
        assert (tmp_path / "sample_predictions.png").exists()

    def test_empty_samples_skips_sample_plot(self, tmp_path):
        viz = ComparisonVisualizer(str(tmp_path),
                                   class_names=["Bg", "Dent", "Scratch",
                                                "No", "Severe"])
        data = {"comparison": _comparison(), "samples": {}}
        viz.generate_all_visualizations(data)
        assert not (tmp_path / "sample_predictions.png").exists()

    def test_empty_per_class_skips_chart(self, tmp_path):
        viz = ComparisonVisualizer(str(tmp_path),
                                   class_names=["Bg", "Dent", "Scratch",
                                                "No", "Severe"])
        comp = _comparison()
        comp["per_class_iou"] = {}
        data = {"comparison": comp, "samples": _samples()}
        viz.generate_all_visualizations(data)
        assert not (tmp_path / "per_class_iou.png").exists()