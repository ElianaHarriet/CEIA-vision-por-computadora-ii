"""Tests for the result report generator (per-class + CI sections)."""
from evaluation.report_generator import ComparisonReport


def _comparison():
    return {
        "mean_iou": {"model_a": 0.53, "model_b": 0.52,
                     "difference": 0.01, "relative_diff": 1.9},
        "mean_relative_area_error": {"model_a": 0.63, "model_b": 0.45,
                                     "difference": 0.18, "relative_diff": 40.0},
        "mean_binary_precision": {"model_a": 0.6, "model_b": 0.55,
                                  "difference": 0.05, "relative_diff": 9.0},
        "mean_binary_recall": {"model_a": 0.7, "model_b": 0.65,
                               "difference": 0.05, "relative_diff": 7.7},
        "mean_binary_f1": {"model_a": 0.65, "model_b": 0.6,
                           "difference": 0.05, "relative_diff": 8.3},
        "per_class_iou": {
            "0": {"model_a": 1.0, "model_b": 1.0, "difference": 0.0},
            "4": {"model_a": 0.5, "model_b": 0.4, "difference": 0.1},
        },
    }


def _hypothesis():
    return {
        "validated": False,
        "p_value": 0.8,
        "evidence": "no significant difference",
        "conclusion": "refuted",
        "ci_iou_instance": (0.4, 0.6),
        "ci_iou_semantic": (0.39, 0.59),
        "ci_difference": (-0.15, 0.17),
    }


def _data():
    return {"comparison": _comparison(), "hypothesis": _hypothesis(),
            "samples": {}}


class TestReportGenerator:
    def test_report_contains_per_class_iou(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        report_lines = report._build_report(_data())
        assert "## Per-Class IoU" in report_lines
        assert "| Severe |" in report_lines

    def test_report_contains_confidence_intervals(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        report_lines = report._build_report(_data())
        assert "## Confidence Intervals" in report_lines
        assert "contains 0" in report_lines

    def test_report_contains_binary_metrics_section(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        report_lines = report._build_report(_data())
        assert "Binary Damage" in report_lines
        assert "Binary F1" in report_lines

    def test_generate_report_writes_files(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        path = report.generate_report(_data())
        assert (tmp_path / "comparison_report.md").exists()
        assert (tmp_path / "comparison_data.json").exists()
        content = (tmp_path / "comparison_report.md").read_text()
        assert "Conclusion" in content