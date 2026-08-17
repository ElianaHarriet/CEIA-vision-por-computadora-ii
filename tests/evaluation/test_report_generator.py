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
        "n_images_common": 108,
        "n_images_instance": 108,
        "n_images_semantic": 119,
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
        assert "Binary Precision" in report_lines

    def test_report_labels_macro_and_binary_separately(self, tmp_path):
        """Macro (5-class) metrics must NOT be presented as binary."""
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        data = _data()
        data["comparison"]["mean_precision"] = {
            "model_a": 0.32, "model_b": 0.33,
            "difference": -0.01, "relative_diff": -2.5,
        }
        report_lines = report._build_report(data)
        # The macro precision must be labeled as such, not as a binary row.
        assert "Precision (macro, 5 clases)" in report_lines

    def test_report_mentions_common_subset(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        report_lines = report._build_report(_data())
        assert "Common subset" in report_lines
        assert "108" in report_lines

    def test_report_handles_missing_per_class(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        data = _data()
        data["comparison"] = {
            "mean_iou": {"model_a": 0.5, "model_b": 0.4,
                         "difference": 0.1, "relative_diff": 25.0},
            "mean_relative_area_error": {"model_a": 0.6, "model_b": 0.4,
                                         "difference": 0.2, "relative_diff": 50.0},
        }
        report_lines = report._build_report(data)
        assert "No data." in report_lines

    def test_report_handles_missing_ci(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        data = _data()
        data["hypothesis"] = {"validated": False, "evidence": "",
                              "conclusion": ""}
        report_lines = report._build_report(data)
        assert "Not computed." in report_lines

    def test_generate_report_writes_files(self, tmp_path):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        path = report.generate_report(_data())
        assert (tmp_path / "comparison_report.md").exists()
        assert (tmp_path / "comparison_data.json").exists()
        content = (tmp_path / "comparison_report.md").read_text()
        assert "Conclusion" in content

    def _build_for_ci(self, tmp_path, ci_difference):
        report = ComparisonReport(str(tmp_path),
                                  class_names=["Bg", "Dent", "Scratch",
                                               "No", "Severe"])
        data = _data()
        data["hypothesis"] = {
            **_hypothesis(),
            "ci_difference": ci_difference,
            "mean_iou_instance": 0.5,
            "mean_iou_semantic": 0.6,
        }
        return report._build_report(data)

    def test_conclusion_significant_when_ci_excludes_zero(self, tmp_path):
        """CI of the difference excluding 0 must NOT be labeled as not
        significant (regression: it was hardcoded as such)."""
        lines = self._build_for_ci(tmp_path, (-0.10, -0.01))
        assert "statistically significant" in lines
        assert "not statistically significant" not in lines

    def test_conclusion_not_significant_when_ci_contains_zero(self, tmp_path):
        lines = self._build_for_ci(tmp_path, (-0.15, 0.17))
        assert "not statistically significant" in lines