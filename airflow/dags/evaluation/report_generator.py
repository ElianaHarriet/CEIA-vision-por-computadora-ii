"""Report generator for evaluation."""
from pathlib import Path
import json


class ComparisonReport:
    """Generator for comparison report."""

    def __init__(self, output_path: str, class_names: list):
        """Initialize report generator."""
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names

    def generate_report(self, data: dict):
        """Generate comparison report."""
        print("Generating report...")
        report = self._build_report(data)
        path = self._save_markdown(report)
        self._save_json(data)
        print(f"✓ Report saved to {path}")
        return str(path)

    def _build_report(self, data: dict):
        """Build report content."""
        sections = [
            self._header(),
            self._summary(data['comparison']),
            self._detailed_metrics(data['comparison']),
            self._per_class_iou(data['comparison']),
            self._confidence_intervals(data['hypothesis']),
            self._hypothesis_validation(data['hypothesis']),
            self._conclusion(data)
        ]
        return '\n\n'.join(sections)

    def _header(self):
        """Generate report header."""
        return "# Model Comparison Report\n\n" \
               "## Instance vs Semantic Segmentation\n\n" \
               "Comparison of YOLOv8-seg (instance) " \
               "vs U-Net (semantic) models."

    def _summary(self, comparison: dict):
        """Generate summary section."""
        iou = comparison.get('mean_iou', {})
        lines = [
            "## Executive Summary",
            "",
            f"- **Instance Model IoU (binary damage)**: {iou.get('model_a', 0):.4f}",
            f"- **Semantic Model IoU (binary damage)**: {iou.get('model_b', 0):.4f}",
            f"- **Difference**: {iou.get('difference', 0):.4f}",
            f"- **Relative Improvement**: "
            f"{iou.get('relative_diff', 0):.2f}%"
        ]
        return '\n'.join(lines)

    def _detailed_metrics(self, comparison: dict):
        """Generate detailed metrics table (Opción B: binary damage mask)."""
        lines = [
            "## Detailed Metrics — Binary Damage (Opción B)",
            "",
            "All values are computed on the flattened binary damage mask "
            "(damage vs no-damage), so precision/recall/F1 are comparable "
            "with the IoU reported here.",
            "",
            "| Metric | Instance | Semantic | Difference | % Change |",
            "|--------|----------|----------|------------|----------|"
        ]
        skip = {'mean_binary_precision', 'mean_binary_recall', 'mean_binary_f1'}
        for key, vals in comparison.items():
            if key.startswith('mean_') and key not in skip:
                name = key.replace('mean_', '').replace('_', ' ').title()
                inst = vals['model_a']
                sem = vals['model_b']
                diff = vals['difference']
                rel = vals['relative_diff']
                line = f"| {name} | {inst:.4f} | {sem:.4f} | " \
                       f"{diff:+.4f} | {rel:+.2f}% |"
                lines.append(line)
        lines += [
            "| Binary Precision | %.4f | %.4f | %+.4f | %+.2f%% |" % (
                comparison['mean_binary_precision']['model_a'],
                comparison['mean_binary_precision']['model_b'],
                comparison['mean_binary_precision']['difference'],
                comparison['mean_binary_precision']['relative_diff'],
            ),
            "| Binary Recall | %.4f | %.4f | %+.4f | %+.2f%% |" % (
                comparison['mean_binary_recall']['model_a'],
                comparison['mean_binary_recall']['model_b'],
                comparison['mean_binary_recall']['difference'],
                comparison['mean_binary_recall']['relative_diff'],
            ),
            "| Binary F1 | %.4f | %.4f | %+.4f | %+.2f%% |" % (
                comparison['mean_binary_f1']['model_a'],
                comparison['mean_binary_f1']['model_b'],
                comparison['mean_binary_f1']['difference'],
                comparison['mean_binary_f1']['relative_diff'],
            ),
        ]
        return '\n'.join(lines)

    def _per_class_iou(self, comparison: dict):
        """Generate per-class IoU table (Opción A: multiclass)."""
        per_class = comparison.get('per_class_iou', {})
        if not per_class:
            return "## Per-Class IoU (Opción A: multiclass)\n\nNo data."
        lines = [
            "## Per-Class IoU (Opción A: multiclass)",
            "",
            "Micro-averaged per class over the test set: "
            "sum(TP) / sum(TP + FP + FN).",
            "",
            "| Class | Instance | Semantic | Difference |",
            "|-------|----------|----------|------------|"
        ]
        for idx, name in enumerate(self.class_names):
            vals = per_class.get(str(idx), {})
            inst = vals.get('model_a', 0.0)
            sem = vals.get('model_b', 0.0)
            diff = vals.get('difference', 0.0)
            lines.append(
                f"| {name} | {inst:.4f} | {sem:.4f} | {diff:+.4f} |"
            )
        return '\n'.join(lines)

    def _confidence_intervals(self, hypothesis: dict):
        """Generate confidence interval section."""
        ci_inst = hypothesis.get('ci_iou_instance')
        ci_sem = hypothesis.get('ci_iou_semantic')
        ci_diff = hypothesis.get('ci_difference')
        if ci_inst is None:
            return "## Confidence Intervals (Bootstrap)\n\nNot computed."
        diff_txt = (
            f"The 95% CI of the difference [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}] "
            + ("**contains 0** — the gap is not statistically significant."
               if ci_diff[0] <= 0 <= ci_diff[1]
               else "does **not** contain 0 — the gap is significant.")
        )
        lines = [
            "## Confidence Intervals (Bootstrap, 95%, seed=2026)",
            "",
            f"- **Instance IoU**: [{ci_inst[0]:.4f}, {ci_inst[1]:.4f}]",
            f"- **Semantic IoU**: [{ci_sem[0]:.4f}, {ci_sem[1]:.4f}]",
            f"- **Difference**: {diff_txt}",
        ]
        return '\n'.join(lines)

    def _hypothesis_validation(self, hypothesis: dict):
        """Generate hypothesis validation section."""
        validated = hypothesis.get('validated', False)
        status = "✅ VALIDATED" if validated else "❌ REFUTED"
        lines = [
            "## Hypothesis Validation",
            "",
            f"**Status**: {status}",
            "",
            "### Hypothesis",
            "Instance segmentation models learn more precise contours,",
            "resulting in more accurate area estimates even when flattened.",
            "",
            "### Evidence",
            hypothesis.get('evidence', 'No evidence provided'),
            "",
            "### Conclusion",
            hypothesis.get('conclusion', 'No conclusion provided')
        ]
        return '\n'.join(lines)

    def _conclusion(self, data: dict):
        """Generate conclusion section."""
        hypothesis = data.get('hypothesis', {})
        ci_diff = hypothesis.get('ci_difference')
        ci_txt = ""
        if ci_diff is not None:
            ci_txt = (
                f"\n\n- Instance IoU minus Semantic IoU, 95% CI "
                f"[{ci_diff[0]:.4f}, {ci_diff[1]:.4f}], "
                + ("contains 0 → the two are statistically indistinguishable "
                   "at the IoU level." if ci_diff[0] <= 0 <= ci_diff[1]
                   else "excludes 0 → the gap is significant.")
            )
        return (
            "## Conclusion\n\n"
            "Based on the evaluation metrics and visualizations:\n"
            "- **IoU (binary damage)**: instance and semantic are comparable "
            f"(diff {data['comparison']['mean_iou']['difference']:+.4f})."
            f"{ci_txt}\n"
            "- **Area estimation**: semantic has a substantially lower "
            f"relative area error ({data['comparison']['mean_relative_area_error']['model_b']:.4f} "
            f"vs {data['comparison']['mean_relative_area_error']['model_a']:.4f}), "
            "supporting the semantic approach for area-based estimates.\n"
            "- **Per-class detail**: see the per-class IoU table for "
            "class-level differences."
        )

    def _save_markdown(self, content: str):
        """Save report as markdown."""
        path = self.output_path / 'comparison_report.md'
        path.write_text(content, encoding='utf-8')
        return path

    def _save_json(self, data: dict):
        """Save data as JSON."""
        path = self.output_path / 'comparison_data.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
