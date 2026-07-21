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
            f"- **Instance Model IoU**: {iou.get('model_a', 0):.4f}",
            f"- **Semantic Model IoU**: {iou.get('model_b', 0):.4f}",
            f"- **Difference**: {iou.get('difference', 0):.4f}",
            f"- **Relative Improvement**: "
            f"{iou.get('relative_diff', 0):.2f}%"
        ]
        return '\n'.join(lines)

    def _detailed_metrics(self, comparison: dict):
        """Generate detailed metrics table."""
        lines = [
            "## Detailed Metrics",
            "",
            "| Metric | Instance | Semantic | Difference | % Change |",
            "|--------|----------|----------|------------|----------|"
        ]
        for key, vals in comparison.items():
            if key.startswith('mean_'):
                name = key.replace('mean_', '').replace('_', ' ').title()
                inst = vals['model_a']
                sem = vals['model_b']
                diff = vals['difference']
                rel = vals['relative_diff']
                line = f"| {name} | {inst:.4f} | {sem:.4f} | " \
                       f"{diff:+.4f} | {rel:+.2f}% |"
                lines.append(line)
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
        return "## Conclusion\n\n" \
               "Based on the evaluation metrics and visualizations, " \
               "we can conclude the comparative performance of both models."

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
