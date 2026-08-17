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
            self._summary(data['comparison'], data.get('hypothesis')),
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

    def _summary(self, comparison: dict, hypothesis: dict):
        """Generate summary section.

        The headline means are the PAIRED ones over the common subset
        (the population the statistical test actually compares), so the
        summary direction agrees with the CI section. The full-set means
        are shown as context.
        """
        hyp = hypothesis or {}
        inst = hyp.get('mean_iou_instance')
        sem = hyp.get('mean_iou_semantic')
        n_common = hyp.get('n_images_common')
        if inst is None or sem is None:
            # Fallback to the (full-set) comparison means.
            iou = comparison.get('mean_iou', {})
            inst = iou.get('model_a', 0)
            sem = iou.get('model_b', 0)
        diff = inst - sem
        rel = (diff / sem * 100) if sem else 0
        lines = [
            "## Executive Summary",
            "",
            f"- **Instance Model IoU (binary damage, paired)**: {inst:.4f}",
            f"- **Semantic Model IoU (binary damage, paired)**: {sem:.4f}",
            f"- **Difference (paired)**: {diff:+.4f}",
            f"- **Relative Improvement (paired)**: {rel:+.2f}%",
        ]
        full_inst = hyp.get('full_mean_iou_instance')
        full_sem = hyp.get('full_mean_iou_semantic')
        if full_inst is not None and full_sem is not None:
            lines += [
                "",
                f"Contexto — medias sobre el conjunto completo "
                f"(instance {hyp.get('n_images_instance')} imágenes, "
                f"semantic {hyp.get('n_images_semantic')}): "
                f"Instance {full_inst:.4f} vs Semantic {full_sem:.4f}.",
            ]
        if n_common:
            lines += [
                f"Las medias pareadas se computan sobre las {n_common} "
                "imágenes evaluadas por ambos modelos."
            ]
        return '\n'.join(lines)

    def _detailed_metrics(self, comparison: dict):
        """Generate detailed metrics tables (Opción B + multiclass macro)."""
        lines = [
            "## Detailed Metrics",
            "",
            "**Tabla 1 — Binary Damage (Opción B)**: todas las filas se "
            "computan sobre la máscara binaria daño vs no-daño (clase "
            "No Damage tratada como fondo), por lo que son comparables "
            "con el IoU reportado.",
            "",
            "| Métrica | Instance | Semantic | Difference | % Change |",
            "|--------|----------|----------|------------|----------|"
        ]
        # Binary-damage metrics (Opción B) are listed explicitly.
        binary_keys = [
            ('mean_iou', 'IoU (binary damage)'),
            ('mean_binary_precision', 'Binary Precision'),
            ('mean_binary_recall', 'Binary Recall'),
            ('mean_binary_f1', 'Binary F1'),
        ]
        for key, label in binary_keys:
            vals = comparison.get(key)
            if vals is None:
                continue
            lines.append(
                f"| {label} | {vals['model_a']:.4f} | {vals['model_b']:.4f} "
                f"| {vals['difference']:+.4f} | {vals['relative_diff']:+.2f}% |"
            )
        lines += [
            "",
            "**Tabla 2 — Multiclass (macro sobre las 5 clases, Opción A)**: "
            "precision/recall/F1 son medias macro de sklearn sobre todas "
            "las clases; NO son binarias. Para métricas binarias ver Tabla 1.",
            "",
            "| Métrica | Instance | Semantic | Difference | % Change |",
            "|--------|----------|----------|------------|----------|"
        ]
        multiclass_keys = [
            ('mean_pixel_accuracy', 'Pixel Accuracy (global)'),
            ('mean_precision', 'Precision (macro, 5 clases)'),
            ('mean_recall', 'Recall (macro, 5 clases)'),
            ('mean_f1_score', 'F1-Score (macro, 5 clases)'),
            ('mean_area_error', 'Area Error (px, absoluto)'),
            ('mean_relative_area_error', 'Relative Area Error'),
        ]
        for key, label in multiclass_keys:
            vals = comparison.get(key)
            if vals is None:
                continue
            lines.append(
                f"| {label} | {vals['model_a']:.4f} | {vals['model_b']:.4f} "
                f"| {vals['difference']:+.4f} | {vals['relative_diff']:+.2f}% |"
            )
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
            f"- **Common subset**: {hypothesis.get('n_images_common', '?')} "
            f"images evaluated by both models (instance "
            f"{hypothesis.get('n_images_instance', '?')}, semantic "
            f"{hypothesis.get('n_images_semantic', '?')}).",
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
        inst = hypothesis.get('mean_iou_instance')
        sem = hypothesis.get('mean_iou_semantic')
        diff = (inst - sem) if inst is not None and sem is not None else None
        ci_txt = ""
        if ci_diff is not None:
            ci_txt = (
                f"\n\n- Instance IoU minus Semantic IoU, 95% CI "
                f"[{ci_diff[0]:.4f}, {ci_diff[1]:.4f}], "
                + ("contains 0 → the two are statistically indistinguishable "
                   "at the IoU level." if ci_diff[0] <= 0 <= ci_diff[1]
                   else "excludes 0 → the gap is significant.")
            )
        if diff is not None:
            significant = ci_diff is not None and not (ci_diff[0] <= 0 <= ci_diff[1])
            sign_txt = (
                "statistically significant" if significant
                else "not statistically significant"
            )
            iou_line = (
                f"- **IoU (binary damage, paired)**: difference {diff:+.4f} "
                f"over the common subset ({sign_txt})."
            )
        else:
            iou_line = (
                f"- **IoU (binary damage)**: difference "
                f"{data['comparison']['mean_iou']['difference']:+.4f}."
            )
        return (
            "## Conclusion\n\n"
            "Based on the evaluation metrics and visualizations:\n"
            f"{iou_line}"
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
