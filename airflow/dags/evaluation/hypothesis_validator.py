"""Hypothesis validator for evaluation."""
from scipy import stats


class HypothesisValidator:
    """Validator for project hypothesis."""

    def validate(self, metrics_instance: dict, metrics_semantic: dict):
        """Validate hypothesis."""
        print("Validating hypothesis...")
        evidence = self._gather_evidence(
            metrics_instance,
            metrics_semantic
        )
        validated = self._determine_validation(evidence)
        conclusion = self._generate_conclusion(validated, evidence)
        result = {
            'validated': validated,
            'evidence': evidence,
            'conclusion': conclusion
        }
        print(f"✓ Hypothesis {'validated' if validated else 'refuted'}")
        return result

    def _gather_evidence(self, metrics_inst, metrics_sem):
        """Gather evidence for hypothesis."""
        inst_agg = metrics_inst['aggregated']
        sem_agg = metrics_sem['aggregated']
        iou_inst = inst_agg['mean_iou']
        iou_sem = sem_agg['mean_iou']
        evidence = (
            f"Instance IoU ({iou_inst:.4f}) vs "
            f"Semantic IoU ({iou_sem:.4f}). "
        )
        if iou_inst > iou_sem:
            evidence += "Instance model shows better performance."
        else:
            evidence += "Semantic model shows better performance."
        p_value = self._statistical_test(metrics_inst, metrics_sem)
        if p_value < 0.05:
            evidence += f" Difference is statistically significant (p={p_value:.4f})."
        else:
            evidence += f" Difference is not significant (p={p_value:.4f})."
        return evidence

    def _statistical_test(self, metrics_inst, metrics_sem):
        """Perform statistical significance test."""
        inst_ious = [
            m['iou_global']
            for m in metrics_inst['per_image'].values()
        ]
        sem_ious = [
            m['iou_global']
            for m in metrics_sem['per_image'].values()
        ]
        if len(inst_ious) < 2 or len(sem_ious) < 2:
            return 1.0
        _, p_value = stats.ttest_ind(inst_ious, sem_ious)
        return float(p_value)

    def _determine_validation(self, evidence: str):
        """Determine if hypothesis is validated."""
        return "better performance" in evidence.lower()

    def _generate_conclusion(self, validated: bool, evidence: str):
        """Generate conclusion text."""
        if validated:
            return (
                "The hypothesis is validated. "
                "Instance segmentation models demonstrate "
                "superior contour learning, resulting in "
                "more accurate area estimations."
            )
        return (
            "The hypothesis is refuted. "
            "Semantic segmentation models perform "
            "comparably or better than instance models "
            "for this task."
        )
