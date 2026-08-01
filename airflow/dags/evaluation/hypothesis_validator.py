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
            'evidence': evidence['summary'],
            'p_value': evidence['p_value'],
            'mean_iou_instance': evidence['iou_inst'],
            'mean_iou_semantic': evidence['iou_sem'],
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
        p_value = self._statistical_test(metrics_inst, metrics_sem)
        if iou_inst > iou_sem:
            summary = (
                f"Instance IoU ({iou_inst:.4f}) vs "
                f"Semantic IoU ({iou_sem:.4f}). "
                "Instance model shows better performance."
            )
        else:
            summary = (
                f"Instance IoU ({iou_inst:.4f}) vs "
                f"Semantic IoU ({iou_sem:.4f}). "
                "Semantic model shows better performance."
            )
        if p_value < 0.05:
            summary += f" Difference is statistically significant (p={p_value:.4f})."
        else:
            summary += f" Difference is not significant (p={p_value:.4f})."
        return {
            'iou_inst': iou_inst,
            'iou_sem': iou_sem,
            'p_value': p_value,
            'summary': summary
        }

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

    def _determine_validation(self, evidence: dict):
        """Determine if hypothesis is validated.

        The hypothesis claims instance segmentation outperforms
        semantic segmentation. It is validated only when the instance
        mean IoU is actually higher AND the difference is statistically
        significant (p < 0.05).
        """
        if evidence['iou_inst'] <= evidence['iou_sem']:
            return False
        return evidence['p_value'] < 0.05

    def _generate_conclusion(self, validated: bool, evidence: dict):
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
