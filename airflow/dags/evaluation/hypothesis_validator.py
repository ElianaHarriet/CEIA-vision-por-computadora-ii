"""Hypothesis validator for evaluation."""
import numpy as np
from scipy import stats


class HypothesisValidator:
    """Validator for project hypothesis."""

    def __init__(self, seed: int = 2026, n_bootstrap: int = 1000):
        """Initialize validator.

        Args:
            seed: RNG seed for the bootstrap, so CIs are reproducible.
            n_bootstrap: number of bootstrap resamples.
        """
        self.seed = seed
        self.n_bootstrap = n_bootstrap

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
            'ci_iou_instance': evidence['ci_iou_inst'],
            'ci_iou_semantic': evidence['ci_iou_sem'],
            'ci_difference': evidence['ci_diff'],
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
        ci_iou_inst = self._bootstrap_ci(self._iou_list(metrics_inst))
        ci_iou_sem = self._bootstrap_ci(self._iou_list(metrics_sem))
        ci_diff = self._bootstrap_ci_difference(
            self._iou_list(metrics_inst),
            self._iou_list(metrics_sem)
        )
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
        # Also weigh the bootstrap CI of the difference: if it contains 0
        # the gap is not significant regardless of which mean is higher.
        if ci_diff[0] <= 0 <= ci_diff[1]:
            summary += (
                f" Bootstrap 95% CI of the difference "
                f"[{ci_diff[0]:.4f}, {ci_diff[1]:.4f}] contains 0, "
                "so the gap is not statistically significant."
            )
        return {
            'iou_inst': iou_inst,
            'iou_sem': iou_sem,
            'p_value': p_value,
            'ci_iou_inst': ci_iou_inst,
            'ci_iou_sem': ci_iou_sem,
            'ci_diff': ci_diff,
            'summary': summary
        }

    def _iou_list(self, metrics):
        """Per-image binary IoU values."""
        return [
            m['iou_global']
            for m in metrics['per_image'].values()
        ]

    def _bootstrap_ci(self, values, alpha=0.05):
        """Percentile bootstrap confidence interval around the mean."""
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return (0.0, 0.0)
        rng = np.random.default_rng(self.seed)
        sample_means = np.array([
            rng.choice(values, size=values.size, replace=True).mean()
            for _ in range(self.n_bootstrap)
        ])
        lo = np.percentile(sample_means, 100 * alpha / 2)
        hi = np.percentile(sample_means, 100 * (1 - alpha / 2))
        return (float(lo), float(hi))

    def _bootstrap_ci_difference(self, values_a, values_b, alpha=0.05):
        """Percentile bootstrap CI for the difference of the means.

        Two independent samples are resampled separately and the mean
        difference is computed per draw. If 0 lies inside the interval the
        gap is not significant.
        """
        a = np.asarray(values_a, dtype=float)
        b = np.asarray(values_b, dtype=float)
        if a.size == 0 and b.size == 0:
            return (0.0, 0.0)
        rng = np.random.default_rng(self.seed + 1)
        sample_diffs = []
        for _ in range(self.n_bootstrap):
            ma = rng.choice(a, size=a.size, replace=True).mean() if a.size else 0.0
            mb = rng.choice(b, size=b.size, replace=True).mean() if b.size else 0.0
            sample_diffs.append(ma - mb)
        sample_diffs = np.asarray(sample_diffs)
        lo = np.percentile(sample_diffs, 100 * alpha / 2)
        hi = np.percentile(sample_diffs, 100 * (1 - alpha / 2))
        return (float(lo), float(hi))

    def _statistical_test(self, metrics_inst, metrics_sem):
        """Perform statistical significance test."""
        inst_ious = self._iou_list(metrics_inst)
        sem_ious = self._iou_list(metrics_sem)
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
