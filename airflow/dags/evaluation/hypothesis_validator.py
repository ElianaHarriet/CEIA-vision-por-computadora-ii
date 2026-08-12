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
        """Validate hypothesis.

        If there is no overlap between the models' evaluated images, the
        comparison is impossible: the hypothesis is treated as "not
        validated" with an explicit message instead of crashing.
        """
        print("Validating hypothesis...")
        common = (set(metrics_instance.get('per_image', {}))
                  & set(metrics_semantic.get('per_image', {})))
        if len(common) < 2:
            result = {
                'validated': False,
                'evidence': (
                    f"Insufficient common data to compare: "
                    f"instance {len(metrics_instance.get('per_image', {}))} "
                    f"images, semantic "
                    f"{len(metrics_semantic.get('per_image', {}))} images, "
                    f"{len(common)} in common."
                ),
                'p_value': 1.0,
                'mean_iou_instance': 0.0,
                'mean_iou_semantic': 0.0,
                'full_mean_iou_instance': 0.0,
                'full_mean_iou_semantic': 0.0,
                'ci_iou_instance': (0.0, 0.0),
                'ci_iou_semantic': (0.0, 0.0),
                'ci_difference': (0.0, 0.0),
                'n_images_common': len(common),
                'n_images_instance': len(metrics_instance.get('per_image', {})),
                'n_images_semantic': len(metrics_semantic.get('per_image', {})),
                'conclusion': (
                    "Not enough common images to validate the hypothesis."
                ),
            }
            print("✗ Hypothesis refuted (insufficient common data)")
            return result
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
            'full_mean_iou_instance': evidence['full_iou_inst'],
            'full_mean_iou_semantic': evidence['full_iou_sem'],
            'ci_iou_instance': evidence['ci_iou_inst'],
            'ci_iou_semantic': evidence['ci_iou_sem'],
            'ci_difference': evidence['ci_diff'],
            'n_images_common': evidence['n_common'],
            'n_images_instance': evidence['n_inst'],
            'n_images_semantic': evidence['n_sem'],
            'conclusion': conclusion
        }
        print(f"✓ Hypothesis {'validated' if validated else 'refuted'}")
        return result

    def _gather_evidence(self, metrics_inst, metrics_sem):
        """Gather evidence for hypothesis.

        Both models are compared on the intersection of their evaluated
        images (the "common subset"). Instance segmentation skips images
        where it produced no detection, so its per-image set can be
        smaller than the semantic one; comparing on the common subset
        removes that selection bias and makes the paired test valid.

        All statistics (means, p-value, CIs) are computed on the common
        subset so the narrative is internally consistent. The full-set
        means are kept as context for the reader.
        """
        common_inst, common_sem = self._paired_ious(
            metrics_inst, metrics_sem
        )
        n_common = len(common_inst)
        iou_inst = float(np.mean(common_inst)) if common_inst else 0.0
        iou_sem = float(np.mean(common_sem)) if common_sem else 0.0
        full_inst = metrics_inst['aggregated']['mean_iou']
        full_sem = metrics_sem['aggregated']['mean_iou']
        p_value = self._statistical_test(common_inst, common_sem)
        ci_iou_inst = self._bootstrap_ci(common_inst)
        ci_iou_sem = self._bootstrap_ci(common_sem)
        ci_diff = self._bootstrap_ci_difference(
            common_inst, common_sem
        )
        if iou_inst > iou_sem:
            summary = (
                f"Instance IoU ({iou_inst:.4f}) vs "
                f"Semantic IoU ({iou_sem:.4f}) on the common subset. "
                "Instance model shows better performance."
            )
        else:
            summary = (
                f"Instance IoU ({iou_inst:.4f}) vs "
                f"Semantic IoU ({iou_sem:.4f}) on the common subset. "
                "Semantic model shows better performance."
            )
        summary += (
            f" Comparison on {n_common} common images "
            f"(instance evaluated {len(metrics_inst['per_image'])}, "
            f"semantic {len(metrics_sem['per_image'])})."
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
            'full_iou_inst': full_inst,
            'full_iou_sem': full_sem,
            'p_value': p_value,
            'ci_iou_inst': ci_iou_inst,
            'ci_iou_sem': ci_iou_sem,
            'ci_diff': ci_diff,
            'n_common': n_common,
            'n_inst': len(metrics_inst['per_image']),
            'n_sem': len(metrics_sem['per_image']),
            'summary': summary
        }

    @staticmethod
    def _paired_ious(metrics_inst, metrics_sem):
        """Per-image binary IoU values on the common subset.

        Returns two aligned lists (instance, semantic) covering only the
        images evaluated by BOTH models, so a paired statistical test is
        valid. Values are sorted by image name so the alignment is stable.
        """
        inst_ious = metrics_inst['per_image']
        sem_ious = metrics_sem['per_image']
        common = sorted(set(inst_ious) & set(sem_ious))
        inst = [inst_ious[name]['iou_global'] for name in common]
        sem = [sem_ious[name]['iou_global'] for name in common]
        return inst, sem

    def _bootstrap_ci(self, values, alpha=0.05):
        """Percentile bootstrap confidence interval around the mean."""
        values = np.asarray(values, dtype=float)
        if values.size == 0 or self.n_bootstrap <= 0:
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

        Paired bootstrap: both samples are resampled on the SAME indices
        (resampling image pairs together), matching the paired t-test. If
        0 lies inside the interval the gap is not significant.
        """
        a = np.asarray(values_a, dtype=float)
        b = np.asarray(values_b, dtype=float)
        if a.size == 0 or b.size == 0 or self.n_bootstrap <= 0:
            return (0.0, 0.0)
        if a.size != b.size:
            raise ValueError(
                "Paired bootstrap requires equal-sized samples "
                f"({a.size} vs {b.size}); use the common subset."
            )
        rng = np.random.default_rng(self.seed + 1)
        n = a.size
        sample_diffs = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            sample_diffs.append((a[idx] - b[idx]).mean())
        sample_diffs = np.asarray(sample_diffs)
        lo = np.percentile(sample_diffs, 100 * alpha / 2)
        hi = np.percentile(sample_diffs, 100 * (1 - alpha / 2))
        return (float(lo), float(hi))

    def _statistical_test(self, inst_ious, sem_ious):
        """Perform paired significance test on the common subset.

        Returns 1.0 (not significant) when the test is degenerate — too
        few samples or a non-finite p-value (e.g. zero-variance paired
        differences, where scipy emits a nan or a numerically meaningless
        p=0.0). This prevents a degenerate case from spuriously
        validating the hypothesis.
        """
        if len(inst_ious) < 2 or len(sem_ious) < 2:
            return 1.0
        _, p_value = stats.ttest_rel(inst_ious, sem_ious)
        p_value = float(p_value)
        if np.isnan(p_value) or not np.isfinite(p_value):
            return 1.0
        # Zero-variance paired differences (e.g. constant offset between
        # the models) make scipy emit a meaningless p=0.0; treat as
        # not significant.
        diffs = np.asarray(inst_ious, dtype=float) - np.asarray(sem_ious, dtype=float)
        if np.ptp(diffs) < 1e-12:
            return 1.0
        return p_value

    def _determine_validation(self, evidence: dict):
        """Determine if hypothesis is validated.

        The hypothesis claims instance segmentation outperforms
        semantic segmentation. It is validated only when the instance
        mean IoU is actually higher AND the difference is statistically
        significant (p < 0.05) AND the bootstrap CI of the difference
        does not contain 0.
        """
        if evidence['iou_inst'] <= evidence['iou_sem']:
            return False
        if evidence['p_value'] >= 0.05:
            return False
        ci_lo, ci_hi = evidence['ci_diff']
        return not (ci_lo <= 0 <= ci_hi)

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
