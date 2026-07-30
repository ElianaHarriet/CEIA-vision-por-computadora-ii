"""Visualization generator for evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ComparisonVisualizer:
    """Visualizer for model comparison."""

    def __init__(self, output_path: str, class_names: list):
        """Initialize visualizer."""
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names

    def generate_all_visualizations(self, data: dict):
        """Generate all visualizations."""
        print("Generating visualizations...")
        self._generate_metrics_comparison(data['comparison'])
        self._generate_iou_bar_chart(data['comparison'])
        self._generate_sample_predictions(data['samples'])
        print(f"✓ Visualizations saved to {self.output_path}")
        return str(self.output_path)

    def _generate_metrics_comparison(self, comparison: dict):
        """Generate metrics comparison chart."""
        metrics = []
        values_a = []
        values_b = []
        for key, vals in comparison.items():
            if key.startswith('mean_'):
                name = key.replace('mean_', '').replace('_', ' ').title()
                metrics.append(name)
                values_a.append(vals['model_a'])
                values_b.append(vals['model_b'])
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, values_a, width, label='Instance')
        ax.bar(x + width/2, values_b, width, label='Semantic')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = self.output_path / 'metrics_comparison.png'
        plt.savefig(path, dpi=150)
        plt.close()

    def _generate_iou_bar_chart(self, comparison: dict):
        """Generate IoU bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        metric = comparison.get('mean_iou', {})
        values = [metric.get('model_a', 0), metric.get('model_b', 0)]
        labels = ['Instance (YOLOv8)', 'Semantic (U-Net)']
        colors = ['#2ecc71', '#3498db']
        bars = ax.bar(labels, values, color=colors)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
        ax.set_ylabel('Mean IoU')
        ax.set_title('Mean IoU Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        path = self.output_path / 'iou_comparison.png'
        plt.savefig(path, dpi=150)
        plt.close()

    def _generate_sample_predictions(self, samples: dict):
        """Generate sample prediction visualizations."""
        if not samples:
            return
        n_samples = min(5, len(samples))
        sample_items = list(samples.items())[:n_samples]
        fig, axes = plt.subplots(4, n_samples, figsize=(4*n_samples, 16))
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        for i, (name, data) in enumerate(sample_items):
            self._plot_sample(axes[:, i], name, data)
        plt.tight_layout()
        path = self.output_path / 'sample_predictions.png'
        plt.savefig(path, dpi=150)
        plt.close()

    def _plot_sample(self, axes, name, data):
        """Plot single sample."""
        axes[0].imshow(data['image'])
        axes[0].set_title(f'Original\n{name}')
        axes[0].axis('off')
        axes[1].imshow(data['gt'], cmap='tab10', vmin=0, vmax=3)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(data['pred_instance'], cmap='tab10', vmin=0, vmax=3)
        iou_inst = data.get('iou_instance', 0)
        axes[2].set_title(f'Instance (IoU: {iou_inst:.3f})')
        axes[2].axis('off')
        axes[3].imshow(data['pred_semantic'], cmap='tab10', vmin=0, vmax=3)
        iou_sem = data.get('iou_semantic', 0)
        axes[3].set_title(f'Semantic (IoU: {iou_sem:.3f})')
        axes[3].axis('off')
