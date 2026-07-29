"""YOLO trainer for instance segmentation."""
import torch
from pathlib import Path
from ultralytics import YOLO
from training.mlflow_manager import MLflowManager


class YOLOTrainer:
    """Trainer for YOLOv8-seg models."""

    def __init__(self, config, mlflow_mgr: MLflowManager):
        """Initialize trainer."""
        self.config = config
        self.mlflow = mlflow_mgr
        self.model = None
        self.results = None

    def train(self, data_yaml: str):
        """Train YOLO model."""
        print(f"Training {self.config.MODEL}...")
        self._load_model()
        self._log_hyperparams()
        self._run_training(data_yaml)
        self._extract_metrics()
        return self._get_results()

    def _load_model(self):
        """Load pretrained model."""
        self.model = YOLO(self.config.MODEL)

    def _log_hyperparams(self):
        """Log hyperparameters to MLflow."""
        params = self._get_hyperparams()
        self.mlflow.log_params(params)

    def _get_hyperparams(self):
        """Get hyperparameters dict."""
        return {
            "model_architecture": self.config.MODEL,
            "epochs": self.config.EPOCHS,
            "batch_size": self.config.BATCH_SIZE,
            "img_size": self.config.IMG_SIZE,
            "device": self._get_device(),
            "dataset": "car-damages-instance",
            "num_classes": 4,
        }

    def _get_device(self):
        """Get device string."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _run_training(self, data_yaml: str):
        """Run training loop."""
        device = self._get_training_device()
        self.results = self.model.train(
            data=data_yaml,
            epochs=self.config.EPOCHS,
            imgsz=self.config.IMG_SIZE,
            batch=self.config.BATCH_SIZE,
            device=device,
            name='car_damage_instance',
            project='/opt/airflow/runs/segment',
            verbose=True,
            patience=self.config.PATIENCE,
            save=True,
            plots=True
        )

    def _get_training_device(self):
        """Get device for training."""
        if torch.cuda.is_available():
            return self.config.DEVICE
        return 'cpu'

    def _extract_metrics(self):
        """Extract and log metrics."""
        metrics = self._get_metrics_dict()
        if metrics:
            self._log_metrics(metrics)

    def _get_metrics_dict(self):
        """Get metrics dictionary."""
        if hasattr(self.results, 'results_dict'):
            return self.results.results_dict
        return {}

    def _log_metrics(self, metrics: dict):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                safe_key = self._sanitize_metric_name(key)
                self.mlflow.log_metric(safe_key, value)
                print(f"  {safe_key}: {value}")

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        """Strip characters MLflow doesn't allow in metric names, e.g. parentheses."""
        return name.replace("(", "").replace(")", "")

    def _get_results(self):
        """Get training results."""
        return {
            "model_path": self._get_model_path(),
            "metrics": self._get_metrics_dict()
        }

    def _get_model_path(self):
        """Get path to best model."""
        path = "/opt/airflow/runs/segment"
        return f"{path}/car_damage_instance/weights/best.pt"

    def log_artifacts(self):
        """Log model and plots to MLflow."""
        self._log_model()
        self._log_plots()

    def _log_model(self):
        """Log model weights."""
        model_path = self._get_model_path()
        self.mlflow.log_artifact(model_path, "model")

    def _log_plots(self):
        """Log training plots."""
        plots_dir = Path("/opt/airflow/runs/segment/car_damage_instance")
        if plots_dir.exists():
            self._log_plot_files(plots_dir)

    def _log_plot_files(self, plots_dir: Path):
        """Log all plot files in directory."""
        for plot_file in plots_dir.glob("*.png"):
            self.mlflow.log_artifact(str(plot_file), "plots")


class YOLOValidator:
    """Validator for YOLO models."""

    def __init__(self, model_path: str, data_yaml: str):
        """Initialize validator."""
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = None

    def validate(self):
        """Validate model."""
        print(f"Validating: {self.model_path}")
        self._ensure_exists()
        self._load_model()
        self._run_validation()
        return self._extract_results()

    def _ensure_exists(self):
        """Ensure model file exists."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Not found: {self.model_path}")

    def _load_model(self):
        """Load model from checkpoint."""
        self.model = YOLO(self.model_path)

    def _run_validation(self):
        """Run validation on valid set."""
        print("Running validation on valid set...")
        self.metrics = self.model.val(data=self.data_yaml)

    def _extract_results(self):
        """Extract validation results."""
        results = {
            "val_mAP50": self._get_box_map50(),
            "val_mAP50-95": self._get_box_map(),
            "val_mask_mAP50": self._get_seg_map50(),
            "val_mask_mAP50-95": self._get_seg_map(),
        }
        self._print_results(results)
        return results

    def _get_box_map50(self):
        """Get box mAP50."""
        if hasattr(self.metrics, 'box'):
            return float(self.metrics.box.map50)
        return None

    def _get_box_map(self):
        """Get box mAP50-95."""
        if hasattr(self.metrics, 'box'):
            return float(self.metrics.box.map)
        return None

    def _get_seg_map50(self):
        """Get segmentation mAP50."""
        if hasattr(self.metrics, 'seg'):
            return float(self.metrics.seg.map50)
        return None

    def _get_seg_map(self):
        """Get segmentation mAP50-95."""
        if hasattr(self.metrics, 'seg'):
            return float(self.metrics.seg.map)
        return None

    def _print_results(self, results: dict):
        """Print validation results."""
        print("Validation metrics:")
        for key, value in results.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
