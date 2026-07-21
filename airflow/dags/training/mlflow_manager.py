"""MLflow management for training."""
import mlflow
from datetime import datetime


class MLflowManager:
    """Manager for MLflow operations."""

    def __init__(self, uri: str, experiment: str):
        """Initialize MLflow manager."""
        self.uri = uri
        self.experiment = experiment
        self._configure()

    def _configure(self):
        """Configure MLflow connection."""
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment)

    def start_run(self, name_prefix: str):
        """Start MLflow run."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f"{name_prefix}-{timestamp}"
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        """Log parameters to MLflow."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int = None):
        """Log single metric."""
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(key, value, step)

    def log_artifact(self, path: str, artifact_path: str = None):
        """Log artifact to MLflow."""
        mlflow.log_artifact(path, artifact_path)

    def log_model(self, model, artifact_path: str):
        """Log model to MLflow."""
        mlflow.pytorch.log_model(model, artifact_path)
