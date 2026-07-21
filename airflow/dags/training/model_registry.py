"""Model registry operations."""
import mlflow


class ModelRegistry:
    """Manager for MLflow Model Registry."""

    def __init__(self, tracking_uri: str):
        """Initialize registry."""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    def register_model(self, run_id: str, model_name: str, desc: str):
        """Register model in registry."""
        print("Registering model in Model Registry...")
        print(f"Run ID: {run_id}")
        print(f"Model name: {model_name}")
        model_uri = self._build_uri(run_id)
        version = self._register(model_uri, model_name)
        self._update_description(model_name, version, desc)
        return self._get_result(model_name, version)

    def _build_uri(self, run_id: str):
        """Build model URI."""
        return f"runs:/{run_id}/model"

    def _register(self, uri: str, name: str):
        """Register model and get version."""
        try:
            model_version = mlflow.register_model(uri, name)
            version = model_version.version
            print(f"✓ Registered: {name} (Version {version})")
            return version
        except Exception as exc:
            print(f"Error registering model: {exc}")
            raise

    def _update_description(self, name: str, version: int, desc: str):
        """Update model version description."""
        client = mlflow.MlflowClient()
        client.update_model_version(
            name=name,
            version=version,
            description=desc
        )

    def _get_result(self, name: str, version: int):
        """Get registration result."""
        return {
            "model_name": name,
            "version": version,
            "stage": "None"
        }


class YOLOModelRegistry(ModelRegistry):
    """Registry for YOLO models."""

    def register_yolo(self, run_id: str, model_name: str, epochs: int):
        """Register YOLO model."""
        desc = f"YOLOv8-seg instance segmentation. {epochs} epochs."
        model_uri = f"runs:/{run_id}/model/best.pt"
        version = self._register(model_uri, model_name)
        self._update_description(model_name, version, desc)
        return self._get_result(model_name, version)
