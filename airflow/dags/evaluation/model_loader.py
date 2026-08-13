"""Model loader for evaluation."""
import mlflow
import torch

# Disable NNPACK IMMEDIATELY after torch import to avoid errors on older CPUs/VMs
torch.backends.nnpack.enabled = False

from pathlib import Path
from ultralytics import YOLO


class ModelLoader:
    """Loader for MLflow models."""

    def __init__(self, tracking_uri: str):
        """Initialize loader."""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    def check_model_availability(self, model_name: str, stage: str):
        """Check if model is available."""
        print(f"Checking model: {model_name} (stage: {stage})")
        client = mlflow.MlflowClient()
        try:
            versions = client.get_latest_versions(model_name, [stage])
            if not versions:
                msg = f"No model in stage {stage}"
                raise ValueError(msg)
            version = versions[0]
            print(f"✓ Found {model_name} v{version.version}")
            return version
        except Exception as exc:
            print(f"✗ Model not found: {exc}")
            raise


class YOLOModelLoader(ModelLoader):
    """Loader for YOLO models."""

    def load_yolo_model(self, model_name: str, stage: str):
        """Load YOLO model from MLflow."""
        version = self.check_model_availability(model_name, stage)
        run_id = version.run_id
        model_uri = f"runs:/{run_id}/model/best.pt"
        print(f"Loading YOLO from: {model_uri}")
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        model = YOLO(local_path)
        return model


class UNetModelLoader(ModelLoader):
    """Loader for U-Net models."""

    def load_unet_model(self, model_name: str, stage: str):
        """Load U-Net model from MLflow."""
        version = self.check_model_availability(model_name, stage)
        model_uri = f"models:/{model_name}/{stage}"
        print(f"Loading U-Net from: {model_uri}")
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model = model.to(device)
        model.eval()
        return model, device
