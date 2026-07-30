"""Environment setup for training."""
import torch
import mlflow


class EnvironmentSetup:
    """Setup training environment."""

    def __init__(self, mlflow_uri: str, experiment: str):
        """Initialize environment setup."""
        self.mlflow_uri = mlflow_uri
        self.experiment = experiment

    def setup(self):
        """Setup complete environment."""
        print("Setting up training environment...")
        cuda_info = self._check_cuda()
        self._verify_libraries()
        self._configure_mlflow()
        return {"cuda_available": cuda_info['available']}

    def _check_cuda(self):
        """Check CUDA availability."""
        available = torch.cuda.is_available()
        print(f"CUDA available: {available}")
        if available:
            self._print_gpu_info()
        else:
            self._warn_cpu()
        return {"available": available}

    def _print_gpu_info(self):
        """Print GPU information."""
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"GPU: {gpu_name}")
        print(f"CUDA Version: {cuda_version}")

    def _warn_cpu(self):
        """Warn about CPU training."""
        print("⚠️  Training on CPU (will be slower)")

    def _verify_libraries(self):
        """Verify required libraries."""
        pass

    def _configure_mlflow(self):
        """Configure MLflow."""
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment)
        print(f"✓ MLflow: {self.mlflow_uri}")
        print(f"✓ Experiment: {self.experiment}")


class YOLOEnvironment(EnvironmentSetup):
    """Environment setup for YOLO training."""

    def _verify_libraries(self):
        """Verify ultralytics is installed."""
        try:
            from ultralytics import YOLO
            print("✓ ultralytics installed")
        except ImportError as exc:
            msg = "ultralytics not installed"
            raise ImportError(msg) from exc


class UNetEnvironment(EnvironmentSetup):
    """Environment setup for U-Net training."""

    def _verify_libraries(self):
        """Verify segmentation_models_pytorch installed."""
        try:
            import segmentation_models_pytorch as smp
            version = smp.__version__
            print(f"✓ segmentation_models_pytorch: {version}")
        except ImportError as exc:
            msg = "segmentation_models_pytorch not installed"
            raise ImportError(msg) from exc
