"""Configuration for training DAGs."""
import os


class TrainingConfig:
    """Configuration for training tasks."""

    @staticmethod
    def get_data_path():
        """Get base data path."""
        return "/opt/airflow/car_damage_detection/data"

    @staticmethod
    def get_instance_path():
        """Get instance data path."""
        base = TrainingConfig.get_data_path()
        return f"{base}/car-damages-ready/instance"

    @staticmethod
    def get_semantic_path():
        """Get semantic data path."""
        base = TrainingConfig.get_data_path()
        return f"{base}/car-damages-ready/semantic"

    @staticmethod
    def get_mlflow_uri():
        """Get MLflow tracking URI."""
        return os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    @staticmethod
    def get_experiment_name():
        """Get MLflow experiment name."""
        default = "car-damage-segmentation"
        return os.getenv("MLFLOW_EXPERIMENT_NAME", default)

    @staticmethod
    def get_instance_model_name():
        """Get instance model name."""
        default = "car-damage-instance-segmentation"
        return os.getenv("MODEL_NAME_INSTANCE", default)

    @staticmethod
    def get_semantic_model_name():
        """Get semantic model name."""
        default = "car-damage-semantic-segmentation"
        return os.getenv("MODEL_NAME_SEMANTIC", default)


class YOLOConfig:
    """Configuration for YOLO training."""

    MODEL = "yolov8s-seg.pt"
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 16
    DEVICE = 0
    PATIENCE = 10


class UNetConfig:
    """Configuration for U-Net training."""

    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    # Mask pixel values are 0 (background, unlabeled) plus one index per
    # entry in CLASS_NAMES (data_preparation_semantic_dag.py) — 4 damage
    # classes + background = 5 distinct values, not 4.
    NUM_CLASSES = 5
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    IMG_SIZE = (640, 640)
    PATIENCE = 10
