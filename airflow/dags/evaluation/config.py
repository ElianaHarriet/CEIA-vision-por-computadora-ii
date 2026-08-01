"""Configuration for evaluation."""
import os


class EvaluationConfig:
    """Configuration for evaluation tasks."""

    @staticmethod
    def get_instance_path():
        """Get instance data path."""
        base = "/opt/airflow/car_damage_detection/car-damages"
        return f"{base}/car-damages-ready/instance"

    @staticmethod
    def get_semantic_path():
        """Get semantic data path."""
        base = "/opt/airflow/car_damage_detection/car-damages"
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

    @staticmethod
    def get_model_stage():
        """Get model stage to load."""
        return os.getenv("MODEL_STAGE", "Production")

    @staticmethod
    def get_results_path():
        """Get path for evaluation results."""
        return "/opt/airflow/evaluation_results"

    NUM_CLASSES = 5
    CLASS_NAMES = [
        "Background",
        "Minor Damage (Dent)",
        "Minor Damage (Scratch)",
        "No Damage",
        "Severe Damage"
    ]
