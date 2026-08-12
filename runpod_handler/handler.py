"""RunPod Serverless handler: runs YOLO/U-Net training on a rented GPU.

Reuses the trainer/config/mlflow code from airflow/dags/training as-is
(COPY'd into /app/training at build time), so training logic stays in one
place instead of being duplicated between Airflow and this handler.

Version: 1.0.2 - Add startup logging and error handling
"""
import os
import sys
import re
import logging
from pathlib import Path

# Configure logging to stdout immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("RunPod handler starting...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info("=" * 60)

sys.path.insert(0, "/app")

try:
    import runpod  # noqa: E402
    logger.info(f"✓ runpod imported successfully (version: {runpod.__version__})")
except Exception as e:
    logger.error(f"✗ Failed to import runpod: {e}")
    raise

try:
    from dataset_fetcher import download_dataset  # noqa: E402
    logger.info("✓ dataset_fetcher imported")
except Exception as e:
    logger.error(f"✗ Failed to import dataset_fetcher: {e}")
    raise

try:
    from training.mlflow_manager import MLflowManager  # noqa: E402
    from training.config import YOLOConfig, UNetConfig  # noqa: E402
    from training.yolo_trainer import YOLOTrainer  # noqa: E402
    from training.unet_trainer import UNetTrainer  # noqa: E402
    from training.dataloader_factory import DataLoaderFactory  # noqa: E402
    logger.info("✓ All training modules imported successfully")
except Exception as e:
    logger.error(f"✗ Failed to import training modules: {e}")
    raise

DATASET_DIR = "/app/dataset"
logger.info(f"Dataset directory: {DATASET_DIR}")
logger.info("Handler initialization complete")
logger.info("=" * 60)


def _build_config(base_config, overrides: dict):
    """Return a copy of base_config with `overrides` applied as class attrs."""
    return type("Config", (base_config,), overrides or {})


def _rewrite_data_yaml(data_yaml_path: str, local_dataset_dir: str):
    """Point data.yaml's train/val/test at the locally downloaded dataset.

    The data.yaml shipped in the dataset bakes in the Airflow container's
    absolute paths (/opt/airflow/...), which don't exist on this machine.
    """
    content = Path(data_yaml_path).read_text(encoding="utf-8")
    for split, subdir in (("train", "train"), ("val", "valid"), ("test", "test")):
        new_path = str(Path(local_dataset_dir) / subdir / "images")
        content = re.sub(
            rf"^{split}:\s*.*$",
            f"{split}: {new_path}",
            content,
            flags=re.MULTILINE,
        )
    Path(data_yaml_path).write_text(content, encoding="utf-8")


def _train_yolo(job_input: dict, mlflow_mgr: MLflowManager) -> dict:
    """Download dataset, train YOLOv8-seg, log to MLflow, return results."""
    # Set S3 credentials from job input
    os.environ["AWS_ACCESS_KEY_ID"] = job_input["s3_access_key"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = job_input["s3_secret_key"]
    
    download_dataset(
        job_input["s3_endpoint_url"],
        job_input.get("dataset_bucket", "data"),
        job_input["dataset_s3_prefix"],
        DATASET_DIR,
    )
    data_yaml = str(Path(DATASET_DIR) / job_input["data_yaml_relpath"])
    _rewrite_data_yaml(data_yaml, DATASET_DIR)

    config = _build_config(YOLOConfig, job_input.get("config_overrides"))
    with mlflow_mgr.start_run(job_input.get("run_name_prefix", "yolov8-seg-instance")) as run:
        run_id = run.info.run_id
        trainer = YOLOTrainer(config, mlflow_mgr)
        results = trainer.train(data_yaml)
        trainer.log_artifacts()
        return {
            "run_id": run_id,
            "model_path": results["model_path"],
            "metrics": results["metrics"],
        }


def _train_unet(job_input: dict, mlflow_mgr: MLflowManager) -> dict:
    """Download dataset, train U-Net, log to MLflow, return results."""
    # Set S3 credentials from job input
    os.environ["AWS_ACCESS_KEY_ID"] = job_input["s3_access_key"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = job_input["s3_secret_key"]
    
    download_dataset(
        job_input["s3_endpoint_url"],
        job_input.get("dataset_bucket", "data"),
        job_input["dataset_s3_prefix"],
        DATASET_DIR,
    )
    config = _build_config(UNetConfig, job_input.get("config_overrides"))
    factory = DataLoaderFactory(
        DATASET_DIR,
        config.IMG_SIZE,
        config.BATCH_SIZE,
        seed=getattr(config, "SEED", 2026),
    )
    train_loader, valid_loader, _ = factory.create_loaders()

    with mlflow_mgr.start_run(job_input.get("run_name_prefix", "unet")) as run:
        run_id = run.info.run_id
        trainer = UNetTrainer(config, mlflow_mgr)
        results = trainer.train(train_loader, valid_loader)
        trainer.save_model()
        return {"run_id": run_id, "best_val_loss": results["best_val_loss"]}


def handler(event):
    """RunPod entrypoint: dispatch to the right trainer based on model_type."""
    logger.info(f"Handler called with event: {event}")
    job_input = event["input"]
    model_type = job_input["model_type"]

    # MLflow's own S3 artifact client (used by mlflow.log_artifact) reads this
    # var directly; without it, it defaults to real AWS S3 instead of the
    # tunneled MinIO, and fails with InvalidAccessKeyId (MinIO creds aren't
    # valid AWS creds).
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = job_input["s3_endpoint_url"]

    mlflow_mgr = MLflowManager(job_input["mlflow_uri"], job_input["experiment_name"])

    if model_type == "yolo":
        return _train_yolo(job_input, mlflow_mgr)
    if model_type == "unet":
        return _train_unet(job_input, mlflow_mgr)
    raise ValueError(f"Unknown model_type: {model_type}")


logger.info("Starting RunPod serverless handler...")
try:
    runpod.serverless.start({"handler": handler})
    logger.info("RunPod serverless handler started successfully")
except Exception as e:
    logger.error(f"Failed to start RunPod serverless handler: {e}", exc_info=True)
    raise
