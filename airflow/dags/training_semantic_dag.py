"""
DAG para entrenamiento del modelo de SEMANTIC SEGMENTATION (U-Net).

Este DAG sube el dataset a MinIO, dispara el entrenamiento en un endpoint de
RunPod Serverless (GPU remota) y registra el experimento resultante en MLflow.

Dataset: car-damages-ready/semantic/
Arquitectura: U-Net con encoder ResNet (default resnet50)
Output: Modelo entrenado + métricas en MLflow
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Add dags directory to Python path
dags_path = Path(__file__).parent
if str(dags_path) not in sys.path:
    sys.path.insert(0, str(dags_path))

from training.config import TrainingConfig, UNetConfig
from training.validators import SemanticDataValidator
from training.profile_config import RunPodProfileConfig

# DAG configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'training_semantic_segmentation',
    default_args=default_args,
    description='Entrena modelo U-Net para semantic segmentation',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'semantic_segmentation', 'unet', 'model'],
)

# Configuration
DATA_PATH = TrainingConfig.get_semantic_path()
MLFLOW_URI = TrainingConfig.get_mlflow_uri()
EXPERIMENT_NAME = TrainingConfig.get_experiment_name()
MODEL_NAME = TrainingConfig.get_semantic_model_name()

# RunPod / dataset upload configuration
DATA_BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
DATASET_S3_PREFIX = "semantic"
RUNPOD_ENDPOINT_ID = RunPodProfileConfig.get_endpoint_id()


def check_data_availability(**context):
    """Verificar que los datos de semantic/ estén disponibles."""
    validator = SemanticDataValidator(DATA_PATH)
    return validator.validate()


def upload_dataset_to_s3(**context):
    """Subir el dataset de semantic/ a MinIO para que RunPod pueda descargarlo."""
    from training.s3_sync import upload_dir_to_s3
    upload_dir_to_s3(DATA_PATH, DATA_BUCKET, DATASET_S3_PREFIX)


def setup_training_environment(**context):
    """Validar que RunPod esté configurado (ya no hay GPU local que chequear)."""
    import mlflow
    from airflow.models import Variable
    from training.runpod_client import RunPodClient
    if not RUNPOD_ENDPOINT_ID:
        raise ValueError("RUNPOD_ENDPOINT_ID no está configurado en .env")
    Variable.get("RUNPOD_API_KEY")  # smoke test: falla si no está en secrets
    RunPodClient(RUNPOD_ENDPOINT_ID).check_balance()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    RunPodProfileConfig.print_active_config()
    print(f"✓ MLflow: {MLFLOW_URI}")
    return {"runpod_endpoint_id": RUNPOD_ENDPOINT_ID}


def create_dataloaders(**context):
    """Validar que el dataset carga bien y publicar sus tamaños por XCom.

    El training real corre en RunPod (train_unet_model no reconstruye estos
    DataLoaders); esta task es solo un chequeo temprano de que el dataset
    está bien formado antes de subirlo y disparar un job de GPU remota.
    """
    from training.dataloader_factory import DataLoaderFactory
    factory = DataLoaderFactory(
        DATA_PATH,
        UNetConfig.IMG_SIZE,
        UNetConfig.BATCH_SIZE
    )
    train_loader, valid_loader, test_loader = factory.create_loaders()
    sizes = {
        'train': len(train_loader.dataset),
        'valid': len(valid_loader.dataset),
        'test': len(test_loader.dataset)
    }
    context['ti'].xcom_push(key='dataset_sizes', value=sizes)
    return sizes


def train_unet_model(**context):
    """Entrenar U-Net en RunPod y esperar el resultado."""
    from training.runpod_client import RunPodClient

    # Usar URLs directas desde .env (con DigitalOcean o localhost)
    mlflow_public_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    s3_public_uri = os.getenv("AWS_ENDPOINT_URL", "http://localhost:9000")

    # Get MinIO credentials from environment (required)
    s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not s3_access_key or not s3_secret_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set (check docker-compose.yaml)"
        )

    config_overrides = {
        "ENCODER": UNetConfig.ENCODER,
        "ENCODER_WEIGHTS": UNetConfig.ENCODER_WEIGHTS,
        "EPOCHS": UNetConfig.EPOCHS,
        "BATCH_SIZE": UNetConfig.BATCH_SIZE,
        "LEARNING_RATE": UNetConfig.LEARNING_RATE,
        "IMG_SIZE": UNetConfig.IMG_SIZE,
        "PATIENCE": UNetConfig.PATIENCE,
        "SEED": UNetConfig.SEED,
    }

    payload = {
        "model_type": "unet",
        "dataset_s3_prefix": DATASET_S3_PREFIX,
        "dataset_bucket": DATA_BUCKET,
        "s3_endpoint_url": s3_public_uri,
        "s3_access_key": s3_access_key,
        "s3_secret_key": s3_secret_key,
        "mlflow_uri": mlflow_public_uri,
        "experiment_name": EXPERIMENT_NAME,
        "run_name_prefix": "unet",
        "config_overrides": config_overrides,
    }

    client = RunPodClient(RUNPOD_ENDPOINT_ID)
    job_id = client.submit_job(payload)
    result = client.poll_job(job_id)
    run_id = result['run_id']
    print(f"MLflow Run ID: {run_id}")

    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(
        key='best_val_loss',
        value=result['best_val_loss']
    )
    return {"run_id": run_id, "best_val_loss": result['best_val_loss']}


def register_model_in_registry(**context):
    """Registrar modelo en MLflow Model Registry."""
    from training.model_registry import ModelRegistry
    run_id = context['ti'].xcom_pull(
        task_ids='train_unet_model',
        key='run_id'
    )
    if not run_id:
        raise ValueError("No run_id from training")
    desc = f"U-Net {UNetConfig.ENCODER} semantic segmentation"
    desc += f". {UNetConfig.EPOCHS} epochs."
    registry = ModelRegistry(MLFLOW_URI)
    return registry.register_model(run_id, MODEL_NAME, desc)


def validate_trained_model(**context):
    """Validar modelo entrenado."""
    print("✓ Validation completed during training")
    best_val_loss = context['ti'].xcom_pull(
        task_ids='train_unet_model',
        key='best_val_loss'
    )
    return {"best_val_loss": best_val_loss}


# Define tasks
check_data = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_availability,
    dag=dag,
)

upload_dataset = PythonOperator(
    task_id='upload_dataset_to_s3',
    python_callable=upload_dataset_to_s3,
    dag=dag,
)

setup_env = PythonOperator(
    task_id='setup_training_environment',
    python_callable=setup_training_environment,
    dag=dag,
)

# prepare_loaders = PythonOperator(
#     task_id='create_dataloaders',
#     python_callable=create_dataloaders,
#     dag=dag,
# )

train_model = PythonOperator(
    task_id='train_unet_model',
    python_callable=train_unet_model,
    dag=dag,
    # Un retry re-dispara el job en RunPod y duplica la facturación: si el
    # job falla o expira, poll_job ya lo cancela. No reintentar.
    retries=0,
)

register_model = PythonOperator(
    task_id='register_model_in_registry',
    python_callable=register_model_in_registry,
    dag=dag,
)

# validate_model = PythonOperator(
#     task_id='validate_trained_model',
#     python_callable=validate_trained_model,
#     dag=dag,
# )

# Define dependencies
# Validación local desactivada: ya se validó en RunPod con GPU
# create_dataloaders comentado: requiere albumentations (solo en RunPod)
deps = [check_data, upload_dataset, setup_env, train_model, register_model]
deps[0] >> deps[1] >> deps[2] >> deps[3] >> deps[4]
