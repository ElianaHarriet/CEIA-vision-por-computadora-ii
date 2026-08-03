"""
DAG para entrenamiento del modelo de INSTANCE SEGMENTATION (YOLOv8-seg).

Este DAG sube el dataset a MinIO, dispara el entrenamiento en un endpoint de
RunPod Serverless (GPU remota) y registra el experimento resultante en MLflow.
La validación del modelo entrenado sigue corriendo localmente en CPU.

Dataset: car-damages-ready/instance/
Arquitectura: YOLOv8-seg
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

from training.config import TrainingConfig, YOLOConfig
from training.validators import InstanceDataValidator

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
    'training_instance_segmentation',
    default_args=default_args,
    description='Entrena modelo YOLOv8-seg para instance segmentation de daños en autos',
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'instance_segmentation', 'yolov8', 'model'],
)

# Configuration
DATA_PATH = TrainingConfig.get_instance_path()
DATA_YAML = f"{DATA_PATH}/data.yaml"
MLFLOW_URI = TrainingConfig.get_mlflow_uri()
EXPERIMENT_NAME = TrainingConfig.get_experiment_name()
MODEL_NAME = TrainingConfig.get_instance_model_name()

# RunPod / dataset upload configuration
DATA_BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
DATASET_S3_PREFIX = "instance"
DATA_YAML_RELPATH = "data.yaml"
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
MLFLOW_TUNNEL_LOG = "/var/log/cloudflared/mlflow.log"
S3_TUNNEL_LOG = "/var/log/cloudflared/s3.log"
LOCAL_MODEL_DOWNLOAD_DIR = "/opt/airflow/runs/segment/car_damage_instance/weights"


def check_data_availability(**context):
    """Verificar que los datos de instance/ estén disponibles."""
    validator = InstanceDataValidator(DATA_PATH)
    return validator.validate()


def upload_dataset_to_s3(**context):
    """Subir el dataset de instance/ a MinIO para que RunPod pueda descargarlo."""
    from training.s3_sync import upload_dir_to_s3
    upload_dir_to_s3(DATA_PATH, DATA_BUCKET, DATASET_S3_PREFIX)


def setup_training_environment(**context):
    """Validar que RunPod esté configurado (ya no hay GPU local que chequear)."""
    import mlflow
    from airflow.models import Variable
    if not RUNPOD_ENDPOINT_ID:
        raise ValueError("RUNPOD_ENDPOINT_ID no está configurado en .env")
    Variable.get("RUNPOD_API_KEY")  # smoke test: falla si no está en secrets
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✓ RunPod endpoint configurado: {RUNPOD_ENDPOINT_ID}")
    print(f"✓ MLflow: {MLFLOW_URI}")
    return {"runpod_endpoint_id": RUNPOD_ENDPOINT_ID}


def train_yolov8_model(**context):
    """Entrenar YOLOv8-seg en RunPod y traer el modelo resultante a disco local."""
    import mlflow
    from training.runpod_client import RunPodClient
    from training.tunnel_url import get_quick_tunnel_url

    mlflow_public_uri = get_quick_tunnel_url(MLFLOW_TUNNEL_LOG)
    s3_public_uri = get_quick_tunnel_url(S3_TUNNEL_LOG)

    # Get MinIO credentials from environment (required)
    # Docker Compose maps these as AWS_* variables
    s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not s3_access_key or not s3_secret_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set (check docker-compose.yaml)"
        )

    payload = {
        "model_type": "yolo",
        "dataset_s3_prefix": DATASET_S3_PREFIX,
        "dataset_bucket": DATA_BUCKET,
        "data_yaml_relpath": DATA_YAML_RELPATH,
        "s3_endpoint_url": s3_public_uri,
        "s3_access_key": s3_access_key,
        "s3_secret_key": s3_secret_key,
        "mlflow_uri": mlflow_public_uri,
        "experiment_name": EXPERIMENT_NAME,
        "run_name_prefix": "yolov8-seg-instance",
        "config_overrides": {},
    }

    client = RunPodClient(RUNPOD_ENDPOINT_ID)
    job_id = client.submit_job(payload)
    result = client.poll_job(job_id)
    run_id = result['run_id']
    print(f"MLflow Run ID: {run_id}")

    # El modelo se entrenó en RunPod; bajarlo de MLflow para poder validarlo
    # localmente en CPU en la siguiente task.
    mlflow.set_tracking_uri(MLFLOW_URI)
    Path(LOCAL_MODEL_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    local_model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model/best.pt",
        dst_path=LOCAL_MODEL_DOWNLOAD_DIR,
    )

    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(key='model_path', value=local_model_path)
    return {
        "run_id": run_id,
        "model_path": local_model_path,
        "metrics": result.get('metrics'),
    }


def register_model_in_registry(**context):
    """Registrar modelo en MLflow Model Registry."""
    run_id = context['ti'].xcom_pull(
        task_ids='train_yolov8_model',
        key='run_id'
    )
    if not run_id:
        raise ValueError("No run_id from training")
    from training.model_registry import YOLOModelRegistry
    registry = YOLOModelRegistry(MLFLOW_URI)
    result = registry.register_yolo(run_id, MODEL_NAME, YOLOConfig.EPOCHS)
    context['ti'].xcom_push(key='model_version', value=result['version'])
    return result


def validate_trained_model(**context):
    """Validar modelo entrenado."""
    import mlflow
    from training.yolo_trainer import YOLOValidator
    from training.mlflow_manager import MLflowManager
    model_path = context['ti'].xcom_pull(
        task_ids='train_yolov8_model',
        key='model_path'
    )
    validator = YOLOValidator(model_path, DATA_YAML)
    results = validator.validate()
    run_id = context['ti'].xcom_pull(
        task_ids='train_yolov8_model',
        key='run_id'
    )
    mlflow_mgr = MLflowManager(MLFLOW_URI, EXPERIMENT_NAME)
    mlflow_mgr._configure()
    with mlflow.start_run(run_id=run_id):
        mlflow_mgr.log_metrics(results)
    print("✓ Validation completed")
    return results


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

train_model = PythonOperator(
    task_id='train_yolov8_model',
    python_callable=train_yolov8_model,
    execution_timeout=timedelta(hours=4), 
    dag=dag,
)

register_model = PythonOperator(
    task_id='register_model_in_registry',
    python_callable=register_model_in_registry,
    dag=dag,
)

validate_model = PythonOperator(
    task_id='validate_trained_model',
    python_callable=validate_trained_model,
    dag=dag,
)

# Define dependencies
check_data >> upload_dataset >> setup_env >> train_model >> register_model >> validate_model
