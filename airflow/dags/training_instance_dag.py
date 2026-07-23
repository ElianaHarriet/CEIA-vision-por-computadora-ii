"""
DAG para entrenamiento del modelo de INSTANCE SEGMENTATION (YOLOv8-seg).

Este DAG entrena un modelo YOLOv8-seg usando los datos preparados
en formato instance y registra el experimento en MLflow.

Dataset: car-damages-ready/instance/
Arquitectura: YOLOv8-seg
Output: Modelo entrenado + métricas en MLflow
"""
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

def check_data_availability(**context):
    """Verificar que los datos de instance/ estén disponibles."""
    validator = InstanceDataValidator(DATA_PATH)
    return validator.validate()


def setup_training_environment(**context):
    """Configurar entorno de entrenamiento."""
    from training.environment import YOLOEnvironment
    env_setup = YOLOEnvironment(MLFLOW_URI, EXPERIMENT_NAME)
    return env_setup.setup()


def train_yolov8_model(**context):
    """Entrenar modelo YOLOv8-seg."""
    from training.mlflow_manager import MLflowManager
    from training.yolo_trainer import YOLOTrainer
    mlflow_mgr = MLflowManager(MLFLOW_URI, EXPERIMENT_NAME)
    with mlflow_mgr.start_run("yolov8-seg-instance") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        trainer = YOLOTrainer(YOLOConfig, mlflow_mgr)
        results = trainer.train(DATA_YAML)
        trainer.log_artifacts()
        context['ti'].xcom_push(key='run_id', value=run_id)
        context['ti'].xcom_push(key='model_path', value=results['model_path'])
        return {
            "run_id": run_id,
            "model_path": results['model_path'],
            "metrics": results['metrics']
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

setup_env = PythonOperator(
    task_id='setup_training_environment',
    python_callable=setup_training_environment,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_yolov8_model',
    python_callable=train_yolov8_model,
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
check_data >> setup_env >> train_model >> register_model >> validate_model
