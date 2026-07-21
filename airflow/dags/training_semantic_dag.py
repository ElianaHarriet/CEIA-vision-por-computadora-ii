"""
DAG para entrenamiento del modelo de SEMANTIC SEGMENTATION (U-Net).

Este DAG entrena un modelo U-Net usando los datos preparados
en formato semantic y registra el experimento en MLflow.

Dataset: car-damages-ready/semantic/
Arquitectura: U-Net con encoder ResNet34
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

from training.config import TrainingConfig, UNetConfig
from training.validators import SemanticDataValidator
from training.environment import UNetEnvironment
from training.mlflow_manager import MLflowManager
from training.dataloader_factory import DataLoaderFactory
from training.unet_trainer import UNetTrainer
from training.model_registry import ModelRegistry

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


def check_data_availability(**context):
    """Verificar que los datos de semantic/ estén disponibles."""
    validator = SemanticDataValidator(DATA_PATH)
    return validator.validate()


def setup_training_environment(**context):
    """Configurar entorno de entrenamiento."""
    env_setup = UNetEnvironment(MLFLOW_URI, EXPERIMENT_NAME)
    return env_setup.setup()


def create_dataloaders(**context):
    """Crear DataLoaders para train, valid, test."""
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
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader
    }


def train_unet_model(**context):
    """Entrenar modelo U-Net."""
    factory = DataLoaderFactory(
        DATA_PATH,
        UNetConfig.IMG_SIZE,
        UNetConfig.BATCH_SIZE
    )
    train_loader, valid_loader, _ = factory.create_loaders()
    mlflow_mgr = MLflowManager(MLFLOW_URI, EXPERIMENT_NAME)
    with mlflow_mgr.start_run("unet") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        trainer = UNetTrainer(UNetConfig, mlflow_mgr)
        results = trainer.train(train_loader, valid_loader)
        trainer.save_model()
        context['ti'].xcom_push(key='run_id', value=run_id)
        context['ti'].xcom_push(
            key='best_val_loss',
            value=results['best_val_loss']
        )
        return {"run_id": run_id, "best_val_loss": results['best_val_loss']}


def register_model_in_registry(**context):
    """Registrar modelo en MLflow Model Registry."""
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

setup_env = PythonOperator(
    task_id='setup_training_environment',
    python_callable=setup_training_environment,
    dag=dag,
)

prepare_loaders = PythonOperator(
    task_id='create_dataloaders',
    python_callable=create_dataloaders,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_unet_model',
    python_callable=train_unet_model,
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
deps = [check_data, setup_env, prepare_loaders]
deps += [train_model, register_model, validate_model]
deps[0] >> deps[1] >> deps[2] >> deps[3] >> deps[4] >> deps[5]
