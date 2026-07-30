"""
DAG para preparación de datos de SEGMENTACIÓN SEMÁNTICA - Dataset A (Car Damages).

Este DAG descarga el dataset de Roboflow con anotaciones de segmentación semántica
y prepara los datos en formato YOLOv8 (instance segmentation) y máscaras PNG (semantic segmentation).

Dataset: Car Damages - Semantic Segmentation
Clases: Minor Damage (Dent), Minor Damage (Scratch), No Damage, Severe Damage

Refactorizado siguiendo principios SOLID y Clean Code.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path

# Add dags directory to Python path
dags_path = Path(__file__).parent
if str(dags_path) not in sys.path:
    sys.path.insert(0, str(dags_path))

from core.config import RoboflowConfig, DatasetPaths
from core.roboflow_client import RoboflowClient
from core.file_operations import FileSystemOperations
from core.logger import DatasetLogger
from core.validators import DatasetValidator
from semantic.dataset_preparer import SemanticDatasetPreparer


# Configuration
CLASS_NAMES = ["Minor Damage (Dent)", "Minor Damage (Scratch)", "No Damage", "Severe Damage"]
NAME_MAP = {
    "Minor Damage -Dent-": "Minor Damage (Dent)",
    "Minor Damage -Scratch-": "Minor Damage (Scratch)",
    "No Damage": "No Damage",
    "Severe Damage": "Severe Damage",
}
SPLITS = ["train", "valid", "test"]

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
    'data_preparation_semantic',
    default_args=default_args,
    description='Descarga y prepara el dataset de segmentación semántica (Car Damages) desde Roboflow',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data', 'preprocessing', 'semantic_segmentation', 'dataset_a'],
)


def check_roboflow_api_key(**context):
    """Verify Roboflow API key is configured."""
    config = RoboflowConfig.from_env('ROBOFLOW_PROJECT', 'ROBOFLOW_VERSION')
    print("✓ ROBOFLOW_API_KEY configurada correctamente")
    return config.api_key


def download_dataset_task(**context):
    """Download dataset from Roboflow."""
    config = RoboflowConfig.from_env('ROBOFLOW_PROJECT', 'ROBOFLOW_VERSION')
    paths = DatasetPaths.create(
        "/opt/airflow/car_damage_detection/car-damages",
        "car-damages-forked",
        "car-damages-ready"
    )
    
    fs = FileSystemOperations()
    
    # Check if dataset already exists
    if _dataset_already_exists(fs, paths):
        print(f"✓ Dataset ya descargado en {paths.raw}")
        print("⏭️  Saltando descarga (usar 'Clear' en Airflow UI para forzar re-descarga)")
        return str(paths.raw)
    
    _clean_existing_data(fs, paths)
    _download_from_roboflow(config, paths, fs)
    _log_download_stats(paths, fs)
    return str(paths.raw)


def _dataset_already_exists(fs: FileSystemOperations, paths: DatasetPaths) -> bool:
    """Check if dataset was already downloaded."""
    if not paths.raw.exists():
        return False
    
    # Verify that all splits exist and have images
    for split in SPLITS:
        split_dir = paths.raw / split
        if not split_dir.exists():
            return False
        if fs.count_files(split_dir, "*.jpg") == 0:
            return False
    
    return True


def _clean_existing_data(fs: FileSystemOperations, paths: DatasetPaths) -> None:
    """Clean existing dataset files."""
    fs.remove_directory(paths.raw)
    fs.remove_directory(paths.ready)
    print("✓ Datos anteriores eliminados")


def _download_from_roboflow(config: RoboflowConfig, paths: DatasetPaths, fs: FileSystemOperations) -> None:
    """Download dataset using Roboflow client."""
    fs.ensure_directory(paths.base)
    DatasetLogger.log_download_start(config.workspace, config.project, config.version)
    
    client = RoboflowClient(config)
    client.download_dataset("coco-segmentation", paths.raw)
    DatasetLogger.log_download_complete(str(paths.raw))


def _log_download_stats(paths: DatasetPaths, fs: FileSystemOperations) -> None:
    """Log download statistics."""
    for split in SPLITS:
        split_dir = paths.raw / split
        count = len(fs.list_images(split_dir))
        DatasetLogger.log_split_stats(split, count)


def prepare_datasets_task(**context):
    """Prepare datasets in instance and semantic formats."""
    paths = DatasetPaths.create(
        "/opt/airflow/car_damage_detection/car-damages",
        "car-damages-forked",
        "car-damages-ready"
    )
    
    preparer = SemanticDatasetPreparer(paths.raw, paths.ready, CLASS_NAMES, NAME_MAP)
    stats = _prepare_all_splits(preparer)
    preparer.create_data_yaml()
    
    print(f"\n✓ Preparación completada")
    print(f"📁 Datos disponibles en: {paths.ready}")
    return stats


def _prepare_all_splits(preparer: SemanticDatasetPreparer) -> dict:
    """Prepare all data splits."""
    stats = {}
    for split in SPLITS:
        DatasetLogger.log_processing_start(split)
        count = preparer.prepare_split(split)
        stats[split] = {'processed': count}
        print(f"  ✓ {count} imágenes procesadas")
    return stats


def validate_prepared_data(**context):
    """Validate prepared dataset integrity."""
    paths = DatasetPaths.create(
        "/opt/airflow/car_damage_detection/car-damages",
        "car-damages-forked",
        "car-damages-ready"
    )
    
    validator = DatasetValidator(paths.ready)
    _validate_structure(validator)
    stats = _collect_validation_stats(paths)
    _validate_data_yaml(paths)
    
    DatasetLogger.log_validation_success()
    DatasetLogger.log_stats(stats)
    context['ti'].xcom_push(key='dataset_stats', value=stats)
    return stats


def _validate_structure(validator: DatasetValidator) -> None:
    """Validate directory structure."""
    required_dirs = [
        "instance/train/images", "instance/train/labels",
        "instance/valid/images", "instance/valid/labels",
        "instance/test/images", "instance/test/labels",
        "semantic/train/images", "semantic/train/masks",
        "semantic/valid/images", "semantic/valid/masks",
        "semantic/test/images", "semantic/test/masks",
    ]
    validator.validate_directories(required_dirs)


def _collect_validation_stats(paths: DatasetPaths) -> dict:
    """Collect validation statistics."""
    fs = FileSystemOperations()
    stats = {}
    
    for split in SPLITS:
        inst_images = fs.count_files(paths.ready / "instance" / split / "images")
        inst_labels = fs.count_files(paths.ready / "instance" / split / "labels", "*.txt")
        sem_images = fs.count_files(paths.ready / "semantic" / split / "images")
        sem_masks = fs.count_files(paths.ready / "semantic" / split / "masks", "*.png")
        
        stats[split] = {
            'instance_images': inst_images,
            'instance_labels': inst_labels,
            'semantic_images': sem_images,
            'semantic_masks': sem_masks
        }
    
    return stats


def _validate_data_yaml(paths: DatasetPaths) -> None:
    """Validate data.yaml exists."""
    validator = DatasetValidator(paths.ready)
    validator.validate_file_exists("instance/data.yaml")
    print(f"✓ data.yaml encontrado")


# Define tasks
check_api_key = PythonOperator(
    task_id='check_roboflow_api_key',
    python_callable=check_roboflow_api_key,
    dag=dag,
)

download_dataset = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset_task,
    execution_timeout=timedelta(minutes=30),
    dag=dag,
)

prepare_datasets = PythonOperator(
    task_id='prepare_datasets',
    python_callable=prepare_datasets_task,
    dag=dag,
)

validate_data = PythonOperator(
    task_id='validate_prepared_data',
    python_callable=validate_prepared_data,
    dag=dag,
)

# Define dependencies
check_api_key >> download_dataset >> prepare_datasets >> validate_data
