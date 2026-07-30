"""
DAG para evaluación y comparación de modelos Instance vs Semantic.

Este DAG:
1. Carga modelos entrenados (YOLOv8-seg y U-Net)
2. Hace predicciones en test set
3. Aplana máscaras del modelo instance
4. Calcula métricas comparativas
5. Genera visualizaciones y reportes
6. Registra resultados en MLflow
7. Valida la hipótesis del proyecto
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import numpy as np
from PIL import Image

# Add dags directory to Python path
dags_path = Path(__file__).parent
if str(dags_path) not in sys.path:
    sys.path.insert(0, str(dags_path))

from evaluation.config import EvaluationConfig

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
    'evaluation_and_comparison',
    default_args=default_args,
    description='Evalúa y compara modelos instance vs semantic',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['evaluation', 'comparison', 'metrics', 'analysis'],
)

# Configuration
INSTANCE_PATH = EvaluationConfig.get_instance_path()
SEMANTIC_PATH = EvaluationConfig.get_semantic_path()
MLFLOW_URI = EvaluationConfig.get_mlflow_uri()
EXPERIMENT_NAME = EvaluationConfig.get_experiment_name()
INSTANCE_MODEL = EvaluationConfig.get_instance_model_name()
SEMANTIC_MODEL = EvaluationConfig.get_semantic_model_name()
MODEL_STAGE = EvaluationConfig.get_model_stage()
RESULTS_PATH = EvaluationConfig.get_results_path()


def check_models_availability(**context):
    """Verificar que los modelos entrenados estén disponibles."""
    from evaluation.model_loader import YOLOModelLoader, UNetModelLoader
    yolo_loader = YOLOModelLoader(MLFLOW_URI)
    unet_loader = UNetModelLoader(MLFLOW_URI)
    yolo_version = yolo_loader.check_model_availability(
        INSTANCE_MODEL,
        MODEL_STAGE
    )
    unet_version = unet_loader.check_model_availability(
        SEMANTIC_MODEL,
        MODEL_STAGE
    )
    result = {
        'instance_run_id': yolo_version.run_id,
        'instance_version': yolo_version.version,
        'semantic_run_id': unet_version.run_id,
        'semantic_version': unet_version.version,
    }
    context['ti'].xcom_push(key='model_info', value=result)
    return result


def load_test_dataset(**context):
    """Cargar dataset de test."""
    from evaluation.dataset_loader import TestDatasetLoader
    loader = TestDatasetLoader(INSTANCE_PATH, SEMANTIC_PATH)
    test_images = loader.load_test_images()
    gt_masks = loader.load_ground_truth_masks(list(test_images.keys()))
    context['ti'].xcom_push(key='test_images', value=test_images)
    context['ti'].xcom_push(key='gt_masks', value=gt_masks)
    return {
        'test_images_count': len(test_images),
        'gt_masks_count': len(gt_masks)
    }


def predict_instance_model(**context):
    """Hacer predicciones con modelo instance (YOLOv8-seg)."""
    from evaluation.model_loader import YOLOModelLoader
    from evaluation.predictor import YOLOPredictor
    test_images = context['ti'].xcom_pull(
        task_ids='load_test_dataset',
        key='test_images'
    )
    yolo_loader = YOLOModelLoader(MLFLOW_URI)
    model = yolo_loader.load_yolo_model(INSTANCE_MODEL, MODEL_STAGE)
    predictor = YOLOPredictor(model)
    predictions = predictor.predict(test_images)
    context['ti'].xcom_push(key='instance_predictions', value=predictions)
    return {'predictions_count': len(predictions)}


def predict_semantic_model(**context):
    """Hacer predicciones con modelo semantic (U-Net)."""
    from evaluation.model_loader import UNetModelLoader
    from evaluation.predictor import UNetPredictor
    test_images = context['ti'].xcom_pull(
        task_ids='load_test_dataset',
        key='test_images'
    )
    unet_loader = UNetModelLoader(MLFLOW_URI)
    model, device = unet_loader.load_unet_model(
        SEMANTIC_MODEL,
        MODEL_STAGE
    )
    predictor = UNetPredictor(model, device, (640, 640))
    predictions = predictor.predict(test_images)
    context['ti'].xcom_push(key='semantic_predictions', value=predictions)
    return {'predictions_count': len(predictions)}


def flatten_instance_masks(**context):
    """Aplanar máscaras del modelo instance."""
    from evaluation.mask_flattener import MaskFlattener
    predictions = context['ti'].xcom_pull(
        task_ids='predict_instance_model',
        key='instance_predictions'
    )
    flattener = MaskFlattener(strategy='or')
    flattened = flattener.flatten_predictions(predictions)
    context['ti'].xcom_push(key='flattened_predictions', value=flattened)
    return {'flattened_count': len(flattened)}


def calculate_metrics(**context):
    """Calcular métricas de evaluación para ambos modelos."""
    from evaluation.metrics_calculator import MetricsCalculator, ComparisonCalculator
    gt_masks = context['ti'].xcom_pull(
        task_ids='load_test_dataset',
        key='gt_masks'
    )
    instance_preds = context['ti'].xcom_pull(
        task_ids='flatten_instance_masks',
        key='flattened_predictions'
    )
    semantic_preds = context['ti'].xcom_pull(
        task_ids='predict_semantic_model',
        key='semantic_predictions'
    )
    calculator = MetricsCalculator(EvaluationConfig.NUM_CLASSES)
    metrics_instance = calculator.calculate_all_metrics(
        instance_preds,
        gt_masks
    )
    metrics_semantic = calculator.calculate_all_metrics(
        semantic_preds,
        gt_masks
    )
    comparator = ComparisonCalculator()
    comparison = comparator.compare_metrics(
        metrics_instance,
        metrics_semantic
    )
    result = {
        'instance_metrics': metrics_instance,
        'semantic_metrics': metrics_semantic,
        'comparison': comparison
    }
    context['ti'].xcom_push(key='all_metrics', value=result)
    return {
        'instance_mean_iou': metrics_instance['aggregated']['mean_iou'],
        'semantic_mean_iou': metrics_semantic['aggregated']['mean_iou']
    }


def generate_visualizations(**context):
    """Generar visualizaciones de comparación."""
    from evaluation.visualizer import ComparisonVisualizer
    all_metrics = context['ti'].xcom_pull(
        task_ids='calculate_metrics',
        key='all_metrics'
    )
    test_images = context['ti'].xcom_pull(
        task_ids='load_test_dataset',
        key='test_images'
    )
    gt_masks = context['ti'].xcom_pull(
        task_ids='load_test_dataset',
        key='gt_masks'
    )
    instance_preds = context['ti'].xcom_pull(
        task_ids='flatten_instance_masks',
        key='flattened_predictions'
    )
    semantic_preds = context['ti'].xcom_pull(
        task_ids='predict_semantic_model',
        key='semantic_predictions'
    )
    samples = _prepare_samples(
        test_images,
        gt_masks,
        instance_preds,
        semantic_preds,
        all_metrics
    )
    viz_data = {
        'comparison': all_metrics['comparison'],
        'samples': samples
    }
    visualizer = ComparisonVisualizer(
        RESULTS_PATH,
        EvaluationConfig.CLASS_NAMES
    )
    viz_path = visualizer.generate_all_visualizations(viz_data)
    context['ti'].xcom_push(key='visualizations_path', value=viz_path)
    return {'visualizations_path': viz_path}


def _prepare_samples(test_imgs, gt_masks, inst_preds, sem_preds, metrics):
    """Prepare sample data for visualization."""
    samples = {}
    img_metrics_inst = metrics['instance_metrics']['per_image']
    img_metrics_sem = metrics['semantic_metrics']['per_image']
    count = 0
    for name in test_imgs.keys():
        if count >= 5:
            break
        if name not in gt_masks:
            continue
        if name not in inst_preds or inst_preds[name] is None:
            continue
        if name not in sem_preds:
            continue
        img = np.array(Image.open(test_imgs[name]).convert('RGB'))
        samples[name] = {
            'image': img,
            'gt': gt_masks[name],
            'pred_instance': inst_preds[name],
            'pred_semantic': sem_preds[name],
            'iou_instance': img_metrics_inst.get(name, {}).get('iou_global', 0),
            'iou_semantic': img_metrics_sem.get(name, {}).get('iou_global', 0),
        }
        count += 1
    return samples


def generate_comparison_report(**context):
    """Generar reporte de comparación."""
    from evaluation.report_generator import ComparisonReport
    all_metrics = context['ti'].xcom_pull(
        task_ids='calculate_metrics',
        key='all_metrics'
    )
    hypothesis = context['ti'].xcom_pull(
        task_ids='validate_hypothesis',
        key='hypothesis_result'
    )
    report_data = {
        'comparison': all_metrics['comparison'],
        'hypothesis': hypothesis,
        'instance_metrics': all_metrics['instance_metrics']['aggregated'],
        'semantic_metrics': all_metrics['semantic_metrics']['aggregated'],
    }
    reporter = ComparisonReport(
        RESULTS_PATH,
        EvaluationConfig.CLASS_NAMES
    )
    report_path = reporter.generate_report(report_data)
    context['ti'].xcom_push(key='report_path', value=report_path)
    return {'report_path': report_path}


def validate_hypothesis(**context):
    """Validar o refutar la hipótesis del proyecto."""
    from evaluation.hypothesis_validator import HypothesisValidator
    all_metrics = context['ti'].xcom_pull(
        task_ids='calculate_metrics',
        key='all_metrics'
    )
    validator = HypothesisValidator()
    result = validator.validate(
        all_metrics['instance_metrics'],
        all_metrics['semantic_metrics']
    )
    context['ti'].xcom_push(key='hypothesis_result', value=result)
    return result


def log_comparison_to_mlflow(**context):
    """Registrar comparación completa en MLflow."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    all_metrics = context['ti'].xcom_pull(
        task_ids='calculate_metrics',
        key='all_metrics'
    )
    hypothesis = context['ti'].xcom_pull(
        task_ids='validate_hypothesis',
        key='hypothesis_result'
    )
    viz_path = context['ti'].xcom_pull(
        task_ids='generate_visualizations',
        key='visualizations_path'
    )
    report_path = context['ti'].xcom_pull(
        task_ids='generate_comparison_report',
        key='report_path'
    )
    model_info = context['ti'].xcom_pull(
        task_ids='check_models_availability',
        key='model_info'
    )
    with mlflow.start_run(run_name='comparison-instance-vs-semantic'):
        _log_metrics_to_mlflow(all_metrics)
        _log_params_to_mlflow(model_info)
        _log_artifacts_to_mlflow(viz_path, report_path)
        mlflow.log_param('hypothesis_validated', hypothesis['validated'])
        run_id = mlflow.active_run().info.run_id
    print(f"✓ Logged to MLflow (Run ID: {run_id})")
    return {'mlflow_run_id': run_id}


def _log_metrics_to_mlflow(all_metrics):
    """Log metrics to MLflow."""
    import mlflow
    comp = all_metrics['comparison']
    for key, vals in comp.items():
        if key.startswith('mean_'):
            mlflow.log_metric(f"instance_{key}", vals['model_a'])
            mlflow.log_metric(f"semantic_{key}", vals['model_b'])
            mlflow.log_metric(f"diff_{key}", vals['difference'])


def _log_params_to_mlflow(model_info):
    """Log params to MLflow."""
    import mlflow
    mlflow.log_param('instance_run_id', model_info['instance_run_id'])
    mlflow.log_param('semantic_run_id', model_info['semantic_run_id'])
    mlflow.log_param('instance_version', model_info['instance_version'])
    mlflow.log_param('semantic_version', model_info['semantic_version'])


def _log_artifacts_to_mlflow(viz_path, report_path):
    """Log artifacts to MLflow."""
    import mlflow
    viz_dir = Path(viz_path)
    for file in viz_dir.glob('*.png'):
        mlflow.log_artifact(str(file), 'visualizations')
    mlflow.log_artifact(report_path, 'reports')
    json_path = Path(viz_path) / 'comparison_data.json'
    if json_path.exists():
        mlflow.log_artifact(str(json_path), 'data')


# Define tasks
check_models = PythonOperator(
    task_id='check_models_availability',
    python_callable=check_models_availability,
    dag=dag,
)

load_test = PythonOperator(
    task_id='load_test_dataset',
    python_callable=load_test_dataset,
    dag=dag,
)

predict_instance = PythonOperator(
    task_id='predict_instance_model',
    python_callable=predict_instance_model,
    dag=dag,
)

predict_semantic = PythonOperator(
    task_id='predict_semantic_model',
    python_callable=predict_semantic_model,
    dag=dag,
)

flatten_masks = PythonOperator(
    task_id='flatten_instance_masks',
    python_callable=flatten_instance_masks,
    dag=dag,
)

calc_metrics = PythonOperator(
    task_id='calculate_metrics',
    python_callable=calculate_metrics,
    dag=dag,
)

validate_hyp = PythonOperator(
    task_id='validate_hypothesis',
    python_callable=validate_hypothesis,
    dag=dag,
)

gen_viz = PythonOperator(
    task_id='generate_visualizations',
    python_callable=generate_visualizations,
    dag=dag,
)

gen_report = PythonOperator(
    task_id='generate_comparison_report',
    python_callable=generate_comparison_report,
    dag=dag,
)

log_mlflow = PythonOperator(
    task_id='log_comparison_to_mlflow',
    python_callable=log_comparison_to_mlflow,
    dag=dag,
)

# Define dependencies
check_models >> load_test
load_test >> [predict_instance, predict_semantic]
predict_instance >> flatten_masks
[flatten_masks, predict_semantic] >> calc_metrics
calc_metrics >> validate_hyp >> gen_viz >> gen_report >> log_mlflow
