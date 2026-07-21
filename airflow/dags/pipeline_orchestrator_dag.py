"""
DAG Orquestador - Pipeline completo de segmentación de daños en autos.

Este DAG ejecuta todo el pipeline en secuencia:
1. Preparación de datos (descarga y formateo)
2. Entrenamiento modelo instance (YOLOv8-seg)
3. Entrenamiento modelo semantic (U-Net)
4. Evaluación y comparación de modelos
"""
from datetime import datetime, timedelta
from airflow import DAG  # pyright: ignore
from airflow.operators.trigger_dagrun import TriggerDagRunOperator  # pyright: ignore

default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(hours=6)  # Pipeline puede tomar varias horas
}

with DAG(
    'pipeline_orchestrator',
    default_args=default_args,
    description='Orquesta pipeline completo: datos → entrenamiento → evaluación',
    tags=['orchestrator', 'pipeline', 'full_workflow'],
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    # Paso 1: Preparación de datos
    trigger_data_prep = TriggerDagRunOperator(
        task_id='trigger_data_preparation',
        trigger_dag_id='data_preparation_semantic',
        wait_for_completion=True,  # Espera a que termine antes de continuar
        poke_interval=30,
        reset_dag_run=True,
    )
    
    # Paso 2: Entrenamiento modelo instance (YOLOv8)
    trigger_train_instance = TriggerDagRunOperator(
        task_id='trigger_training_instance',
        trigger_dag_id='training_instance_segmentation',
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True,
    )
    
    # Paso 3: Entrenamiento modelo semantic (U-Net)
    trigger_train_semantic = TriggerDagRunOperator(
        task_id='trigger_training_semantic',
        trigger_dag_id='training_semantic_segmentation',
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True,
    )
    
    # Paso 4: Evaluación y comparación
    trigger_evaluation = TriggerDagRunOperator(
        task_id='trigger_evaluation',
        trigger_dag_id='evaluation_and_comparison',
        wait_for_completion=False,  # Última tarea, no necesita esperar
    )
    
    # Define el flujo del pipeline
    (
        trigger_data_prep
        >> trigger_train_instance
        >> trigger_train_semantic
        >> trigger_evaluation
    )  # pyright: ignore
