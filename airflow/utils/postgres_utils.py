"""
Utilities for PostgreSQL database operations.
"""

import json
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any, Optional
import os


def get_postgres_connection():
    """
    Create PostgreSQL connection using environment variables.
    
    Returns:
        psycopg2.connection: Database connection
    """
    return psycopg2.connect(
        host='postgres',
        port=5432,
        database=os.getenv('PG_DATABASE', 'airflow'),
        user=os.getenv('PG_USER', 'airflow'),
        password=os.getenv('PG_PASSWORD', 'airflow')
    )

def ensure_metrics_table_exists():
    """
    Create ml_metrics schema and model_training table if they don't exist.
    This ensures the table is available even if the initialization script didn't run.
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        # Create schema
        cur.execute("CREATE SCHEMA IF NOT EXISTS ml_metrics;")
        
        # Create table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ml_metrics.model_training (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(255) NOT NULL,
            model_name VARCHAR(255) NOT NULL,
            experiment_name VARCHAR(255),
            training_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            
            -- Metrics
            train_r2 FLOAT,
            test_r2 FLOAT,
            train_mae FLOAT,
            test_mae FLOAT,
            train_rmse FLOAT,
            test_rmse FLOAT,
            test_mape FLOAT,
            
            -- Optuna metrics
            optuna_best_cv_rmse FLOAT,
            n_trials INT,
            
            -- Model parameters (stored as JSONB for flexibility)
            model_params JSONB,
            
            -- Dataset info
            train_size INT,
            test_size INT,
            n_features INT,
            target_mean FLOAT,
            target_std FLOAT,
            
            -- Validation
            is_valid BOOLEAN,
            is_promoted BOOLEAN DEFAULT FALSE,
            model_version VARCHAR(50),
            model_stage VARCHAR(50),
            
            -- Metadata
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(run_id)
        );
        """
        cur.execute(create_table_query)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_training_date ON ml_metrics.model_training(training_date DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON ml_metrics.model_training(model_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON ml_metrics.model_training(run_id);")
        
        # Grant permissions
        cur.execute("GRANT USAGE ON SCHEMA ml_metrics TO airflow;")
        cur.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_metrics TO airflow;")
        cur.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_metrics TO airflow;")
        
        conn.commit()
        print("✓ Tabla ml_metrics.model_training verificada/creada")
        
    except Exception as e:
        conn.rollback()
        print(f"⚠ Error verificando tabla: {str(e)}")
    finally:
        cur.close()
        conn.close()

def insert_training_metrics(
    run_id: str,
    model_name: str,
    experiment_name: str,
    metrics: Dict[str, float],
    model_params: Dict[str, Any],
    data_info: Dict[str, Any],
    is_valid: bool,
    optuna_cv_rmse: Optional[float] = None,
    n_trials: Optional[int] = None
) -> int:
    """
    Insert training metrics into PostgreSQL database.
    
    Args:
        run_id: MLflow run ID
        model_name: Name of the model
        experiment_name: Name of the experiment
        metrics: Dictionary with training metrics
        model_params: Model hyperparameters
        data_info: Dataset information
        is_valid: Whether model passed validation
        optuna_cv_rmse: Optuna best CV RMSE (optional)
        n_trials: Number of Optuna trials (optional)
        
    Returns:
        int: ID of inserted record
    """
    ensure_metrics_table_exists()
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        insert_query = """
        INSERT INTO ml_metrics.model_training (
            run_id, model_name, experiment_name,
            train_r2, test_r2, train_mae, test_mae, train_rmse, test_rmse, test_mape,
            optuna_best_cv_rmse, n_trials,
            model_params,
            train_size, test_size, n_features, target_mean, target_std,
            is_valid
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s,
            %s,
            %s, %s, %s, %s, %s,
            %s
        )
        ON CONFLICT (run_id) 
        DO UPDATE SET
            test_r2 = EXCLUDED.test_r2,
            test_mae = EXCLUDED.test_mae,
            test_rmse = EXCLUDED.test_rmse,
            test_mape = EXCLUDED.test_mape,
            is_valid = EXCLUDED.is_valid
        RETURNING id;
        """
        
        cur.execute(insert_query, (
            run_id, model_name, experiment_name,
            metrics.get('train_r2'), metrics.get('test_r2'),
            metrics.get('train_mae'), metrics.get('test_mae'),
            metrics.get('train_rmse'), metrics.get('test_rmse'),
            metrics.get('test_mape'),
            optuna_cv_rmse, n_trials,
            Json(model_params),
            data_info.get('train_size'), data_info.get('test_size'),
            data_info.get('n_features'), data_info.get('target_mean'),
            data_info.get('target_std'),
            is_valid
        ))
        
        record_id = cur.fetchone()[0]
        conn.commit()
        
        print(f"✓ Métricas guardadas en PostgreSQL (ID: {record_id})")
        return record_id
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error guardando métricas en PostgreSQL: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()


def update_model_promotion(run_id: str, model_version: str, model_stage: str) -> None:
    """
    Update model promotion status in PostgreSQL.
    
    Args:
        run_id: MLflow run ID
        model_version: Model version number
        model_stage: Model stage (Production, Staging, etc.)
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        update_query = """
        UPDATE ml_metrics.model_training
        SET is_promoted = TRUE,
            model_version = %s,
            model_stage = %s
        WHERE run_id = %s;
        """
        
        cur.execute(update_query, (model_version, model_stage, run_id))
        conn.commit()
        
        print(f"✓ Estado de promoción actualizado en PostgreSQL")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error actualizando promoción en PostgreSQL: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()


def get_latest_metrics(model_name: str, limit: int = 10) -> list:
    """
    Get latest training metrics for a model.
    
    Args:
        model_name: Name of the model
        limit: Number of records to retrieve
        
    Returns:
        list: List of metric records
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        query = """
        SELECT 
            run_id, training_date, test_r2, test_mae, test_rmse, test_mape,
            is_valid, is_promoted, model_version, model_stage
        FROM ml_metrics.model_training
        WHERE model_name = %s
        ORDER BY training_date DESC
        LIMIT %s;
        """
        
        cur.execute(query, (model_name, limit))
        results = cur.fetchall()
        
        return results
        
    finally:
        cur.close()
        conn.close()
