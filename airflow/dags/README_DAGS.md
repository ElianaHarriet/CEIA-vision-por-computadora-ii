# 📋 DAGs del Proyecto

## ✅ DAGs Activos

### `data_preparation_semantic_dag.py`
**Propósito:** Descarga y prepara el dataset Car Damages desde Roboflow.

**Estado:** ✅ Implementado y funcional

**Ejecución:** Manual (trigger desde UI de Airflow)

**Salidas:**
- `car_damage_detection/car-damages/car-damages-ready/instance/` - Formato YOLOv8 para instance segmentation
- `car_damage_detection/car-damages/car-damages-ready/semantic/` - Máscaras PNG para semantic segmentation

**Tareas:**
1. `check_roboflow_api_key` - Valida configuración de API
2. `download_dataset` - Descarga desde Roboflow en formato COCO
3. `prepare_datasets` - Convierte a ambos formatos (instance + semantic)
4. `validate_prepared_data` - Valida integridad de datos

**Variables de entorno requeridas:**
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT`
- `ROBOFLOW_VERSION`

---

### `training_instance_dag.py`
**Propósito:** Entrena modelo YOLOv8-seg para instance segmentation.

**Estado:** 🚧 Esqueleto creado - Pendiente implementación

**Ejecución:** Manual (ejecutar después de preparar datos)

**Input:** `car-damages-ready/instance/`

**Output:** Modelo YOLOv8-seg entrenado + métricas en MLflow

**Tareas pendientes:**
1. ⏳ `check_data_availability` - Verificar datos instance disponibles
2. ⏳ `setup_training_environment` - Configurar entorno (GPU, librerías)
3. ⏳ `train_yolov8_model` - Entrenar YOLOv8-seg
4. ⏳ `log_to_mlflow` - Registrar experimento en MLflow
5. ⏳ `validate_trained_model` - Validar modelo entrenado

**Librerías requeridas:**
- `ultralytics` (YOLOv8)
- `mlflow`
- `torch`

---

### `training_semantic_dag.py`
**Propósito:** Entrena modelo U-Net/DeepLab para semantic segmentation.

**Estado:** 🚧 Esqueleto creado - Pendiente implementación

**Ejecución:** Manual (ejecutar después de preparar datos)

**Input:** `car-damages-ready/semantic/`

**Output:** Modelo U-Net/DeepLab entrenado + métricas en MLflow

**Tareas pendientes:**
1. ⏳ `check_data_availability` - Verificar datos semantic disponibles
2. ⏳ `setup_training_environment` - Configurar entorno
3. ⏳ `prepare_dataset_loaders` - Preparar DataLoaders
4. ⏳ `train_semantic_model` - Entrenar U-Net/DeepLab
5. ⏳ `log_to_mlflow` - Registrar experimento en MLflow
6. ⏳ `validate_trained_model` - Validar modelo entrenado

**Librerías requeridas:**
- `segmentation_models_pytorch` o `mmsegmentation`
- `mlflow`
- `torch`
- `torchvision`

---

### `evaluation_comparison_dag.py`
**Propósito:** Evalúa y compara ambos modelos entrenados.

**Estado:** 🚧 Esqueleto creado - Pendiente implementación

**Ejecución:** Manual (ejecutar después de entrenar ambos modelos)

**Input:** 
- Modelo YOLOv8-seg entrenado
- Modelo U-Net entrenado
- Test dataset

**Output:** 
- Métricas comparativas
- Reporte de comparación
- Visualizaciones
- Validación de hipótesis

**Tareas pendientes:**
1. ⏳ `check_models_availability` - Verificar modelos entrenados
2. ⏳ `load_test_dataset` - Cargar datos de test
3. ⏳ `predict_instance_model` - Predicciones con YOLOv8
4. ⏳ `predict_semantic_model` - Predicciones con U-Net
5. ⏳ `flatten_instance_masks` - Aplanar máscaras instance
6. ⏳ `calculate_metrics` - Calcular IoU, Precision, Recall, etc.
7. ⏳ `generate_comparison_report` - Generar reporte comparativo
8. ⏳ `generate_visualizations` - Crear gráficos y visualizaciones
9. ⏳ `log_comparison_to_mlflow` - Registrar en MLflow
10. ⏳ `validate_hypothesis` - Validar/refutar hipótesis

**Librerías requeridas:**
- `scikit-learn` (métricas)
- `matplotlib`, `seaborn` (visualizaciones)
- `numpy`, `opencv-python`
- `mlflow`

---

## ⚠️ DAGs Desactivados

### `data_preparation_instance_dag.py.disabled`
**Estado:** Desactivado (renombrado a `.disabled`)

**Razón:** Este DAG descargaba Dataset B (Car Damage Detection) que NO se usa en la metodología actual del proyecto.

**Metodología actual:** Se usa UN SOLO dataset (Car Damages - Dataset A) convertido a DOS formatos para entrenar ambos modelos (instance y semantic).

**Si necesitas reactivarlo:**
1. Renombrar `data_preparation_instance_dag.py.disabled` → `data_preparation_instance_dag.py`
2. Configurar variables de entorno en `.env`:
   - `ROBOFLOW_PROJECT_INSTANCE`
   - `ROBOFLOW_VERSION_INSTANCE`

---

## 📊 Flujo Completo del Proyecto

```
1. data_preparation_semantic_dag (✅ Listo)
        ↓
   ┌────┴────┐
   ↓         ↓
2a. training_instance_dag    2b. training_semantic_dag
   (🚧 Pendiente)               (🚧 Pendiente)
   ↓         ↓
   └────┬────┘
        ↓
3. evaluation_comparison_dag (🚧 Pendiente)
        ↓
   📊 Resultados finales
```

## 🔧 Cómo Implementar los DAGs Pendientes

Cada DAG pendiente tiene:
- ✅ Estructura completa de tareas
- ✅ Documentación detallada con comentarios `TODO`
- ✅ Ejemplos de código en los comentarios
- ✅ Dependencias entre tareas definidas

**Para implementar:**
1. Abrir el archivo del DAG
2. Buscar comentarios `TODO:`
3. Seguir los ejemplos de código proporcionados
4. Implementar la funcionalidad
5. Testar localmente
6. Ejecutar en Airflow

## 📖 Más Información

Ver [METODOLOGIA.md](../../METODOLOGIA.md) para entender la metodología completa del proyecto y los detalles de cada fase.
