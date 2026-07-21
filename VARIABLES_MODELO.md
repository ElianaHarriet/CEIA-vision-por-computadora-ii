# 🔧 Variables de Modelo - Guía de Uso

## 📋 Variables Configuradas en `.env`

```bash
# MLflow Experiment Name (agrupa todos los runs)
MLFLOW_EXPERIMENT_NAME=car-damage-segmentation

# Nombres de modelos en MLflow Model Registry
MODEL_NAME_INSTANCE=car-damage-instance-segmentation
MODEL_NAME_SEMANTIC=car-damage-semantic-segmentation

# Configuración de despliegue
MODEL_STAGE=Production
MODEL_VERSION_INSTANCE=latest
MODEL_VERSION_SEMANTIC=latest
```

## 🎯 Cómo Usar en el Código

### 1. En Training DAGs (Airflow)

#### Training Instance DAG:
```python
import os
import mlflow

# Configurar experimento
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "car-damage-segmentation")
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# Entrenar modelo
with mlflow.start_run(run_name="yolov8s-seg-v1"):
    # ... código de entrenamiento ...
    
    # Registrar modelo en Model Registry
    MODEL_NAME = os.getenv("MODEL_NAME_INSTANCE", "car-damage-instance-segmentation")
    mlflow.pytorch.log_model(model, "model")
    
    # Registrar en Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, MODEL_NAME)
```

#### Training Semantic DAG:
```python
import os
import mlflow

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "car-damage-segmentation")
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="unet-resnet34-v1"):
    # ... código de entrenamiento ...
    
    MODEL_NAME = os.getenv("MODEL_NAME_SEMANTIC", "car-damage-semantic-segmentation")
    mlflow.pytorch.log_model(model, "model")
    
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, MODEL_NAME)
```

### 2. En FastAPI (Model Serving)

#### Cargar modelos al inicio:
```python
import os
import mlflow
from fastapi import FastAPI

app = FastAPI()

# Variables de entorno
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME_INSTANCE = os.getenv("MODEL_NAME_INSTANCE", "car-damage-instance-segmentation")
MODEL_NAME_SEMANTIC = os.getenv("MODEL_NAME_SEMANTIC", "car-damage-semantic-segmentation")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Configurar MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Variables globales para modelos
instance_model = None
semantic_model = None

@app.on_event("startup")
async def load_models():
    """Cargar modelos al iniciar la API"""
    global instance_model, semantic_model
    
    try:
        # Cargar modelo instance
        instance_model_uri = f"models:/{MODEL_NAME_INSTANCE}/{MODEL_STAGE}"
        instance_model = mlflow.pyfunc.load_model(instance_model_uri)
        print(f"✓ Modelo instance cargado: {instance_model_uri}")
        
        # Cargar modelo semantic
        semantic_model_uri = f"models:/{MODEL_NAME_SEMANTIC}/{MODEL_STAGE}"
        semantic_model = mlflow.pytorch.load_model(semantic_model_uri)
        print(f"✓ Modelo semantic cargado: {semantic_model_uri}")
        
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        raise
```

#### Usar en endpoints:
```python
@app.post("/predict/instance")
async def predict_instance(file: UploadFile):
    """Predicción con modelo instance"""
    if instance_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Procesar imagen
    image = await process_upload(file)
    
    # Predicción
    prediction = instance_model.predict(image)
    
    return {"prediction": prediction}

@app.post("/predict/semantic")
async def predict_semantic(file: UploadFile):
    """Predicción con modelo semantic"""
    if semantic_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    image = await process_upload(file)
    prediction = semantic_model(image)
    
    return {"prediction": prediction}
```

#### Endpoint de información de modelos:
```python
@app.get("/models/info")
async def models_info():
    """Info sobre modelos cargados"""
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # Info modelo instance
    instance_versions = client.search_model_versions(f"name='{MODEL_NAME_INSTANCE}'")
    instance_prod = [v for v in instance_versions if v.current_stage == MODEL_STAGE]
    
    # Info modelo semantic
    semantic_versions = client.search_model_versions(f"name='{MODEL_NAME_SEMANTIC}'")
    semantic_prod = [v for v in semantic_versions if v.current_stage == MODEL_STAGE]
    
    return {
        "instance_model": {
            "name": MODEL_NAME_INSTANCE,
            "stage": MODEL_STAGE,
            "version": instance_prod[0].version if instance_prod else None,
            "loaded": instance_model is not None
        },
        "semantic_model": {
            "name": MODEL_NAME_SEMANTIC,
            "stage": MODEL_STAGE,
            "version": semantic_prod[0].version if semantic_prod else None,
            "loaded": semantic_model is not None
        }
    }
```

### 3. En Evaluation DAG

```python
import os
import mlflow

# Cargar modelos para evaluación
MODEL_NAME_INSTANCE = os.getenv("MODEL_NAME_INSTANCE")
MODEL_NAME_SEMANTIC = os.getenv("MODEL_NAME_SEMANTIC")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri("http://mlflow:5000")

# Cargar modelos
instance_model_uri = f"models:/{MODEL_NAME_INSTANCE}/{MODEL_STAGE}"
semantic_model_uri = f"models:/{MODEL_NAME_SEMANTIC}/{MODEL_STAGE}"

instance_model = mlflow.pyfunc.load_model(instance_model_uri)
semantic_model = mlflow.pytorch.load_model(semantic_model_uri)

# ... código de evaluación ...
```

## 🔄 Transiciones de Stage

### Promover modelo a Production:
```python
import mlflow

client = mlflow.MlflowClient(tracking_uri="http://mlflow:5000")

MODEL_NAME = os.getenv("MODEL_NAME_INSTANCE")

# Promover versión 2 a Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=2,
    stage="Production"
)

# Archivar versión anterior (versión 1)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=1,
    stage="Archived"
)
```

## 📊 Stages Disponibles

- **None** - Modelo recién registrado
- **Staging** - Modelo en pruebas
- **Production** - Modelo en producción
- **Archived** - Modelo antiguo archivado

## 🎨 Cambiar Nombres de Modelos

Si quieres cambiar los nombres, solo edita `.env`:

```bash
# Opción corta
MODEL_NAME_INSTANCE=yolov8-seg
MODEL_NAME_SEMANTIC=unet

# Opción con arquitectura
MODEL_NAME_INSTANCE=yolov8s-seg-car-damage
MODEL_NAME_SEMANTIC=unet-resnet34-car-damage

# Opción con versión
MODEL_NAME_INSTANCE=car-damage-yolov8-v1
MODEL_NAME_SEMANTIC=car-damage-unet-v1
```

Luego reinicia los servicios:
```bash
docker compose --profile all restart
```

## 🔍 Verificar Variables en Contenedores

### Verificar en Airflow:
```bash
docker exec airflow-apiserver env | grep MODEL
```

### Verificar en FastAPI:
```bash
docker exec fastapi env | grep MODEL
```

## 💡 Tips

1. **Usar valores por defecto:** Nota que todas las variables tienen valores por defecto en `docker-compose.yaml` con el operador `:-`
   ```yaml
   MODEL_NAME_INSTANCE: ${MODEL_NAME_INSTANCE:-car-damage-instance-segmentation}
   ```

2. **Testing:** Puedes cambiar `MODEL_STAGE` a `Staging` para probar modelos sin afectar producción

3. **Versionado específico:** Cambia `MODEL_VERSION_INSTANCE=latest` a `MODEL_VERSION_INSTANCE=2` para usar una versión específica

4. **Múltiples experimentos:** Puedes cambiar `MLFLOW_EXPERIMENT_NAME` para separar experimentos diferentes

---

**Última actualización:** 2026-07-19
