# 🚀 Roadmap para Producción

## 📊 Estado Actual del Proyecto

### ✅ Lo que YA TIENES Funcionando

#### 1. Infraestructura Base
- ✅ Docker Compose con todos los servicios
- ✅ Airflow (orquestación)
- ✅ MLflow (tracking básico)
- ✅ MinIO (storage S3-compatible)
- ✅ PostgreSQL (metadata stores)
- ✅ FastAPI (placeholder básico)
- ✅ Networking entre servicios

#### 2. Preparación de Datos
- ✅ DAG de descarga y preparación de datos
- ✅ Conversión a ambos formatos (instance + semantic)
- ✅ Validación de datos

#### 3. Documentación
- ✅ README completo
- ✅ METODOLOGIA.md
- ✅ Estructura de DAGs documentada

---

## ❌ Lo que FALTA para Producción

### 📦 FASE 1: MLflow - Gestión de Modelos

#### 1.1 Tracking de Experimentos ⏳
**Estado:** Parcialmente configurado (servidor levantado, pero sin uso)

**Falta implementar:**
- [ ] Logging en DAGs de entrenamiento (params, metrics, artifacts)
- [ ] Registro de datasets usados
- [ ] Versionado de modelos
- [ ] Tags y anotaciones en experimentos

**Código a agregar en training DAGs:**
```python
import mlflow
import mlflow.pytorch
import mlflow.sklearn

# Configurar tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("car-damage-segmentation")

with mlflow.start_run(run_name="yolov8s-seg-v1"):
    # Log params
    mlflow.log_param("model", "yolov8s-seg")
    mlflow.log_param("epochs", 100)
    mlflow.log_param("batch_size", 16)
    
    # Log metrics
    mlflow.log_metric("train_loss", loss)
    mlflow.log_metric("val_mAP", map_score)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("training_curves.png")
```

#### 1.2 Model Registry ⏳
**Estado:** No implementado

**Falta implementar:**
- [ ] Registro de modelos en Model Registry
- [ ] Transiciones de stage (None → Staging → Production)
- [ ] Versionado de modelos
- [ ] Aliases (best, champion, etc.)

**Código a implementar:**
```python
# Registrar modelo
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "car-damage-instance-segmentation")

# Promover a Production
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="car-damage-instance-segmentation",
    version=1,
    stage="Production"
)
```

#### 1.3 Model Serving desde MLflow ⏳
**Estado:** No implementado

**Opciones:**

**Opción A - MLflow Model Serving (Simple):**
```bash
# Servir modelo directamente con MLflow
mlflow models serve -m "models:/car-damage-instance-segmentation/Production" -p 5001
```

**Opción B - Custom FastAPI (Recomendado para producción):**
- Ver FASE 2 abajo

---

### 🔌 FASE 2: API REST - FastAPI

#### 2.1 Estructura de la API ⏳
**Estado:** Solo placeholder básico

**Falta implementar:**

```
dockerfiles/fastapi/
├── app.py                      # ❌ Solo placeholder
├── requirements.txt            # ⚠️ Faltan dependencias de ML
├── models/                     # ❌ No existe
│   ├── __init__.py
│   ├── instance_segmentation.py
│   └── semantic_segmentation.py
├── schemas/                    # ❌ No existe
│   ├── __init__.py
│   ├── request.py
│   └── response.py
├── services/                   # ❌ No existe
│   ├── __init__.py
│   ├── model_loader.py
│   └── prediction.py
└── utils/                      # ❌ No existe
    ├── __init__.py
    ├── image_processing.py
    └── mask_processing.py
```

#### 2.2 Endpoints Necesarios ⏳

**Endpoints mínimos:**

```python
# 1. Health check
@app.get("/health")
async def health_check():
    """Verificar que API y modelos estén funcionando"""
    
# 2. Predicción con modelo instance
@app.post("/predict/instance")
async def predict_instance(file: UploadFile):
    """
    Predict usando YOLOv8-seg
    Input: Imagen
    Output: Máscaras por instancia + bounding boxes + scores
    """

# 3. Predicción con modelo semantic
@app.post("/predict/semantic")
async def predict_semantic(file: UploadFile):
    """
    Predict usando U-Net
    Input: Imagen
    Output: Máscara semántica (clasificación pixel-wise)
    """

# 4. Comparación (ambos modelos)
@app.post("/predict/compare")
async def predict_compare(file: UploadFile):
    """
    Predict con ambos modelos y comparar
    Input: Imagen
    Output: Resultados de ambos + métricas comparativas
    """

# 5. Información de modelos
@app.get("/models/info")
async def models_info():
    """
    Info sobre modelos cargados:
    - Versiones
    - Métricas de entrenamiento
    - Stage (Production/Staging)
    """

# 6. Batch prediction
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile]):
    """Predicciones en batch (múltiples imágenes)"""
```

#### 2.3 Integración con MLflow ⏳
**Estado:** No existe

**Falta implementar:**
```python
# services/model_loader.py
import mlflow

class ModelLoader:
    def __init__(self):
        mlflow.set_tracking_uri("http://mlflow:5000")
        
    def load_instance_model(self):
        """Cargar modelo instance desde MLflow Model Registry"""
        model_uri = "models:/car-damage-instance-segmentation/Production"
        return mlflow.pyfunc.load_model(model_uri)
    
    def load_semantic_model(self):
        """Cargar modelo semantic desde MLflow Model Registry"""
        model_uri = "models:/car-damage-semantic-segmentation/Production"
        return mlflow.pytorch.load_model(model_uri)
```

#### 2.4 Schemas Pydantic ⏳
**Estado:** No existen

**Falta implementar:**
```python
# schemas/response.py
from pydantic import BaseModel
from typing import List, Dict

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

class InstancePrediction(BaseModel):
    boxes: List[BoundingBox]
    masks: List[List[List[int]]]  # Binary masks
    num_instances: int
    processing_time_ms: float

class SemanticPrediction(BaseModel):
    mask: List[List[int]]  # Class IDs per pixel
    class_distribution: Dict[str, float]  # % por clase
    processing_time_ms: float

class ComparePrediction(BaseModel):
    instance: InstancePrediction
    semantic: SemanticPrediction
    comparison_metrics: Dict[str, float]
```

#### 2.5 Dependencias Faltantes ⏳
**Estado:** requirements.txt muy básico

**Agregar a `dockerfiles/fastapi/requirements.txt`:**
```txt
# Actual
fastapi>=0.116
uvicorn>=0.27

# FALTA AGREGAR:
# ML Libraries
mlflow>=2.10.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
segmentation-models-pytorch>=0.3.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0

# AWS/S3
boto3>=1.28.0

# Image processing
python-multipart>=0.0.6

# Monitoring (opcional)
prometheus-fastapi-instrumentator>=6.1.0
```

---

### 🎯 FASE 3: Despliegue de Modelos en Producción

#### 3.1 Estrategia de Despliegue ⏳

**Opción A - Cargar modelos en FastAPI al inicio:**
```python
# app.py
from fastapi import FastAPI
from services.model_loader import ModelLoader

app = FastAPI()
loader = ModelLoader()

# Cargar modelos al iniciar
@app.on_event("startup")
async def load_models():
    global instance_model, semantic_model
    instance_model = loader.load_instance_model()
    semantic_model = loader.load_semantic_model()
```

**Opción B - Lazy loading (cargar on-demand):**
```python
# Cargar solo cuando se necesita
@app.post("/predict/instance")
async def predict_instance(file: UploadFile):
    model = get_or_load_instance_model()  # Cache
    ...
```

**Opción C - Model serving separado (Escalable):**
- Servicio dedicado para modelo instance
- Servicio dedicado para modelo semantic
- API Gateway (FastAPI) enruta requests

#### 3.2 Optimizaciones de Inferencia ⏳

**Falta implementar:**
- [ ] GPU support en containers
- [ ] Batch inference
- [ ] Model caching
- [ ] Image preprocessing pipeline
- [ ] Response compression

#### 3.3 Actualización de docker-compose.yaml ⏳

**Agregar configuración de GPU (opcional):**
```yaml
services:
  fastapi:
    # ... configuración actual ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

---

### 📊 FASE 4: Monitoreo y Observabilidad

#### 4.1 Logging ⏳
**Estado:** Básico (logs de Docker)

**Falta implementar:**
- [ ] Structured logging en FastAPI
- [ ] Log aggregation
- [ ] Request/response logging
- [ ] Error tracking

**Código a implementar:**
```python
import logging
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.post("/predict/instance")
async def predict_instance(file: UploadFile):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Request {request_id} started", extra={
        "request_id": request_id,
        "filename": file.filename,
        "content_type": file.content_type
    })
    
    try:
        result = await process_prediction(file)
        logger.info(f"Request {request_id} completed", extra={
            "request_id": request_id,
            "duration_ms": (time.time() - start_time) * 1000
        })
        return result
    except Exception as e:
        logger.error(f"Request {request_id} failed", extra={
            "request_id": request_id,
            "error": str(e)
        })
        raise
```

#### 4.2 Métricas ⏳
**Estado:** No implementado

**Falta implementar:**
- [ ] Prometheus metrics
- [ ] Request rate, latency, errors
- [ ] Model performance metrics
- [ ] Resource usage (CPU, GPU, RAM)

**Código a implementar:**
```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter(
    'predictions_total', 
    'Total predictions',
    ['model_type', 'status']
)

prediction_latency = Histogram(
    'prediction_duration_seconds',
    'Prediction latency',
    ['model_type']
)
```

#### 4.3 Health Checks ⏳
**Estado:** Básico en docker-compose

**Falta implementar:**
```python
@app.get("/health/ready")
async def readiness():
    """Verifica que API está lista (modelos cargados)"""
    try:
        if instance_model is None or semantic_model is None:
            return {"status": "not_ready", "reason": "models_not_loaded"}
        return {"status": "ready"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}

@app.get("/health/live")
async def liveness():
    """Verifica que API está viva"""
    return {"status": "alive"}
```

---

### 🔐 FASE 5: Seguridad y Autenticación

#### 5.1 Autenticación API ⏳
**Estado:** No existe

**Falta implementar:**
- [ ] API Keys
- [ ] JWT tokens
- [ ] Rate limiting

**Código a implementar:**
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict/instance")
async def predict_instance(
    file: UploadFile,
    api_key: str = Security(verify_api_key)
):
    ...
```

#### 5.2 Rate Limiting ⏳
**Estado:** No existe

**Falta implementar:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict/instance")
@limiter.limit("10/minute")
async def predict_instance(...):
    ...
```

---

### 📈 FASE 6: CI/CD y Automatización

#### 6.1 Testing ⏳
**Estado:** No existe

**Falta crear:**
- [ ] Unit tests para API
- [ ] Integration tests
- [ ] Model validation tests
- [ ] Performance tests

**Estructura a crear:**
```
tests/
├── __init__.py
├── test_api.py
├── test_models.py
├── test_predictions.py
└── test_integration.py
```

#### 6.2 CI/CD Pipeline ⏳
**Estado:** No existe

**Falta implementar:**
- [ ] GitHub Actions / GitLab CI
- [ ] Automated testing
- [ ] Docker image building
- [ ] Deployment automation

---

## 📋 Checklist de Producción

### Crítico (Debe estar antes de producción)
- [ ] MLflow Model Registry funcionando
- [ ] API con endpoints de predicción funcionales
- [ ] Modelos cargables desde MLflow
- [ ] Health checks implementados
- [ ] Error handling robusto
- [ ] Logging estructurado

### Importante (Debería estar)
- [ ] Autenticación API
- [ ] Rate limiting
- [ ] Métricas de monitoreo
- [ ] Tests unitarios
- [ ] Tests de integración

### Deseable (Nice to have)
- [ ] CI/CD pipeline
- [ ] GPU optimization
- [ ] Batch prediction
- [ ] Model A/B testing
- [ ] Distributed tracing

---

## 🎯 Plan de Implementación Sugerido

### Sprint 1: MLflow Integration (1-2 semanas)
1. Implementar logging en training DAGs
2. Configurar Model Registry
3. Registrar primeros modelos
4. Validar artifacts storage en MinIO

### Sprint 2: API Básica (1-2 semanas)
1. Crear estructura de carpetas FastAPI
2. Implementar schemas Pydantic
3. Implementar endpoint `/predict/instance`
4. Implementar endpoint `/predict/semantic`
5. Agregar dependencias necesarias

### Sprint 3: Integración MLflow-API (1 semana)
1. Model loader desde MLflow
2. Cargar modelos en FastAPI startup
3. Testing de predicciones end-to-end

### Sprint 4: Production-Ready (1-2 semanas)
1. Health checks
2. Logging estructurado
3. Error handling
4. Autenticación básica
5. Documentación de API (Swagger)

### Sprint 5: Monitoreo y Optimización (1 semana)
1. Métricas de Prometheus
2. Optimizaciones de inferencia
3. Tests de carga
4. Ajustes de performance

---

## 📚 Recursos y Referencias

### MLflow
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)

### FastAPI
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [FastAPI ML Serving](https://fastapi.tiangolo.com/advanced/async-sql-databases/)

### Deployment
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [ML Model Deployment](https://ml-ops.org/content/model-serving)

---

**Última actualización:** 2026-07-19  
**Versión:** 1.0
