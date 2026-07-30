# Training Module - Arquitectura Modular

## Estructura

El módulo `training/` implementa una arquitectura modular siguiendo principios SOLID con métodos ≤6 líneas.

```
training/
├── __init__.py
├── config.py              # Configuración centralizada
├── validators.py          # Validadores de datos
├── environment.py         # Setup de entorno
├── mlflow_manager.py      # Gestión de MLflow
├── datasets.py            # Datasets para semantic segmentation
├── dataloader_factory.py  # Factory para DataLoaders
├── yolo_trainer.py        # Trainer para YOLOv8-seg
├── unet_trainer.py        # Trainer para U-Net
└── model_registry.py      # Registro de modelos en MLflow
```

## Módulos

### config.py
Configuración centralizada que obtiene valores de variables de entorno.

**Clases:**
- `TrainingConfig` - Configuración general (paths, MLflow URI, model names)
- `YOLOConfig` - Hiperparámetros para YOLOv8 (epochs, batch_size, etc.)
- `UNetConfig` - Hiperparámetros para U-Net (encoder, learning_rate, etc.)

### validators.py
Validadores de estructura de datos con Single Responsibility Principle.

**Clases:**
- `InstanceDataValidator` - Valida datos instance/ (images + labels + data.yaml)
- `SemanticDataValidator` - Valida datos semantic/ (images + masks)

**Métodos ≤6 líneas:** Cada validación está separada en métodos pequeños.

### environment.py
Setup de entorno de entrenamiento con herencia para diferentes modelos.

**Clases:**
- `EnvironmentSetup` - Base class para setup de entorno
- `YOLOEnvironment` - Setup específico para YOLO (verifica ultralytics)
- `UNetEnvironment` - Setup específico para U-Net (verifica segmentation_models_pytorch)

### mlflow_manager.py
Manager para operaciones de MLflow sin lógica de negocio.

**Clase:**
- `MLflowManager` - Encapsula todas las operaciones de MLflow
  - `start_run()` - Inicia MLflow run con timestamp
  - `log_params()` - Log de hiperparámetros
  - `log_metric()` - Log de métrica individual
  - `log_metrics()` - Log de múltiples métricas
  - `log_artifact()` - Log de artefactos
  - `log_model()` - Log de modelo PyTorch

### datasets.py
Datasets modulares para semantic segmentation.

**Clases:**
- `SemanticSegmentationDataset` - Dataset con transformaciones
- `TransformFactory` - Factory para crear transforms de train/val

### dataloader_factory.py
Factory pattern para creación de DataLoaders.

**Clase:**
- `DataLoaderFactory` - Crea train/valid/test loaders con transformaciones correctas

### yolo_trainer.py
Trainer modular para YOLOv8-seg con separación de responsabilidades.

**Clases:**
- `YOLOTrainer` - Entrenamiento de YOLOv8
  - `train()` - Pipeline completo de entrenamiento
  - `log_artifacts()` - Log de modelo y plots
- `YOLOValidator` - Validación de modelos YOLOv8
  - `validate()` - Validación en valid set
  - Extrae métricas mAP50, mAP50-95 para boxes y masks

### unet_trainer.py
Trainer modular para U-Net con training loop optimizado.

**Clase:**
- `UNetTrainer` - Entrenamiento de U-Net
  - `train()` - Pipeline completo de entrenamiento
  - `_train_epoch()` - Entrenamiento de una época
  - `_validate_epoch()` - Validación de una época
  - `save_model()` - Guarda modelo en MLflow

**Features:**
- Early stopping con patience configurable
- ReduceLROnPlateau scheduler
- DiceLoss para segmentación multiclase
- Progress bars con tqdm

### model_registry.py
Registro de modelos en MLflow Model Registry.

**Clases:**
- `ModelRegistry` - Registro genérico de modelos
- `YOLOModelRegistry` - Registro específico para YOLO (maneja path `best.pt`)

## Uso en DAGs

### Instance Segmentation DAG

```python
from training.config import TrainingConfig, YOLOConfig
from training.validators import InstanceDataValidator
from training.environment import YOLOEnvironment
from training.mlflow_manager import MLflowManager
from training.yolo_trainer import YOLOTrainer
from training.model_registry import YOLOModelRegistry

# Validar datos
validator = InstanceDataValidator(DATA_PATH)
validator.validate()

# Setup entorno
env = YOLOEnvironment(MLFLOW_URI, EXPERIMENT_NAME)
env.setup()

# Entrenar
mlflow_mgr = MLflowManager(MLFLOW_URI, EXPERIMENT_NAME)
with mlflow_mgr.start_run("yolov8-seg-instance"):
    trainer = YOLOTrainer(YOLOConfig, mlflow_mgr)
    results = trainer.train(DATA_YAML)
    trainer.log_artifacts()

# Registrar
registry = YOLOModelRegistry(MLFLOW_URI)
registry.register_yolo(run_id, MODEL_NAME, YOLOConfig.EPOCHS)
```

### Semantic Segmentation DAG

```python
from training.config import TrainingConfig, UNetConfig
from training.validators import SemanticDataValidator
from training.environment import UNetEnvironment
from training.dataloader_factory import DataLoaderFactory
from training.unet_trainer import UNetTrainer
from training.model_registry import ModelRegistry

# Validar datos
validator = SemanticDataValidator(DATA_PATH)
validator.validate()

# Setup entorno
env = UNetEnvironment(MLFLOW_URI, EXPERIMENT_NAME)
env.setup()

# Crear DataLoaders
factory = DataLoaderFactory(DATA_PATH, UNetConfig.IMG_SIZE, UNetConfig.BATCH_SIZE)
train_loader, valid_loader, _ = factory.create_loaders()

# Entrenar
mlflow_mgr = MLflowManager(MLFLOW_URI, EXPERIMENT_NAME)
with mlflow_mgr.start_run("unet"):
    trainer = UNetTrainer(UNetConfig, mlflow_mgr)
    results = trainer.train(train_loader, valid_loader)
    trainer.save_model()

# Registrar
registry = ModelRegistry(MLFLOW_URI)
registry.register_model(run_id, MODEL_NAME, description)
```

## Principios SOLID Aplicados

### Single Responsibility Principle (SRP)
- Cada clase tiene una única responsabilidad
- `MLflowManager` solo maneja MLflow
- `YOLOTrainer` solo entrena YOLO
- `Validators` solo validan datos

### Open/Closed Principle (OCP)
- `EnvironmentSetup` es base class extensible
- `YOLOEnvironment` y `UNetEnvironment` extienden sin modificar base

### Liskov Substitution Principle (LSP)
- Subclases pueden reemplazar a sus clases base
- `YOLOEnvironment` puede usarse donde se espera `EnvironmentSetup`

### Interface Segregation Principle (ISP)
- Interfaces pequeñas y específicas
- `YOLOTrainer` y `UNetTrainer` tienen interfaces diferentes según necesidades

### Dependency Inversion Principle (DIP)
- Dependencias inyectadas (MLflowManager, Config)
- No hay hardcoding de configuraciones

## Métodos Cortos

Todos los métodos tienen **≤6 líneas de código** (excluyendo docstrings y líneas en blanco).

Ejemplos:
```python
def _check_cuda(self):
    """Check CUDA availability."""
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        self._print_gpu_info()
    else:
        self._warn_cpu()
    return {"available": available}

def _create_model(self):
    """Create U-Net model."""
    self.model = smp.Unet(
        encoder_name=self.config.ENCODER,
        encoder_weights=self.config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=self.config.NUM_CLASSES
    )
```

## Testing

Para testear los módulos individualmente:

```python
# Test validator
from training.validators import InstanceDataValidator
validator = InstanceDataValidator("/path/to/data")
validator.validate()

# Test environment
from training.environment import YOLOEnvironment
env = YOLOEnvironment("http://mlflow:5000", "experiment")
env.setup()

# Test MLflow manager
from training.mlflow_manager import MLflowManager
mgr = MLflowManager("http://mlflow:5000", "experiment")
mgr.log_params({"batch_size": 16})
```

## Próximos Pasos

1. Ejecutar DAGs refactorizados en Airflow
2. Verificar que los modelos se entrenan correctamente
3. Validar que se registran en MLflow Model Registry
4. Promover modelos a Production
5. Implementar carga de modelos en FastAPI
