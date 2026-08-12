# Evaluation Module - Arquitectura Modular

## Estructura

El módulo `evaluation/` implementa evaluación y comparación de modelos siguiendo SOLID con métodos ≤6 líneas.

```
evaluation/
├── __init__.py
├── config.py               # Configuración de evaluación
├── dataset_loader.py       # Carga de test dataset
├── model_loader.py         # Carga de modelos desde MLflow
├── predictor.py            # Predictores para YOLO y U-Net
├── mask_flattener.py       # Aplanamiento de máscaras instance
├── metrics_calculator.py   # Cálculo de métricas
├── visualizer.py           # Generación de visualizaciones
├── report_generator.py     # Generación de reportes
└── hypothesis_validator.py # Validación de hipótesis
```

## Módulos

### config.py
Configuración centralizada para evaluación.

**Clase:**
- `EvaluationConfig` - Paths, model names, stage, results path
  - 5 clases (label space compartido): Background, Dent, Scratch, No Damage, Severe
  - `NUM_CLASSES = 5`; la clase No Damage se trata como fondo en las métricas binarias

### dataset_loader.py
Carga de dataset de test con validación de disponibilidad.

**Clase:**
- `TestDatasetLoader` - Carga imágenes y máscaras de test
  - `load_test_images()` - Encuentra imágenes comunes entre datasets
  - `load_ground_truth_masks()` - Carga máscaras ground truth

**Features:**
- Valida que ambos datasets tengan las mismas imágenes
- Retorna diccionarios {image_name: path}

### model_loader.py
Carga de modelos desde MLflow Model Registry.

**Clases:**
- `ModelLoader` - Base loader para verificar disponibilidad
- `YOLOModelLoader` - Carga modelos YOLO desde MLflow
  - Descarga `best.pt` del run
  - Crea instancia de YOLO
- `UNetModelLoader` - Carga modelos U-Net desde MLflow
  - Usa `mlflow.pytorch.load_model()`
  - Configura device (CUDA/CPU)
  - Pone modelo en modo eval

### predictor.py
Predictores para hacer inferencia en test set.

**Clases:**
- `YOLOPredictor` - Predicciones con YOLO
  - Extrae máscaras, boxes, confidences
  - Usa tqdm para progress bar
- `UNetPredictor` - Predicciones con U-Net
  - Preprocessing con Albumentations
  - Postprocessing con argmax
  - Resize a tamaño original

**Métodos ≤6 líneas:** Separación en load, preprocess, predict, postprocess.

### mask_flattener.py
Aplanamiento de máscaras instance a formato semantic.

**Clase:**
- `MaskFlattener` - Aplana múltiples instancias en una máscara
  - Estrategia 'or': OR lógico (cualquier pixel activo)
  - Estrategia 'confidence': Prioriza alta confianza
  - Estrategia 'class_aware': Multiclass preservando la clase (usada en el DAG)
  - Estrategia 'last': Última instancia gana

**Uso (estrategia del DAG):**
```python
flattener = MaskFlattener(
    strategy='class_aware',
    class_map={0: 1, 1: 2, 2: 3, 3: 4},  # YOLO class -> semantic value
)
flattened = flattener.flatten_predictions(predictions)
```

### metrics_calculator.py
Cálculo de métricas de segmentación.

**Clases:**
- `MetricsCalculator` - Métricas por imagen y agregadas
  - IoU global (binario daño vs no-daño)
  - IoU per class (micro-promedio)
  - Binary precision/recall/F1 (Opción B, máscara daño vs no-daño)
  - Precision/Recall/F1 (macro, 5 clases, sklearn)
  - Pixel accuracy, area error
- `ComparisonCalculator` - Comparación entre modelos
  - Diferencia absoluta
  - Diferencia relativa (%)
  - Per-class IoU con delta por clase

**Convenciones importantes:**
- Una imagen sin daño (unión == 0) puntúa IoU = 1.0, no 0.0.
- Una clase ausente puntúa IoU = 1.0 (per-image y agregada).
- Las máscaras se sanitizan (NaN→0, valores fuera de rango→clip) antes
  de calcular métricas.

**Métricas:**
- `mean_iou` - IoU binario promedio
- `mean_binary_*` - Precision/Recall/F1 sobre la máscara binaria de daño
- `mean_precision` - Precisión macro (5 clases) promedio
- `mean_recall` - Recall macro (5 clases) promedio
- `mean_f1_score` - F1-Score macro (5 clases) promedio
- `per_class_iou` - IoU micro-promedio por clase + n_images_with_class

### visualizer.py
Generación de visualizaciones de comparación.

**Clase:**
- `ComparisonVisualizer` - Gráficos de comparación
  - `generate_metrics_comparison()` - Bar chart de métricas
  - `generate_iou_bar_chart()` - Comparación de IoU
  - `generate_per_class_iou_chart()` - Bar chart grouped por clase
  - `generate_sample_predictions()` - Grid de predicciones

**Visualizaciones:**
- Comparación lado a lado: Original | GT | Instance | Semantic
- Bar charts con valores de métricas
- Grid de 5 muestras con IoU

### report_generator.py
Generación de reporte en Markdown y JSON.

**Clase:**
- `ComparisonReport` - Reporte estructurado
  - Executive summary
  - Tabla de métricas detalladas
  - Validación de hipótesis
  - Conclusión

**Outputs:**
- `comparison_report.md` - Reporte en markdown
- `comparison_data.json` - Datos en JSON

### hypothesis_validator.py
Validación estadística de la hipótesis del proyecto.

**Clase:**
- `HypothesisValidator` - Valida o refuta hipótesis
  - Compara IoU de ambos modelos sobre el **subset común** de imágenes
  - Paired t-test para significancia estadística (p < 0.05)
  - Bootstrap CI (95%, seed=2026, 1000 resamples) del IoU y la diferencia
  - La hipótesis se valida solo si IoU_instance > IoU_semantic, p < 0.05
    y el CI de la diferencia no contiene 0
  - Genera evidencia y conclusión

**Hipótesis:**
"Los modelos de instance segmentation aprenden contornos más precisos, resultando en estimaciones de área más exactas incluso al aplanar instancias."

## Uso en DAG

### Flujo del DAG

```
check_models → load_test → [predict_instance, predict_semantic]
                                     ↓                ↓
                            flatten_masks           │
                                     └──────┬────────┘
                                            ↓
                                    calculate_metrics
                                            ↓
                                  validate_hypothesis
                                            ↓
                                 generate_visualizations
                                            ↓
                               generate_comparison_report
                                            ↓
                                 log_comparison_to_mlflow
```

### Ejemplo de Uso

```python
from evaluation.config import EvaluationConfig
from evaluation.model_loader import YOLOModelLoader, UNetModelLoader
from evaluation.dataset_loader import TestDatasetLoader
from evaluation.predictor import YOLOPredictor, UNetPredictor
from evaluation.mask_flattener import MaskFlattener
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.hypothesis_validator import HypothesisValidator

# 1. Cargar modelos
yolo_loader = YOLOModelLoader(MLFLOW_URI)
unet_loader = UNetModelLoader(MLFLOW_URI)
yolo_model = yolo_loader.load_yolo_model(MODEL_NAME, 'Production')
unet_model, device = unet_loader.load_unet_model(MODEL_NAME, 'Production')

# 2. Cargar test dataset
loader = TestDatasetLoader(INSTANCE_PATH, SEMANTIC_PATH)
test_images = loader.load_test_images()
gt_masks = loader.load_ground_truth_masks(list(test_images.keys()))

# 3. Hacer predicciones
yolo_predictor = YOLOPredictor(yolo_model)
unet_predictor = UNetPredictor(unet_model, device, (640, 640))
yolo_preds = yolo_predictor.predict(test_images)
unet_preds = unet_predictor.predict(test_images)

# 4. Aplanar máscaras instance (class_aware, label space 0-4 compartido)
flattener = MaskFlattener(
    strategy='class_aware',
    class_map={0: 1, 1: 2, 2: 3, 3: 4},
)
yolo_flat = flattener.flatten_predictions(yolo_preds)

# 5. Calcular métricas
calculator = MetricsCalculator(num_classes=5)
metrics_yolo = calculator.calculate_all_metrics(yolo_flat, gt_masks)
metrics_unet = calculator.calculate_all_metrics(unet_preds, gt_masks)

# 6. Validar hipótesis
validator = HypothesisValidator()
result = validator.validate(metrics_yolo, metrics_unet)
print(f"Hypothesis validated: {result['validated']}")
```

## Principios SOLID Aplicados

### Single Responsibility (SRP)
- Cada clase tiene una única responsabilidad
- `Predictor` solo hace predicciones
- `MetricsCalculator` solo calcula métricas
- `Visualizer` solo genera gráficos

### Open/Closed (OCP)
- `ModelLoader` es extensible para nuevos modelos
- `MaskFlattener` soporta múltiples estrategias

### Liskov Substitution (LSP)
- `YOLOModelLoader` y `UNetModelLoader` extienden `ModelLoader`
- Pueden usarse intercambiablemente

### Interface Segregation (ISP)
- Interfaces pequeñas y específicas
- Cada predictor tiene su propia interfaz

### Dependency Inversion (DIP)
- Configuración inyectada
- No hay hardcoding

## Métodos Cortos

Todos los métodos tienen **≤6 líneas** (excluyendo docstrings).

Ejemplos:
```python
def _flatten_or(self, masks):
    """Flatten using OR logic."""
    if len(masks) == 0:
        return None
    result = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        result = np.logical_or(result, mask > 0.5)
    return result.astype(np.uint8)

def _calculate_iou(self, pred, gt):
    """Calculate global IoU on the binary damage mask."""
    pred_bin = self._damage_mask(pred)
    gt_bin = self._damage_mask(gt)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    # No-damage images score 1.0 (a correct miss, not a spurious 0).
    return 1.0 if union == 0 else float(intersection / union)
```

## Outputs del DAG

### Visualizaciones
- `metrics_comparison.png` - Bar chart de todas las métricas
- `iou_comparison.png` - Comparación específica de IoU
- `per_class_iou.png` - Bar chart grouped de IoU por clase
- `sample_predictions.png` - Grid 4x5 con predicciones

### Reportes
- `comparison_report.md` - Reporte completo en markdown
- `comparison_data.json` - Todos los datos en JSON

### MLflow
- Métricas: `instance_*`, `semantic_*`, `diff_*`, per-class
  `*_iou_class_*`, `p_value`, `ci_iou_instance_*`, `ci_iou_semantic_*`,
  `ci_diff_*`
- Parámetros: run IDs, versiones de modelos, `hypothesis_validated`,
  `n_images_common`
- Artifacts: visualizaciones, reportes, datos

## Próximos Pasos

1. Entrenar ambos modelos (instance + semantic)
2. Promover modelos a Production en MLflow
3. Ejecutar DAG de evaluación
4. Analizar reporte y visualizaciones
5. Validar o refutar hipótesis del proyecto
6. Ajustar modelos si es necesario
