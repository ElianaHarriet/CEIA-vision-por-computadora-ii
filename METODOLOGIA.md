# 📐 Metodología del Proyecto: Comparación Instance vs Semantic Segmentation

## 🔑 Concepto Clave del Proyecto

**Pregunta Central:**  
¿Qué arquitectura de segmentación (instance vs semantic) es más precisa para detectar y medir áreas de daño en vehículos?

**Enfoque Metodológico:**
- **UN solo dataset** (Car Damages - 4 clases)
- **DOS formatos diferentes** (instance + semantic)
- **DOS modelos diferentes** (YOLOv8-seg vs U-Net)
- **Comparación justa:** Solo cambia la arquitectura, los datos son idénticos

**Por qué funciona:**  
El dataset semántico de Roboflow contiene anotaciones con polígonos por objeto. Estos polígonos pueden convertirse a:
1. **Formato instance:** Archivos .txt con coordenadas (YOLOv8)
2. **Formato semantic:** Máscaras PNG con IDs de clase (U-Net)

**Resultado:** Ambos modelos entrenan con las mismas imágenes y anotaciones, permitiendo una comparación científicamente válida.

---

## 🎯 Objetivo Principal

Comparar el desempeño de dos arquitecturas de segmentación (instance vs semantic) para detectar y clasificar daños en vehículos, utilizando el **mismo dataset** convertido a diferentes formatos.

**Hipótesis a Validar:**
> Los modelos de instance segmentation aprenden contornos más precisos debido a su enfoque en detectar objetos individuales. Por lo tanto, incluso al colapsar las instancias en una máscara única, deberían proporcionar estimaciones de área de daño más exactas que los modelos de semantic segmentation.

## 📊 Dataset Utilizado

**Dataset Único: Car Damages (Dataset A)**

- **Fuente:** https://universe.roboflow.com/project-p5nyc/car-damages-v3gyz
- **Tipo original:** Segmentación Semántica
- **Clases:** 4 categorías
  - Minor Damage (Dent) - Abolladuras leves
  - Minor Damage (Scratch) - Rayones superficiales
  - No Damage - Sin daños visibles
  - Severe Damage - Daños severos/graves

**Estadísticas:**
- Total: 2,324 imágenes
- Train: 1,974 imágenes
- Valid: 231 imágenes
- Test: 119 imágenes

## 🔄 Preparación de Datos

### ¿Por Qué un Dataset "Semántico" Funciona para Instance Segmentation?

**Concepto clave:** El "tipo" de dataset (semántico vs instancias) se refiere más a cómo está **etiquetado en Roboflow** que a las anotaciones subyacentes.

**Dataset A en Roboflow:**
- Categorizado como: "Semantic Segmentation"
- Anotaciones: Polígonos que delimitan objetos dañados
- Formato de descarga: COCO Segmentation (JSON con polígonos)

**La conversión es posible porque:**
- Las anotaciones COCO contienen polígonos por objeto individual
- Estos polígonos pueden representarse de dos formas:
  1. **Lista de coordenadas** → Formato instance (YOLOv8 .txt)
  2. **Máscara rasterizada** → Formato semantic (PNG con IDs de clase)

### Proceso de Conversión Automática

El DAG `data_preparation_semantic` descarga el dataset en formato COCO y lo convierte a **DOS formatos**:

### 1. Formato Instance Segmentation (YOLOv8)
```
car-damages-ready/instance/
├── train/
│   ├── images/
│   └── labels/    # Archivos .txt con polígonos YOLOv8
├── valid/
├── test/
└── data.yaml
```

### 2. Formato Semantic Segmentation (Máscaras PNG)
```
car-damages-ready/semantic/
├── train/
│   ├── images/
│   └── masks/     # Máscaras PNG (cada pixel = ID de clase)
├── valid/
└── test/
```

**Ambos formatos contienen las MISMAS imágenes y anotaciones**, solo cambia la representación:
- Instance: Coordenadas de polígonos en archivos .txt
- Semantic: Máscaras con pixeles coloreados según la clase

**Ventaja clave:** Esto garantiza que ambos modelos entrenen con exactamente los mismos datos, permitiendo una comparación justa donde la única variable es la arquitectura del modelo.

## 🤖 Modelos a Entrenar

### Modelo 1: Instance Segmentation
- **Arquitectura:** YOLOv8-seg (YOLOv8n-seg, YOLOv8s-seg, o YOLOv8x-seg)
- **Datos de entrenamiento:** `car-damages-ready/instance/`
- **Formato de entrada:** Imágenes + archivos .txt con polígonos en formato YOLOv8
- **Formato de salida:** Máscaras individuales por cada instancia detectada
- **Características:**
  - Detecta y segmenta cada objeto de daño por separado
  - Puede identificar múltiples instancias de la misma clase
  - Aprende a distinguir contornos precisos de objetos individuales
  - Genera bounding boxes + máscaras de segmentación

**Proceso de predicción:**
```
Imagen → YOLOv8-seg → [Máscara_rayón_1, Máscara_rayón_2, Máscara_abolladura_1, ...]
```

### Modelo 2: Semantic Segmentation
- **Arquitectura:** U-Net, DeepLab v3+, PSPNet, o similar
- **Datos de entrenamiento:** `car-damages-ready/semantic/`
- **Formato de entrada:** Imágenes + máscaras PNG (cada pixel tiene un ID de clase: 0, 1, 2, 3)
- **Formato de salida:** Máscara única con clasificación pixel-wise
- **Características:**
  - Clasifica cada pixel en una de las 4 clases
  - No distingue entre instancias individuales de la misma clase
  - Genera una máscara continua por clase
  - Enfoque: ¿A qué clase pertenece este pixel?

**Proceso de predicción:**
```
Imagen → U-Net → Máscara_única (cada pixel clasificado: 0=Dent, 1=Scratch, 2=No_Damage, 3=Severe)
```

## 📏 Metodología de Comparación

### Paso 1: Entrenamiento
- Entrenar **Modelo 1 (Instance)** con datos de `instance/`
- Entrenar **Modelo 2 (Semantic)** con datos de `semantic/`
- Configuración base:
  - Misma resolución de entrada (ej: 640x640)
  - Similar número de epochs
  - Mismos splits (train/valid/test)
  - Optimizadores comparables (ej: Adam para ambos)

### Paso 2: Inferencia en Test Set
- Evaluar ambos modelos en el **mismo conjunto de test**
- Modelo Instance genera múltiples máscaras por imagen
- Modelo Semantic genera una máscara única por imagen

```
Test Image 1 → YOLOv8-seg → [Máscara_A, Máscara_B, Máscara_C]
Test Image 1 → U-Net     → [Máscara_única]
```

### Paso 3: Post-procesamiento (Aplanamiento de Instancias)

**Objetivo:** Hacer las predicciones de instance comparables con las de semantic.

**Proceso de aplanamiento:**
```python
# Predicciones del modelo instance
mascaras_instance = [mascara_1, mascara_2, mascara_3, ...]  # Múltiples máscaras

# Aplanar: Fusionar todas las instancias en una sola máscara
mascara_aplanada = fusionar_mascaras(mascaras_instance)

# Ahora mascara_aplanada tiene el mismo formato que la predicción semantic
```

**Métodos de fusión:**
1. **OR lógico:** Si cualquier máscara tiene un pixel activo, activarlo en resultado
2. **Prioridad por confianza:** En superposiciones, usar la instancia con mayor score
3. **Última instancia gana:** Sobrescribir en caso de overlap

**Resultado:** Ambas predicciones (instance aplanada y semantic) son máscaras únicas comparables.

### Paso 4: Comparación por Estrategia

Se proponen dos estrategias de comparación:

#### Opción A: Comparación Multiclase (Recomendada)
- Mantener las **4 clases originales**
- Calcular métricas por clase individual
- Métricas:
  - **IoU (Intersection over Union)** por clase
  - **Precision, Recall, F1-Score** por clase
  - **Área de daño calculada** por clase (en pixels o cm²)
  - **Accuracy pixel-wise**

**Ventajas:**
- Análisis más detallado
- Evalúa capacidad de clasificar tipos de daño
- Identifica en qué clases cada modelo es mejor

**Desafío:**
- Resolver superposiciones en modelo instance (diferentes instancias de misma clase)

#### Opción B: Comparación Binaria (Simplificada)
- Colapsar clases a **detección binaria**:
  - `Clases 0, 1, 3` → **"Daño"** (1)
  - `Clase 2` → **"No Daño"** (0)
- Calcular métricas binarias

**Ventajas:**
- Simplifica superposiciones
- Comparación más limpia
- Enfoque en "¿detecta daño o no?"

**Desventajas:**
- Pierde granularidad de tipos de daño
- Menos información científica

### Paso 5: Cálculo de Métricas

**Métricas cuantitativas:**
```
1. IoU (Intersection over Union):
   IoU = (Área_intersección) / (Área_unión)
   
2. Precision:
   Precision = TP / (TP + FP)
   
3. Recall:
   Recall = TP / (TP + FN)
   
4. F1-Score:
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   
5. Pixel Accuracy:
   Accuracy = Pixels_correctos / Pixels_totales
   
6. Área de Daño:
   Diferencia absoluta: |Área_predicha - Área_ground_truth|
   Error relativo: |Área_predicha - Área_ground_truth| / Área_ground_truth
```

**Métricas cualitativas:**
- Visualización de predicciones vs ground truth
- Análisis de casos de éxito y fallo
- Calidad visual de contornos
- Capacidad de detectar daños pequeños

### Paso 6: Comparación Final

**Tablas de resultados por modelo:**

| Métrica | YOLOv8-seg (Instance) | U-Net (Semantic) | Diferencia |
|---------|----------------------|------------------|------------|
| IoU Global | 0.XX | 0.XX | ±X.XX |
| IoU Dent | 0.XX | 0.XX | ±X.XX |
| IoU Scratch | 0.XX | 0.XX | ±X.XX |
| IoU Severe | 0.XX | 0.XX | ±X.XX |
| Precision | 0.XX | 0.XX | ±X.XX |
| Recall | 0.XX | 0.XX | ±X.XX |
| F1-Score | 0.XX | 0.XX | ±X.XX |
| Error Área (%) | X.XX% | X.XX% | ±X.XX% |

**Análisis estadístico:**
- Test de significancia (ej: t-test) para determinar si diferencias son estadísticamente significativas
- Intervalos de confianza

## 💡 Hipótesis a Validar

**Hipótesis Principal:**
> Los modelos de instance segmentation (YOLOv8-seg) aprenden representaciones de contornos más precisas debido a su enfoque en detectar y segmentar objetos individuales. Por lo tanto, incluso al colapsar las instancias en una máscara única, deberían proporcionar estimaciones de área de daño más exactas que los modelos de semantic segmentation.

**Razonamiento:**
1. **Instance segmentation:**
   - Aprende a distinguir límites de objetos individuales
   - Cada instancia debe tener contornos bien definidos
   - La loss function penaliza predicciones que mezclan instancias

2. **Semantic segmentation:**
   - Aprende a clasificar regiones amplias
   - No tiene incentivo para separar instancias
   - Puede generar máscaras más "borrosas" en los bordes

**Hipótesis Secundarias:**
- Instance segmentation será mejor en detectar daños pequeños (rayones)
- Semantic segmentation podría ser más rápido en inferencia
- Instance segmentation tendrá mejor precisión en áreas con múltiples daños cercanos

**Métricas de Validación:**
- Si IoU_instance > IoU_semantic → Hipótesis validada
- Si Error_área_instance < Error_área_semantic → Hipótesis validada
- Análisis cualitativo de contornos

## ⚠️ Dataset B: Análisis Secundario Opcional

### ¿Qué es Dataset B?

**Dataset B:** Car Damage Detection  
**Fuente:** https://universe.roboflow.com/college-qxdrt/car-damage-detection-ha5mm  
**Características:**
- ~4,869 imágenes (más grande que Dataset A)
- 1 clase genérica: "car-damage"
- Formato nativo: Instance segmentation (YOLOv8)

### Rol en el Proyecto

**Dataset B NO se usa en el análisis principal** por estas razones:

#### Problema de Incompatibilidad de Clases
```
Dataset A:                    Dataset B:
├─ 0: Minor Damage (Dent)    └─ 0: car-damage (genérico)
├─ 1: Minor Damage (Scratch)
├─ 2: No Damage
└─ 3: Severe Damage

❌ No se pueden comparar modelos con diferentes ground truths
```

**Consecuencia:**
- IoU de "Dent" vs IoU de "car-damage" → Conceptos diferentes
- Métricas no comparables entre datasets
- Diferentes distribuciones de datos

### Posibles Usos Secundarios (Opcional)

Si el equipo tiene tiempo, Dataset B podría usarse para:

#### 1. Experimento Separado: Instance vs Semantic en Dataset B
```
Dataset B → Convertir a ambos formatos
    ↓
├→ instance/ → YOLOv8-seg (1 clase)
└→ semantic/ → U-Net (1 clase)
    ↓
Comparación: ¿Qué arquitectura es mejor para detección binaria?
```

**Resultado:** Experimento independiente, NO comparable con Dataset A.

#### 2. Validación Externa (Comparación Binaria)
```
1. Entrenar modelos con Dataset A (4 clases)
2. Colapsar predicciones a binario (daño/no-daño)
3. Evaluar en Dataset B como conjunto externo
4. Medir generalización a detección genérica
```

**Limitación:** Requiere alinear semántica de clases (4 clases → 1 clase).

#### 3. Transfer Learning
```
1. Pre-entrenar con Dataset B (~4,869 imágenes)
2. Fine-tunar con Dataset A (clases específicas)
3. Evaluar si más datos pre-entrenamiento mejoran performance
```

**Complejidad:** Requiere estrategia de transfer learning cuidadosa.

### Recomendación del Profesor

El profesor mencionó:
> "Evaluar ese que han propuesto [Dataset B], pero también uno de segmentación semántica [Dataset A]"

**Interpretación:**
- Evaluar ambos datasets, pero como **experimentos separados**
- Dataset A: Análisis principal (comparación rigurosa con 4 clases)
- Dataset B: Análisis secundario opcional (detección binaria)

### Estado Actual

**Dataset B en el proyecto:**
- ✅ DAG implementado: `data_preparation_instance_dag.py.disabled`
- 📦 Código preservado pero desactivado (extensión `.disabled`)
- 📝 Documentado para reactivación futura

**Para reactivar Dataset B:**
1. Renombrar: `data_preparation_instance_dag.py.disabled` → `data_preparation_instance_dag.py`
2. Configurar variables en `.env`:
   ```bash
   ROBOFLOW_PROJECT_INSTANCE=car-damage-detection-ha5mm-XXXXX
   ROBOFLOW_VERSION_INSTANCE=1
   ```
3. Actualizar `docker-compose.yaml` con variables de Dataset B
4. Ejecutar DAG en Airflow

### Conclusión sobre Dataset B

**Para el proyecto principal:**
- ❌ NO usar Dataset B
- ✅ Concentrarse en Dataset A (comparación rigurosa)
- ✅ Metodología científicamente válida

**Para extensiones futuras:**
- ✅ Dataset B puede explorarse como análisis secundario
- ✅ Código está listo para reactivarse cuando sea necesario
- ⚠️ No comparar directamente Dataset A vs Dataset B (incompatible)

## 📁 Estructura de Datos Esperada

```
car_damage_detection/car-damages/
├── car-damages-forked/          # Datos raw descargados de Roboflow
│   ├── train/
│   ├── valid/
│   └── test/
└── car-damages-ready/           # Datos procesados y listos
    ├── instance/                # ← Para entrenar modelo de instancias
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── valid/
    │   ├── test/
    │   └── data.yaml
    └── semantic/                # ← Para entrenar modelo semántico
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── valid/
        └── test/
```

## 🚀 Flujo de Trabajo Completo

### Fase 1: Preparación de Datos ✅
```bash
# 1. Levantar infraestructura
docker compose --profile all up

# 2. Acceder a Airflow
# URL: http://localhost:8080
# User: airflow / Pass: airflow

# 3. Ejecutar DAG: data_preparation_semantic
# - Descarga Dataset A desde Roboflow
# - Convierte a formato instance (YOLOv8)
# - Convierte a formato semantic (máscaras PNG)
# - Valida integridad de datos
```

**Resultado esperado:**
```
car_damage_detection/car-damages/car-damages-ready/
├── instance/
│   ├── train/ (1,974 imágenes)
│   ├── valid/ (231 imágenes)
│   ├── test/ (119 imágenes)
│   └── data.yaml
└── semantic/
    ├── train/ (1,974 imágenes)
    ├── valid/ (231 imágenes)
    └── test/ (119 imágenes)
```

### Fase 2: Entrenamiento Modelo Instance

**Arquitectura recomendada:** YOLOv8-seg (YOLOv8n-seg, YOLOv8s-seg, o YOLOv8x-seg)

```python
from ultralytics import YOLO

# Cargar modelo pre-entrenado
model = YOLO('yolov8s-seg.pt')

# Entrenar
results = model.train(
    data='car-damages-ready/instance/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='car_damage_instance',
    project='mlruns'
)

# Evaluar
metrics = model.val()

# Guardar modelo
model.save('models/yolov8_instance.pt')
```

**Registrar en MLflow:**
- Hiperparámetros
- Métricas de entrenamiento (loss, mAP, IoU)
- Modelo entrenado
- Ejemplos de predicciones

### Fase 3: Entrenamiento Modelo Semantic

**Arquitectura recomendada:** U-Net, DeepLab v3+, o PSPNet

```python
import segmentation_models_pytorch as smp
import mlflow

# Definir modelo
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4  # 4 clases de daño
)

# Entrenar (pseudocódigo)
trainer = SemanticSegmentationTrainer(
    model=model,
    data_path='car-damages-ready/semantic/',
    epochs=100,
    batch_size=16
)

# Entrenar con MLflow tracking
with mlflow.start_run(run_name='car_damage_semantic'):
    history = trainer.train()
    
    # Registrar métricas
    mlflow.log_metrics(history)
    
    # Guardar modelo
    mlflow.pytorch.log_model(model, "model")
```

### Fase 4: Evaluación y Comparación

```python
# 1. Cargar modelos entrenados
yolo_model = YOLO('models/yolov8_instance.pt')
unet_model = load_semantic_model('models/unet_semantic.pt')

# 2. Hacer predicciones en test set
test_images = load_test_images('car-damages-ready/instance/test/images/')

for image in test_images:
    # Predicción instance
    yolo_pred = yolo_model.predict(image)
    yolo_masks = yolo_pred.masks  # Múltiples máscaras
    
    # Aplanar máscaras de instance
    yolo_flat = flatten_masks(yolo_masks)
    
    # Predicción semantic
    unet_pred = unet_model.predict(image)
    unet_mask = unet_pred  # Una sola máscara
    
    # Comparar con ground truth
    gt_mask = load_ground_truth(image)
    
    # Calcular métricas
    iou_yolo = calculate_iou(yolo_flat, gt_mask)
    iou_unet = calculate_iou(unet_mask, gt_mask)
    
    area_error_yolo = calculate_area_error(yolo_flat, gt_mask)
    area_error_unet = calculate_area_error(unet_mask, gt_mask)

# 3. Agregar métricas globales
results = {
    'yolo_instance': aggregate_metrics(yolo_metrics),
    'unet_semantic': aggregate_metrics(unet_metrics)
}

# 4. Visualizar y reportar
generate_comparison_report(results)
```

### Fase 5: Análisis de Resultados

**Análisis cuantitativo:**
1. Tablas comparativas de métricas
2. Gráficos de distribución de IoU por clase
3. Análisis estadístico de significancia
4. Curvas Precision-Recall

**Análisis cualitativo:**
1. Visualización lado a lado: Ground Truth | Instance | Semantic
2. Identificar casos donde instance es mejor
3. Identificar casos donde semantic es mejor
4. Análisis de errores comunes

**Conclusiones:**
- ¿Se validó la hipótesis?
- ¿Qué arquitectura es mejor para este problema?
- ¿En qué escenarios cada modelo es superior?
- Recomendaciones para uso práctico

## 📚 Referencias y Recursos

### Papers Relevantes
- **YOLOv8:** [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/tasks/segment/)
- **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **DeepLab v3+:** Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (2018)
- **Instance vs Semantic:** Kirillov et al., "Panoptic Segmentation" (2019)

### Datasets
- **Dataset A (Principal):** [Car Damages - Roboflow](https://universe.roboflow.com/project-p5nyc/car-damages-v3gyz)
- **Dataset B (Opcional):** [Car Damage Detection - Roboflow](https://universe.roboflow.com/college-qxdrt/car-damage-detection-ha5mm)

### Herramientas
- **Airflow:** Orquestación de pipelines de datos
- **MLflow:** Tracking de experimentos y modelos
- **Roboflow:** Gestión y descarga de datasets
- **Docker:** Containerización de servicios

### Librerías de ML Recomendadas
```python
# Instance Segmentation
- ultralytics (YOLOv8)
- detectron2 (Mask R-CNN)

# Semantic Segmentation
- segmentation_models_pytorch (U-Net, DeepLab, PSPNet)
- mmsegmentation (múltiples arquitecturas)
- tensorflow/keras (modelos custom)

# Métricas y Evaluación
- scikit-learn
- numpy
- opencv-python
- matplotlib/seaborn (visualización)
```

### Recomendación del Profesor
> "Comparar los resultados de entrenar un modelo de segmentación por instancias con un modelo de segmentación semántica. Para segmentación semántica les recomiendo este dataset: Car Damages Dataset"

### Contacto del Equipo
- **Santiago Bartolini Rizzo** - [santiagobartolini@gmail.com](mailto:santiagobartolini@gmail.com)
- **Luis Ali** - [aliluis@gmail.com](mailto:aliluis@gmail.com)
- **Eliana Harriet** - [eharriet@fi.uba.ar](mailto:eharriet@fi.uba.ar)

---

**Última actualización:** 2026-07-19  
**Versión:** 2.0 - Metodología definitiva con Dataset A único
