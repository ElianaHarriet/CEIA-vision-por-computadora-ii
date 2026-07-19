"""
DAG para preparación de datos de detección de daños en autos.

Este DAG descarga el dataset de Roboflow y prepara los datos
en formato YOLOv8 (instance segmentation) y máscaras PNG (semantic segmentation).
"""

from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import shutil

from airflow import DAG
from airflow.operators.python import PythonOperator

from roboflow import Roboflow
from PIL import Image, ImageDraw

# Configuración del dataset
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")
VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))
DATA_BASE = Path("/opt/airflow/car_damage_detection/data")
DATASET_RAW = DATA_BASE / "car-damages-forked"
DATASET_READY = DATA_BASE / "car-damages-ready"

CLASS_NAMES = ["Minor Damage (Dent)", "Minor Damage (Scratch)", "No Damage", "Severe Damage"]
NAME_MAP = {
    "Minor Damage -Dent-": "Minor Damage (Dent)",
    "Minor Damage -Scratch-": "Minor Damage (Scratch)",
    "No Damage": "No Damage",
    "Severe Damage": "Severe Damage",
}

# Configuración por defecto del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    'data_preparation',
    default_args=default_args,
    description='Descarga y prepara el dataset de car damages desde Roboflow',
    schedule=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),  # Fecha fija en el pasado
    catchup=False,
    tags=['data', 'preprocessing', 'car_damage_detection'],
)


def check_roboflow_api_key(**context):
    """
    Verifica que la API key de Roboflow esté configurada.
    """
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY no está configurada. "
            "Por favor configurar en el archivo .env o como variable de Airflow."
        )
    print(f"✓ ROBOFLOW_API_KEY configurada correctamente")
    return api_key


def download_dataset_task(**context):
    """
    Descarga el dataset de Roboflow en formato COCO segmentation.
    Siempre borra los datos existentes para forzar re-descarga.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY no configurada")
    
    # Borrar datos existentes si existen
    if DATASET_RAW.exists():
        print(f"🗑️  Borrando dataset existente en {DATASET_RAW}")
        shutil.rmtree(DATASET_RAW)
        print(f"✓ Dataset anterior eliminado")
    
    if DATASET_READY.exists():
        print(f"🗑️  Borrando datos procesados existentes en {DATASET_READY}")
        shutil.rmtree(DATASET_READY)
        print(f"✓ Datos procesados anteriores eliminados")
    
    # Crear directorio base si no existe
    DATA_BASE.mkdir(parents=True, exist_ok=True)
    print(f"📁 Directorio base creado/verificado: {DATA_BASE}")
    
    print(f"⬇️  Descargando dataset desde Roboflow:")
    print(f"    workspace={WORKSPACE}")
    print(f"    project={PROJECT}")
    print(f"    version={VERSION}")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download("coco-segmentation", location=str(DATASET_RAW))
    
    print(f"✓ Dataset descargado exitosamente en {DATASET_RAW}")
    
    # Verificar que se descargó correctamente
    if not DATASET_RAW.exists():
        raise Exception(f"Error: El dataset no se encuentra en {DATASET_RAW} después de la descarga")
    
    # Contar archivos descargados
    splits = ["train", "valid", "test"]
    for split in splits:
        split_dir = DATASET_RAW / split
        if split_dir.exists():
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            print(f"  {split}: {len(images)} imágenes")
    
    return f"Dataset descargado: {DATASET_RAW}"


def prepare_split(split):
    """
    Procesa un split (train/valid/test) del dataset COCO
    """
    src_split = DATASET_RAW / split
    json_path = src_split / "_annotations.coco.json"
    
    if not json_path.exists():
        print(f"⚠️  No se encontró {json_path}, saltando split {split}")
        return
    
    print(f"\n🔄 Procesando split: {split}")
    print(f"   Leyendo anotaciones desde: {json_path}")
    
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Total imágenes en COCO: {len(data['images'])}")
    print(f"   Total anotaciones: {len(data['annotations'])}")
    
    img_info = {img["id"]: img for img in data["images"]}
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    # Mapear categorías COCO a índices de clase
    cat_map = {}
    for cat in data["categories"]:
        mapped = NAME_MAP.get(cat["name"])
        if mapped is not None:
            cat_map[cat["id"]] = class_to_idx[mapped]
    
    print(f"   Categorías mapeadas: {len(cat_map)}")
    
    # Agrupar anotaciones por imagen
    anns_by_img = {}
    for ann in data["annotations"]:
        if ann["category_id"] not in cat_map:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    
    print(f"   Imágenes con anotaciones: {len(anns_by_img)}")
    
    # Crear directorios de salida
    inst_dir = DATASET_READY / "instance" / split
    inst_labels_dir = inst_dir / "labels"
    inst_images_dir = inst_dir / "images"
    inst_labels_dir.mkdir(parents=True, exist_ok=True)
    inst_images_dir.mkdir(parents=True, exist_ok=True)
    
    sem_dir = DATASET_READY / "semantic" / split
    sem_images_dir = sem_dir / "images"
    sem_masks_dir = sem_dir / "masks"
    sem_images_dir.mkdir(parents=True, exist_ok=True)
    sem_masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   ✓ Directorios de salida creados")
    
    # Procesar cada imagen
    processed_count = 0
    for img_id, anns in anns_by_img.items():
        info = img_info[img_id]
        w, h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem
        
        src_img = src_split / info["file_name"]
        if not src_img.exists():
            continue
        
        # Copiar imagen a ambos formatos
        shutil.copy2(src_img, inst_images_dir / info["file_name"])
        shutil.copy2(src_img, sem_images_dir / info["file_name"])
        
        # Formato instance (YOLOv8): archivo .txt con polígonos normalizados
        lines = []
        for ann in anns:
            segs = ann.get("segmentation")
            if not segs or not segs[0] or len(segs[0]) < 6:
                continue
            poly = segs[0]
            norm = [str(poly[i] / w if i % 2 == 0 else poly[i] / h) for i in range(len(poly))]
            cls_id = cat_map[ann["category_id"]]
            lines.append(f"{cls_id} " + " ".join(norm))
        
        if lines:
            (inst_labels_dir / f"{stem}.txt").write_text("\n".join(lines))
        
        # Formato semantic: máscara PNG con índices de clase
        sem_mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(sem_mask)
        for ann in anns:
            segs = ann.get("segmentation")
            if not segs or not segs[0] or len(segs[0]) < 6:
                continue
            cls_id = cat_map[ann["category_id"]] + 1  # +1 porque 0 es background
            coords = [(segs[0][i], segs[0][i+1]) for i in range(0, len(segs[0]), 2)]
            draw.polygon(coords, fill=cls_id)
        sem_mask.save(sem_masks_dir / f"{stem}.png")
        
        processed_count += 1
    
    print(f"   ✓ Split {split} procesado: {processed_count} imágenes")
    print(f"   📁 Instance: {inst_dir}")
    print(f"   📁 Semantic: {sem_dir}")


def prepare_datasets_task(**context):
    """
    Prepara los datasets en formato instance (YOLOv8) y semantic segmentation
    """
    print("🔧 Iniciando preparación de datasets...")
    print(f"📂 Dataset raw: {DATASET_RAW}")
    print(f"📂 Dataset ready: {DATASET_READY}")
    
    # Verificar que existe el dataset raw
    if not DATASET_RAW.exists():
        raise Exception(f"Dataset raw no encontrado en {DATASET_RAW}")
    
    # Procesar cada split
    for split in ["train", "valid", "test"]:
        prepare_split(split)
    
    # Crear data.yaml para YOLOv8
    inst_yaml = f"""train: {DATASET_READY}/instance/train/images
val: {DATASET_READY}/instance/valid/images
test: {DATASET_READY}/instance/test/images
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    data_yaml_path = DATASET_READY / "instance" / "data.yaml"
    data_yaml_path.write_text(inst_yaml)
    print(f"✓ Archivo data.yaml creado en {data_yaml_path}")
    
    # Resumen
    print("\n📊 Resumen de datasets preparados:")
    stats = {}
    for split in ["train", "valid", "test"]:
        inst_images = len(list((DATASET_READY / "instance" / split / "images").glob("*")))
        sem_images = len(list((DATASET_READY / "semantic" / split / "images").glob("*")))
        stats[split] = {'instance': inst_images, 'semantic': sem_images}
        print(f"  {split}: instance={inst_images} images, semantic={sem_images} images")
    
    print(f"\n✓ Preparación de datasets completada")
    print(f"📁 Los datos están disponibles en: {DATASET_READY}")
    return stats


def validate_prepared_data(**context):
    """
    Valida que los datos preparados tengan el formato correcto
    """
    
    # Validar estructura de directorios
    required_dirs = [
        "instance/train/images",
        "instance/train/labels",
        "instance/valid/images",
        "instance/valid/labels",
        "instance/test/images",
        "instance/test/labels",
        "semantic/train/images",
        "semantic/train/masks",
        "semantic/valid/images",
        "semantic/valid/masks",
        "semantic/test/images",
        "semantic/test/masks",
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = DATASET_READY / dir_path
        if not full_path.exists():
            missing_dirs.append(str(full_path))
    
    if missing_dirs:
        raise Exception(f"Directorios faltantes: {', '.join(missing_dirs)}")
    
    # Contar archivos en cada split
    stats = {}
    for split in ['train', 'valid', 'test']:
        inst_images = len(list((DATASET_READY / "instance" / split / "images").glob("*")))
        inst_labels = len(list((DATASET_READY / "instance" / split / "labels").glob("*.txt")))
        sem_images = len(list((DATASET_READY / "semantic" / split / "images").glob("*")))
        sem_masks = len(list((DATASET_READY / "semantic" / split / "masks").glob("*.png")))
        
        stats[split] = {
            'instance_images': inst_images,
            'instance_labels': inst_labels,
            'semantic_images': sem_images,
            'semantic_masks': sem_masks
        }
        
        print(f"{split.upper()}:")
        print(f"  Instance: {inst_images} images, {inst_labels} labels")
        print(f"  Semantic: {sem_images} images, {sem_masks} masks")
    
    # Verificar data.yaml
    data_yaml = DATASET_READY / "instance" / "data.yaml"
    if not data_yaml.exists():
        raise Exception("Archivo data.yaml no encontrado")
    
    print("\n✓ Validación completada exitosamente")
    print(f"✓ data.yaml encontrado en {data_yaml}")
    
    context['ti'].xcom_push(key='dataset_stats', value=stats)
    return stats


# Definir las tareas
check_api_key = PythonOperator(
    task_id='check_roboflow_api_key',
    python_callable=check_roboflow_api_key,
    dag=dag,
)

download_dataset = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset_task,
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

# Definir dependencias (orden de ejecución)
check_api_key >> download_dataset >> prepare_datasets >> validate_data
