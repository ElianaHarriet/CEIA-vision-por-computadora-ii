"""
FastAPI application for car damage segmentation model serving.

Endpoints:
- POST /predict/instance: Predict using instance segmentation model
- POST /predict/semantic: Predict using semantic segmentation model
- POST /predict/compare: Compare both models on the same image
- GET /models/info: Get information about loaded models
- GET /health: Health check endpoint
"""

import os
import base64
import io
import tempfile

import torch

# Disable NNPACK IMMEDIATELY after torch import to avoid errors on older CPUs/VMs
torch.backends.nnpack.enabled = False

import mlflow
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from evaluation.model_loader import YOLOModelLoader, UNetModelLoader
from evaluation.predictor import YOLOPredictor, UNetPredictor
from evaluation.mask_flattener import MaskFlattener

app = FastAPI(
    title="Car Damage Segmentation API",
    description="API for comparing instance vs semantic segmentation models",
    version="1.0.0"
)

# Index 0 = background, copied from EvaluationConfig.CLASS_NAMES
# (airflow/dags/evaluation/config.py) so this service stays decoupled from
# evaluation/config.py's Airflow-only path assumptions.
CLASS_NAMES = ["Background", "Minor Damage (Dent)", "Minor Damage (Scratch)", "No Damage", "Severe Damage"]

# Color per class id (RGB), used to build the visual overlays returned by
# /predict/compare. Shared with the demo's legend so colors stay consistent.
CLASS_COLORS = {
    0: (17, 24, 39),     # Background  - dark slate (never tinted on overlays)
    1: (245, 158, 11),   # Minor Damage (Dent)    - amber
    2: (250, 204, 21),   # Minor Damage (Scratch) - yellow
    3: (34, 197, 94),    # No Damage              - green
    4: (239, 68, 68),    # Severe Damage          - red
}

# Environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME_INSTANCE = os.getenv("MODEL_NAME_INSTANCE", "car-damage-instance-segmentation")
MODEL_NAME_SEMANTIC = os.getenv("MODEL_NAME_SEMANTIC", "car-damage-semantic-segmentation")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_VERSION_INSTANCE = os.getenv("MODEL_VERSION_INSTANCE", "latest")
MODEL_VERSION_SEMANTIC = os.getenv("MODEL_VERSION_SEMANTIC", "latest")

# Populated by load_models() on startup
instance_predictor = None
semantic_predictor = None
mask_flattener = None
instance_model_info = None
semantic_model_info = None


def _resolve_version(loader, model_name: str, version_override: str):
    """Return the ModelVersion to load: a pinned version, or the stage's latest."""
    if version_override != "latest":
        return mlflow.MlflowClient().get_model_version(model_name, version_override)
    return loader.check_model_availability(model_name, MODEL_STAGE)


def _load_instance_model():
    """Load the YOLO instance-segmentation model, tolerating failure."""
    global instance_predictor, instance_model_info
    try:
        loader = YOLOModelLoader(MLFLOW_TRACKING_URI)
        version = _resolve_version(loader, MODEL_NAME_INSTANCE, MODEL_VERSION_INSTANCE)
        model_uri = f"runs:/{version.run_id}/model/best.pt"
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        from ultralytics import YOLO
        model = YOLO(local_path)
        instance_predictor = YOLOPredictor(model, conf=float(os.getenv("YOLO_CONF", "0.4")))
        instance_model_info = {"version": version.version, "run_id": version.run_id, "stage": version.current_stage}
        print(f"✓ Instance model loaded: v{version.version}")
    except Exception as exc:
        print(f"✗ Failed to load instance model: {exc}")


def _load_semantic_model():
    """Load the U-Net semantic-segmentation model, tolerating failure."""
    global semantic_predictor, semantic_model_info
    try:
        loader = UNetModelLoader(MLFLOW_TRACKING_URI)
        version = _resolve_version(loader, MODEL_NAME_SEMANTIC, MODEL_VERSION_SEMANTIC)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_uri = f"models:/{MODEL_NAME_SEMANTIC}/{version.version}"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model = model.to(device)
        model.eval()
        img_size = (int(os.getenv("UNET_IMG_SIZE", "640")),) * 2
        semantic_predictor = UNetPredictor(model, device, img_size=img_size)
        semantic_model_info = {"version": version.version, "run_id": version.run_id, "stage": version.current_stage}
        print(f"✓ Semantic model loaded: v{version.version}")
    except Exception as exc:
        print(f"✗ Failed to load semantic model: {exc}")


@app.on_event("startup")
async def load_models():
    """Load both models from the MLflow Model Registry on startup."""
    global mask_flattener
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _load_instance_model()
    _load_semantic_model()
    mask_flattener = MaskFlattener(strategy='class_aware', class_map={0: 1, 1: 2, 2: 3, 3: 4})


def _validate_upload(file: UploadFile):
    """Reject non-image uploads — this endpoint is exposed directly to the internet."""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")


def _save_upload_to_tmp(contents: bytes, filename: str) -> str:
    """Persist uploaded bytes to a temp file — predictors take a path, not bytes."""
    suffix = os.path.splitext(filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        return tmp.name


def _encode_mask_png(mask: np.ndarray) -> str:
    """Encode a class-id mask as a base64 PNG for JSON transport."""
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_overlay_png(image_path: str, mask: np.ndarray, alpha: float = 0.55) -> str:
    """Blend the class-id mask (colorized) over the original photo as a base64 PNG.

    Only damaged/relevant classes are tinted; background pixels keep the original
    photo so the result reads as "damage highlighted on the real car".
    """
    base = Image.open(image_path).convert("RGB")
    base_arr = np.array(base)
    h, w = mask.shape[:2]
    if base_arr.shape[:2] != (h, w):
        base = base.resize((w, h))
        base_arr = np.array(base)

    color = np.zeros_like(base_arr)
    for cid, rgb in CLASS_COLORS.items():
        if cid == 0:
            continue  # leave background untouched
        color[mask == cid] = rgb

    overlay = base_arr.copy()
    tinted = mask != 0
    overlay[tinted] = (
        base_arr[tinted] * (1 - alpha) + color[tinted] * alpha
    ).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _class_distribution(mask: np.ndarray) -> dict:
    """Pixel count and percentage per class present in a mask."""
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    return {
        CLASS_NAMES[int(u)]: {"pixels": int(c), "percentage": round(100 * float(c) / total, 2)}
        for u, c in zip(unique, counts)
    }


def _iou_per_class(mask_a: np.ndarray, mask_b: np.ndarray, classes=(1, 2, 3, 4)) -> dict:
    """Per-class IoU between two class-id masks — model agreement, not ground-truth accuracy."""
    result = {}
    for c in classes:
        a, b = (mask_a == c), (mask_b == c)
        union = (a | b).sum()
        result[CLASS_NAMES[c]] = 1.0 if union == 0 else float((a & b).sum()) / float(union)
    return result


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Car Damage Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models_info": "/models/info",
            "predict_instance": "POST /predict/instance",
            "predict_semantic": "POST /predict/semantic",
            "compare_models": "POST /predict/compare"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint. Verifies API is running and models are loaded."""
    models_loaded = instance_predictor is not None and semantic_predictor is not None
    return {
        "status": "healthy" if models_loaded else "degraded",
        "instance_model_loaded": instance_predictor is not None,
        "semantic_model_loaded": semantic_predictor is not None,
        "message": "All models loaded" if models_loaded else "Models not loaded yet"
    }


@app.get("/models/info")
async def models_info():
    """Get version/run_id/stage of the models actually loaded."""
    return {
        "instance_model": {
            "name": MODEL_NAME_INSTANCE,
            "stage": MODEL_STAGE,
            "loaded": instance_predictor is not None,
            **(instance_model_info or {})
        },
        "semantic_model": {
            "name": MODEL_NAME_SEMANTIC,
            "stage": MODEL_STAGE,
            "loaded": semantic_predictor is not None,
            **(semantic_model_info or {})
        }
    }


@app.post("/predict/instance")
async def predict_instance(file: UploadFile = File(...)):
    """Predict using instance segmentation model (YOLOv8-seg)."""
    if instance_predictor is None:
        raise HTTPException(status_code=503, detail="Instance model not loaded")
    _validate_upload(file)
    contents = await file.read()
    tmp_path = _save_upload_to_tmp(contents, file.filename)
    try:
        pred = instance_predictor._predict_single(tmp_path)
        classes = [int(c) for c in pred["classes"]]
        return {
            "num_detections": len(classes),
            "boxes": [b.tolist() for b in pred["boxes"]],
            "classes": classes,
            "class_names": [CLASS_NAMES[c + 1] for c in classes],
            "confidences": [float(c) for c in pred["confidences"]]
        }
    finally:
        os.unlink(tmp_path)


@app.post("/predict/semantic")
async def predict_semantic(file: UploadFile = File(...)):
    """Predict using semantic segmentation model (U-Net)."""
    if semantic_predictor is None:
        raise HTTPException(status_code=503, detail="Semantic model not loaded")
    _validate_upload(file)
    contents = await file.read()
    tmp_path = _save_upload_to_tmp(contents, file.filename)
    try:
        mask = semantic_predictor._predict_single(tmp_path)
        # Damaged classes are 1 (Dent), 2 (Scratch), 4 (Severe) — 0 is background, 3 is "No Damage".
        damaged = int(np.isin(mask, [1, 2, 4]).sum())
        return {
            "mask_shape": list(mask.shape),
            "class_distribution": _class_distribution(mask),
            "total_damaged_area_pixels": damaged,
            "total_damaged_area_pct": round(100 * damaged / mask.size, 2),
            "mask_png_base64": _encode_mask_png(mask)
        }
    finally:
        os.unlink(tmp_path)


@app.post("/predict/compare")
async def compare_models(file: UploadFile = File(...)):
    """Compare both models on the same image (main endpoint for this project)."""
    if instance_predictor is None or semantic_predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    _validate_upload(file)
    contents = await file.read()
    tmp_path = _save_upload_to_tmp(contents, file.filename)
    try:
        instance_pred = instance_predictor._predict_single(tmp_path)
        semantic_mask = semantic_predictor._predict_single(tmp_path)

        flat_instance = mask_flattener.flatten_predictions({"upload": instance_pred})["upload"]
        if flat_instance is None:
            flat_instance = np.zeros_like(semantic_mask)

        iou = _iou_per_class(flat_instance, semantic_mask)
        # Overlays reuse the masks already computed above (no extra inference).
        instance_overlay = _encode_overlay_png(tmp_path, flat_instance)
        semantic_overlay = _encode_overlay_png(tmp_path, semantic_mask)
        # Damaged classes are 1 (Dent), 2 (Scratch), 4 (Severe) — 0 is background, 3 is "No Damage".
        return {
            "instance_segmentation": {
                "num_detections": len(instance_pred["classes"]),
                "total_damaged_area_pixels": int(np.isin(flat_instance, [1, 2, 4]).sum()),
                "class_distribution": _class_distribution(flat_instance),
                "overlay_png_base64": instance_overlay,
                # Máscara cruda de class-ids (0-4) para comparar contra ground truth.
                "mask_png_base64": _encode_mask_png(flat_instance)
            },
            "semantic_segmentation": {
                "total_damaged_area_pixels": int(np.isin(semantic_mask, [1, 2, 4]).sum()),
                "class_distribution": _class_distribution(semantic_mask),
                "overlay_png_base64": semantic_overlay,
                "mask_png_base64": _encode_mask_png(semantic_mask)
            },
            "comparison": {
                "iou_per_class": iou,
                "mean_iou": round(sum(iou.values()) / len(iou), 4),
                "note": "IoU measures agreement between the two models, not accuracy against ground truth (none exists for a user-uploaded photo)."
            }
        }
    finally:
        os.unlink(tmp_path)
