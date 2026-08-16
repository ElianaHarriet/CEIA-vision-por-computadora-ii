"""
Gradio Demo - Car Damage Detection
Compara YOLO (instancia) vs U-Net (semántica) sobre imágenes del set de test,
usando el ground truth para medir el IoU real de cada modelo.
"""
import os
import io
import glob
import base64

import gradio as gr
import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv

# Cargar variables desde el .env de la raíz del repo (mismo patrón que el resto
# del proyecto). No se commitea ningún valor real acá.
load_dotenv()

_HOST = os.getenv("FASTAPI_HOST", "localhost")
_PORT = os.getenv("FASTAPI_PORT", "8800")
API_URL = os.getenv("FASTAPI_URL", f"http://{_HOST}:{_PORT}")
DEMO_PORT = int(os.getenv("DEMO_PORT", "7860"))

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

CLASS_NAMES = ["Background", "Minor Damage (Dent)", "Minor Damage (Scratch)", "No Damage", "Severe Damage"]
# Paleta por clase (RGB), consistente con CLASS_COLORS del backend.
CLASS_COLORS = {
    0: (17, 24, 39),
    1: (245, 158, 11),
    2: (250, 204, 21),
    3: (34, 197, 94),
    4: (239, 68, 68),
}
DAMAGE_CLASSES = (1, 2, 3, 4)


# ----------------------------- utilidades -----------------------------
def _b64_to_array(b64: str) -> np.ndarray:
    """Decodifica un PNG en base64 a un array 2D de class-ids."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))))


def _b64_to_image(b64: str):
    if not b64:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def _overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Colorea las clases de daño de `mask` sobre `image` (fondo intacto)."""
    base = np.array(image.convert("RGB"))
    h, w = mask.shape[:2]
    if base.shape[:2] != (h, w):
        base = np.array(image.convert("RGB").resize((w, h)))
    color = np.zeros_like(base)
    for cid, rgb in CLASS_COLORS.items():
        if cid == 0:
            continue
        color[mask == cid] = rgb
    out = base.copy()
    tinted = mask != 0
    out[tinted] = (base[tinted] * (1 - alpha) + color[tinted] * alpha).astype(np.uint8)
    return Image.fromarray(out)


def _iou_vs_gt(pred: np.ndarray, gt: np.ndarray) -> dict:
    """IoU por clase de `pred` contra el ground truth `gt`.

    Solo sobre las clases presentes en el GT (evita inflar el promedio con
    clases ausentes). Devuelve {nombre_clase: iou}.
    """
    if pred.shape != gt.shape:
        pred_img = Image.fromarray(pred.astype(np.uint8)).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST
        )
        pred = np.array(pred_img)
    result = {}
    for c in DAMAGE_CLASSES:
        if not (gt == c).any():
            continue
        a, b = (pred == c), (gt == c)
        union = (a | b).sum()
        result[CLASS_NAMES[c]] = float((a & b).sum()) / float(union) if union else 0.0
    return result


def _load_samples():
    """Lista los ejemplos disponibles: (path_imagen, path_gt, label)."""
    items = []
    for img_path in sorted(glob.glob(os.path.join(SAMPLES_DIR, "sample_*.jpg"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(SAMPLES_DIR, f"{stem}_gt.png")
        if os.path.exists(gt_path):
            items.append((img_path, gt_path, stem))
    return items


SAMPLES = _load_samples()
GALLERY = [(img, label) for img, _, label in SAMPLES]


# ----------------------------- lógica principal -----------------------------
def _metric_card(label, value, accent):
    return (
        f'<div class="metric-card" style="border-top:3px solid {accent};">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>'
    )


def _legend_html():
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:rgb{CLASS_COLORS[c]}"></span>{CLASS_NAMES[c]}</span>'
        for c in DAMAGE_CLASSES
    )
    return f'<div class="legend">{chips}</div>'


def _empty(msg):
    return msg, None, None, None, "", {}, {}


def analyze(selected_index, progress=gr.Progress()):
    """Corre ambos modelos sobre el ejemplo elegido y compara contra el GT."""
    if selected_index is None or selected_index < 0 or selected_index >= len(SAMPLES):
        return _empty('<div class="status status-warn">Elegí un ejemplo de la galería.</div>')

    img_path, gt_path, _ = SAMPLES[selected_index]
    try:
        progress(0.2, desc="Enviando imagen al servicio de modelos…")
        with open(img_path, "rb") as fh:
            files = {"file": (os.path.basename(img_path), fh, "image/jpeg")}
            resp = requests.post(f"{API_URL}/predict/compare", files=files, timeout=120)

        if resp.status_code != 200:
            return _empty(f'<div class="status status-err">Error {resp.status_code}: {resp.text}</div>')

        progress(0.7, desc="Comparando contra el ground truth…")
        data = resp.json()
        inst = data["instance_segmentation"]
        sem = data["semantic_segmentation"]

        original = Image.open(img_path).convert("RGB")
        gt_mask = np.array(Image.open(gt_path))

        # Overlays de predicciones (los arma el backend) y del ground truth (acá).
        yolo_img = _b64_to_image(inst.get("overlay_png_base64"))
        unet_img = _b64_to_image(sem.get("overlay_png_base64"))
        gt_img = _overlay(original, gt_mask)

        # IoU real de cada modelo contra el GT.
        yolo_mask = _b64_to_array(inst["mask_png_base64"])
        unet_mask = _b64_to_array(sem["mask_png_base64"])
        yolo_iou = _iou_vs_gt(yolo_mask, gt_mask)
        unet_iou = _iou_vs_gt(unet_mask, gt_mask)

        yolo_miou = round(sum(yolo_iou.values()) / len(yolo_iou), 3) if yolo_iou else 0.0
        unet_miou = round(sum(unet_iou.values()) / len(unet_iou), 3) if unet_iou else 0.0

        if yolo_miou > unet_miou:
            winner, w_accent = "YOLOv8-seg", "#818cf8"
        elif unet_miou > yolo_miou:
            winner, w_accent = "U-Net", "#22d3ee"
        else:
            winner, w_accent = "Empate", "#94a3b8"

        cards = (
            _metric_card("mIoU · YOLO vs GT", f"{yolo_miou:.3f}", "#818cf8")
            + _metric_card("mIoU · U-Net vs GT", f"{unet_miou:.3f}", "#22d3ee")
            + _metric_card("Más cercano al GT", winner, w_accent)
        )
        metrics_html = f'<div class="metrics-grid">{cards}</div>'

        status = ('<div class="status status-ok">Comparación lista · el IoU mide '
                  'precisión real contra el ground truth</div>')
        return status, yolo_img, unet_img, gt_img, metrics_html, yolo_iou, unet_iou

    except requests.exceptions.Timeout:
        return _empty('<div class="status status-err">El servidor tardó demasiado (timeout).</div>')
    except requests.exceptions.ConnectionError:
        return _empty(f'<div class="status status-err">No se pudo conectar con el servicio de modelos ({API_URL}).</div>')
    except Exception as exc:  # noqa: BLE001
        return _empty(f'<div class="status status-err">Error inesperado: {exc}</div>')


CUSTOM_CSS = """
.gradio-container {
    background: radial-gradient(1200px 600px at 15% -10%, #1e293b 0%, transparent 55%),
                radial-gradient(1000px 500px at 110% 10%, #312e81 0%, transparent 50%),
                linear-gradient(160deg, #0b1120 0%, #0f172a 60%, #111827 100%) !important;
    color: #e2e8f0 !important;
    --block-background-fill: rgba(30, 41, 59, 0.55) !important;
    --block-label-background-fill: rgba(15, 23, 42, 0.85) !important;
    --block-label-text-color: #cbd5e1 !important;
    --block-title-text-color: #e2e8f0 !important;
    --input-background-fill: rgba(15, 23, 42, 0.6) !important;
    --panel-background-fill: rgba(30, 41, 59, 0.55) !important;
    --border-color-primary: rgba(148, 163, 184, 0.16) !important;
    --body-text-color: #e2e8f0 !important;
    --body-text-color-subdued: #94a3b8 !important;
}
.gradio-container [data-testid="image"], .gradio-container .image-frame,
.gradio-container .image-container, .gradio-container .empty {
    background: rgba(15, 23, 42, 0.55) !important;
}
/* Resaltado fuerte del thumbnail seleccionado en la galería. */
#sample-gallery .thumbnail-item.selected,
#sample-gallery button.selected,
#sample-gallery .selected {
    outline: 3px solid #22d3ee !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.55), 0 0 20px rgba(34, 211, 238, 0.6) !important;
    border-radius: 10px !important;
    transform: scale(1.02);
    transition: all 0.12s ease-in-out;
}
#hero { text-align: center; padding: 26px 18px 6px; }
#hero h1 {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
    background: linear-gradient(90deg, #818cf8, #22d3ee 55%, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#hero p { color: #94a3b8; margin: 8px 0 0; }
.panel {
    background: rgba(30, 41, 59, 0.55) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 16px !important;
}
.legend { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; padding:6px 0 2px; }
.chip { display:inline-flex; align-items:center; gap:7px; background:rgba(148,163,184,0.10);
        border:1px solid rgba(148,163,184,0.18); padding:5px 11px; border-radius:999px;
        font-size:0.82rem; color:#cbd5e1; }
.chip .dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
.metrics-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:4px 0 2px; }
.metric-card { background:rgba(15,23,42,0.7); border:1px solid rgba(148,163,184,0.14);
               border-radius:14px; padding:16px 14px; text-align:center; }
.metric-value { font-size:1.5rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
.metric-label { font-size:0.78rem; color:#94a3b8; margin-top:6px; text-transform:uppercase; letter-spacing:0.04em; }
.status { border-radius:12px; padding:12px 16px; font-weight:600; text-align:center; margin:4px 0; }
.status-ok { background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.35); color:#86efac; }
.status-warn { background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.35); color:#fcd34d; }
.status-err { background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.35); color:#fca5a5; }
#analyze-btn { background:linear-gradient(90deg,#6366f1,#22d3ee) !important; border:none !important;
               color:white !important; font-weight:700 !important; }
.note { color:#94a3b8; font-size:0.85rem; line-height:1.5; }
@media (max-width: 820px) { .metrics-grid { grid-template-columns: repeat(1, 1fr); } }
"""

with gr.Blocks(title="Car Damage Detection", theme=gr.themes.Soft(
    primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
), css=CUSTOM_CSS) as demo:

    selected_state = gr.State(value=None)

    gr.HTML(
        '<div id="hero"><h1>🚗 Car Damage Detection</h1>'
        '<p>Comparación de <b>YOLOv8-seg</b> (instancias) vs <b>U-Net</b> (semántica) sobre '
        'imágenes del set de test. El <b>ground truth</b> permite medir el IoU real de cada modelo.</p></div>'
    )

    with gr.Row():
        with gr.Column(scale=5, elem_classes="panel"):
            gallery = gr.Gallery(
                value=GALLERY, label="Ejemplos del set de test", columns=5,
                height=200, allow_preview=False, object_fit="cover",
                elem_id="sample-gallery",
            )
            preview_out = gr.Image(label="Ejemplo seleccionado", type="filepath",
                                   height=240, interactive=False)
            analyze_btn = gr.Button("🔍 Analizar y comparar con ground truth",
                                    variant="primary", size="lg", elem_id="analyze-btn")
            gr.HTML(_legend_html())
        with gr.Column(scale=5, elem_classes="panel"):
            status_out = gr.HTML('<div class="status status-warn">Elegí un ejemplo de la galería.</div>')
            metrics_out = gr.HTML()

    with gr.Row():
        yolo_out = gr.Image(label="YOLO · Instancias", type="pil", height=280)
        unet_out = gr.Image(label="U-Net · Semántica", type="pil", height=280)
        gt_out = gr.Image(label="Ground Truth (anotación humana)", type="pil", height=280)

    with gr.Row():
        yolo_iou_out = gr.Label(label="YOLO · IoU por clase vs ground truth", num_top_classes=4)
        unet_iou_out = gr.Label(label="U-Net · IoU por clase vs ground truth", num_top_classes=4)

    gr.HTML(
        '<div class="note">El <b>IoU</b> (Intersection over Union) compara la predicción de cada '
        'modelo contra el <b>ground truth</b> (la anotación humana). Va de 0 (sin coincidencia) a 1 '
        '(coincidencia perfecta), y solo se promedia sobre las clases presentes en la anotación.</div>'
    )

    def _on_select(evt: gr.SelectData):
        idx = evt.index
        preview = SAMPLES[idx][0] if 0 <= idx < len(SAMPLES) else None
        label = SAMPLES[idx][2] if 0 <= idx < len(SAMPLES) else ""
        info = (f'<div class="status status-ok">Seleccionado: <b>{label}</b> · '
                'listo para analizar</div>') if preview else ""
        return idx, preview, info

    gallery.select(fn=_on_select, inputs=None, outputs=[selected_state, preview_out, status_out])

    analyze_btn.click(
        fn=analyze,
        inputs=[selected_state],
        outputs=[status_out, yolo_out, unet_out, gt_out, metrics_out, yolo_iou_out, unet_iou_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=DEMO_PORT, share=False)
