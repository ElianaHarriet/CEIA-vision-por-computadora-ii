"""
Gradio Demo - Car Damage Detection
Interfaz para comparar modelos YOLO (instance) vs U-Net (semantic).
"""
import os
import io
import base64

import gradio as gr
import requests
from PIL import Image
from dotenv import load_dotenv

# Cargar variables desde el .env de la raíz del repo (mismo patrón que el resto
# del proyecto). No se commitea ningún valor real acá.
load_dotenv()

# La URL del FastAPI se arma desde las mismas vars que ya usa docker-compose:
# FASTAPI_HOST + FASTAPI_PORT. Se puede overridear entera con FASTAPI_URL.
_HOST = os.getenv("FASTAPI_HOST", "localhost")
_PORT = os.getenv("FASTAPI_PORT", "8800")
API_URL = os.getenv("FASTAPI_URL", f"http://{_HOST}:{_PORT}")

# Puerto donde sirve esta demo de Gradio.
DEMO_PORT = int(os.getenv("DEMO_PORT", "7860"))

# Paleta por clase, consistente con CLASS_COLORS del backend (app.py del FastAPI).
CLASS_PALETTE = {
    "Background": "#111827",
    "Minor Damage (Dent)": "#f59e0b",
    "Minor Damage (Scratch)": "#facc15",
    "No Damage": "#22c55e",
    "Severe Damage": "#ef4444",
}


def _b64_to_image(b64: str):
    """Decodifica un PNG en base64 a PIL.Image."""
    if not b64:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def _metric_card(label: str, value: str, accent: str) -> str:
    """HTML de una tarjeta de métrica."""
    return f"""
    <div class="metric-card" style="border-top: 3px solid {accent};">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def _legend_html() -> str:
    """Leyenda de colores de clases."""
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{color}"></span>{name}</span>'
        for name, color in CLASS_PALETTE.items()
        if name != "Background"
    )
    return f'<div class="legend">{chips}</div>'


def _empty_outputs(message_html):
    """Estado vacío/errores para todas las salidas."""
    return message_html, None, None, "", {}, {}, {}


def predict_and_compare(image, progress=gr.Progress()):
    """Envía la imagen a /predict/compare y arma las salidas visuales."""
    if image is None:
        return _empty_outputs(
            '<div class="status status-warn">Subí una foto de un auto para empezar.</div>'
        )

    try:
        progress(0.15, desc="Preparando imagen…")
        img_pil = Image.fromarray(image.astype("uint8"), "RGB")
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=92)
        buf.seek(0)

        progress(0.45, desc="Ejecutando YOLO y U-Net…")
        files = {"file": ("image.jpg", buf, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict/compare", files=files, timeout=120)

        if response.status_code != 200:
            return _empty_outputs(
                f'<div class="status status-err">Error {response.status_code}: {response.text}</div>'
            )

        progress(0.85, desc="Armando resultados…")
        data = response.json()
        inst = data["instance_segmentation"]
        sem = data["semantic_segmentation"]
        comp = data["comparison"]

        # Overlays
        inst_img = _b64_to_image(inst.get("overlay_png_base64"))
        sem_img = _b64_to_image(sem.get("overlay_png_base64"))

        # Tarjetas de métricas
        mean_iou = comp["mean_iou"]
        iou_accent = "#22c55e" if mean_iou >= 0.6 else "#f59e0b" if mean_iou >= 0.3 else "#ef4444"
        cards = (
            _metric_card("Detecciones YOLO", str(inst["num_detections"]), "#818cf8")
            + _metric_card("Área dañada · YOLO", f'{inst["total_damaged_area_pixels"]:,} px', "#f59e0b")
            + _metric_card("Área dañada · U-Net", f'{sem["total_damaged_area_pixels"]:,} px', "#facc15")
            + _metric_card("IoU promedio", f"{mean_iou:.3f}", iou_accent)
        )
        metrics_html = f'<div class="metrics-grid">{cards}</div>'

        # Distribuciones como barras (gr.Label espera valores 0-1)
        inst_dist = {
            name: d["percentage"] / 100.0
            for name, d in inst["class_distribution"].items()
        }
        sem_dist = {
            name: d["percentage"] / 100.0
            for name, d in sem["class_distribution"].items()
        }
        iou_dist = dict(comp["iou_per_class"])

        status = '<div class="status status-ok">Análisis completo · daños resaltados sobre la foto</div>'
        return status, inst_img, sem_img, metrics_html, inst_dist, sem_dist, iou_dist

    except requests.exceptions.Timeout:
        return _empty_outputs(
            '<div class="status status-err">El servidor tardó demasiado en responder (timeout).</div>'
        )
    except requests.exceptions.ConnectionError:
        return _empty_outputs(
            f'<div class="status status-err">No se pudo conectar con el servicio de modelos ({API_URL}).</div>'
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_outputs(f'<div class="status status-err">Error inesperado: {exc}</div>')


CUSTOM_CSS = """
.gradio-container {
    background: radial-gradient(1200px 600px at 15% -10%, #1e293b 0%, transparent 55%),
                radial-gradient(1000px 500px at 110% 10%, #312e81 0%, transparent 50%),
                linear-gradient(160deg, #0b1120 0%, #0f172a 60%, #111827 100%) !important;
    color: #e2e8f0 !important;
}
#hero {
    text-align: center;
    padding: 26px 18px 6px;
}
#hero h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #818cf8, #22d3ee 55%, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
#hero p { color: #94a3b8; margin: 8px 0 0; font-size: 1rem; }
.panel {
    background: rgba(30, 41, 59, 0.55) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(6px);
}
.legend { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; padding: 6px 0 2px; }
.chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(148,163,184,0.10); border: 1px solid rgba(148,163,184,0.18);
    padding: 5px 11px; border-radius: 999px; font-size: 0.82rem; color: #cbd5e1;
}
.chip .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.metrics-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 4px 0 2px;
}
.metric-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 14px; padding: 16px 14px; text-align: center;
}
.metric-value { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
.metric-label { font-size: 0.78rem; color: #94a3b8; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.04em; }
.status {
    border-radius: 12px; padding: 12px 16px; font-weight: 600; text-align: center; margin: 4px 0;
}
.status-ok   { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.35);  color: #86efac; }
.status-warn { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35); color: #fcd34d; }
.status-err  { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.35);  color: #fca5a5; }
#analyze-btn {
    background: linear-gradient(90deg, #6366f1, #22d3ee) !important;
    border: none !important; color: white !important; font-weight: 700 !important;
    font-size: 1.02rem !important;
}
.note { color: #94a3b8; font-size: 0.85rem; line-height: 1.5; }
@media (max-width: 820px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } }
"""

with gr.Blocks(title="Car Damage Detection", theme=gr.themes.Soft(
    primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
), css=CUSTOM_CSS) as demo:

    gr.HTML(
        '<div id="hero">'
        '<h1>🚗 Car Damage Detection</h1>'
        '<p>Comparación de daños con dos enfoques: <b>YOLOv8-seg</b> (instancias) '
        'vs <b>U-Net</b> (semántica). Subí una foto y mirá los daños resaltados.</p>'
        '</div>'
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes="panel"):
            image_input = gr.Image(label="Foto del auto", type="numpy", height=340)
            analyze_btn = gr.Button("🔍 Analizar daños", variant="primary", size="lg", elem_id="analyze-btn")
            gr.HTML(_legend_html())
        with gr.Column(scale=6, elem_classes="panel"):
            status_out = gr.HTML(
                '<div class="status status-warn">Subí una foto de un auto para empezar.</div>'
            )
            metrics_out = gr.HTML()

    with gr.Row():
        instance_img_out = gr.Image(label="YOLO · Instancias (overlay)", type="pil", height=300)
        semantic_img_out = gr.Image(label="U-Net · Semántica (overlay)", type="pil", height=300)

    with gr.Row():
        with gr.Column():
            instance_dist_out = gr.Label(label="YOLO · Distribución por clase", num_top_classes=5)
        with gr.Column():
            semantic_dist_out = gr.Label(label="U-Net · Distribución por clase", num_top_classes=5)

    iou_out = gr.Label(label="Acuerdo entre modelos · IoU por clase", num_top_classes=4)

    gr.HTML(
        '<div class="note">El <b>IoU</b> (Intersection over Union) mide el <b>acuerdo entre los '
        'dos modelos</b>, no la precisión contra ground truth (no existe para una foto subida por '
        'el usuario). Va de 0 (sin acuerdo) a 1 (acuerdo perfecto).</div>'
    )

    analyze_btn.click(
        fn=predict_and_compare,
        inputs=[image_input],
        outputs=[status_out, instance_img_out, semantic_img_out, metrics_out,
                 instance_dist_out, semantic_dist_out, iou_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=DEMO_PORT, share=False)
