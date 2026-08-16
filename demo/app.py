"""
Gradio Demo - Car Damage Detection
Interfaz para comparar modelos YOLO vs U-Net
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

def predict_and_compare(image):
    """
    Envía la imagen al endpoint /predict/compare y muestra resultados.
    """
    if image is None:
        return "Por favor, sube una imagen", None, None, None
    
    try:
        # Convertir imagen a bytes
        img_pil = Image.fromarray(image.astype('uint8'), 'RGB')
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        buf.seek(0)
        
        # Enviar al API
        files = {'file': ('image.jpg', buf, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict/compare", files=files, timeout=60)
        
        if response.status_code != 200:
            return f"Error {response.status_code}: {response.text}", None, None, None
        
        result = response.json()
        
        # Extraer resultados
        instance_data = result['instance_segmentation']
        semantic_data = result['semantic_segmentation']
        comparison = result['comparison']
        
        # Decodificar máscaras (si están en base64)
        # El endpoint devuelve class_distribution, no máscaras directas en /compare
        # Vamos a mostrar los datos numéricos
        
        # Formatear resultados de instance
        instance_text = f"""**YOLO (Instance Segmentation)**
        
📊 **Detecciones:** {instance_data['num_detections']}
🔍 **Área dañada:** {instance_data['total_damaged_area_pixels']:,} píxeles

**Distribución por clase:**
"""
        for class_name, data in instance_data['class_distribution'].items():
            instance_text += f"\n- {class_name}: {data['pixels']:,} px ({data['percentage']:.1f}%)"
        
        # Formatear resultados de semantic
        semantic_text = f"""**U-Net (Semantic Segmentation)**
        
🔍 **Área dañada:** {semantic_data['total_damaged_area_pixels']:,} píxeles ({semantic_data.get('total_damaged_area_pct', 0):.1f}%)

**Distribución por clase:**
"""
        for class_name, data in semantic_data['class_distribution'].items():
            semantic_text += f"\n- {class_name}: {data['pixels']:,} px ({data['percentage']:.1f}%)"
        
        # Formatear comparación
        comparison_text = f"""**📊 Comparación entre Modelos**

**IoU (Intersection over Union) por clase:**
"""
        for class_name, iou_value in comparison['iou_per_class'].items():
            comparison_text += f"\n- {class_name}: {iou_value:.3f}"
        
        comparison_text += f"\n\n**IoU Promedio:** {comparison['mean_iou']:.3f}"
        comparison_text += f"\n\n⚠️ {comparison['note']}"
        
        # Intentar decodificar máscara semántica si existe
        semantic_mask = None
        if 'mask_png_base64' in semantic_data:
            try:
                mask_bytes = base64.b64decode(semantic_data['mask_png_base64'])
                semantic_mask = Image.open(io.BytesIO(mask_bytes))
            except Exception as e:
                print(f"Error decodificando máscara: {e}")
        
        return instance_text, semantic_text, comparison_text, semantic_mask
        
    except requests.exceptions.Timeout:
        return "⏱️ Timeout: El servidor tardó demasiado en responder", None, None, None
    except requests.exceptions.ConnectionError:
        return f"❌ Error de conexión: No se puede conectar a {API_URL}", None, None, None
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None

# Crear interfaz
with gr.Blocks(title="Car Damage Detection - YOLO vs U-Net", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 Car Damage Detection - Comparación de Modelos
    
    Sube una foto de un auto para detectar daños usando dos modelos diferentes:
    - **YOLO (Instance Segmentation)**: Detecta objetos individuales dañados
    - **U-Net (Semantic Segmentation)**: Clasifica cada píxel de la imagen
    
    El sistema compara ambos modelos y calcula métricas de acuerdo.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Sube una foto del auto", type="numpy")
            predict_btn = gr.Button("🔍 Analizar Daños", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            semantic_mask_output = gr.Image(label="Máscara Semántica (U-Net)", type="pil")
    
    with gr.Row():
        with gr.Column():
            instance_output = gr.Markdown(label="Resultados YOLO")
        
        with gr.Column():
            semantic_output = gr.Markdown(label="Resultados U-Net")
    
    comparison_output = gr.Markdown(label="Comparación")
    
    predict_btn.click(
        fn=predict_and_compare,
        inputs=[image_input],
        outputs=[instance_output, semantic_output, comparison_output, semantic_mask_output]
    )
    
    gr.Markdown("""
    ---
    ### 📝 Notas:
    - El IoU (Intersection over Union) mide el **acuerdo entre modelos**, no la precisión contra ground truth
    - Los valores de IoU van de 0 (sin acuerdo) a 1 (acuerdo perfecto)
    - Las clases son: Minor Damage (Dent), Minor Damage (Scratch), No Damage, Severe Damage
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=DEMO_PORT,
        share=False
    )
