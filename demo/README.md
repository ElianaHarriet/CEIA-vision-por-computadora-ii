# Demo Gradio - Car Damage Detection

Interfaz gráfica para comparar los modelos YOLO vs U-Net en detección de daños de autos.
Consume el endpoint `/predict/compare` del servicio FastAPI.

El usuario elige una imagen del **set de test** (que tiene anotación humana /
*ground truth*), y la demo muestra el overlay de daños de cada modelo junto al
ground truth, calculando el **IoU real de cada modelo contra la verdad** — así
la comparación es objetiva (precisión), no solo acuerdo entre modelos.

## Ejemplos (`samples/`)

`samples/` contiene 25 imágenes del split de test con sus máscaras de ground
truth (`sample_XX.jpg` + `sample_XX_gt.png`). Se curaron a partir de
`data/semantic.tar.gz` (bucket `data` de MinIO) eligiendo casos con variedad de
daño (dent, scratch, severe) y en orientación natural. Las máscaras son PNG de
class-ids (0-4), la misma convención que usa la evaluación del proyecto.

## Configuración

La demo se configura desde el `.env` de la raíz del repo (mismo patrón que el
resto del proyecto). Variables relevantes:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `FASTAPI_HOST` | `localhost` | Host donde corre el FastAPI |
| `FASTAPI_PORT` | `8800` | Puerto del FastAPI |
| `FASTAPI_URL` | *(derivada)* | URL completa; si se setea, tiene prioridad sobre HOST/PORT |
| `DEMO_PORT` | `7860` | Puerto donde sirve esta demo |

Para consumir el FastAPI desplegado en DigitalOcean:
```dotenv
FASTAPI_HOST=24.144.120.67
```

Para consumir un FastAPI local, dejar `FASTAPI_HOST=localhost`.

## Uso con Docker (recomendado)

La demo está integrada en el `docker-compose.yaml` bajo los perfiles
`infrastructure` y `all`. Dentro de la red de Docker le pega al FastAPI por el
nombre de servicio (`http://fastapi:8800`), así que no depende de la IP pública.

```bash
# Levantar la demo (rebuild incluido)
docker compose --profile infrastructure up -d --build demo
```

Acceder en: http://<host>:7860 (o el `DEMO_PORT` configurado)

En el droplet de DigitalOcean:
```bash
docker compose --profile infrastructure up -d --build demo
# luego abrir http://24.144.120.67:7860
```

## Uso local (sin Docker)

```bash
pip install -r requirements.txt
python app.py
```

Disponible en: http://localhost:7860 (o el `DEMO_PORT` configurado).
Para apuntar al FastAPI de DigitalOcean, alcanza con `FASTAPI_HOST=24.144.120.67`
en el `.env` de la raíz.

## Características

- 🖼️ Galería de ejemplos del set de test (con ground truth)
- 🟠 Overlay de daños de YOLO (instancias) sobre la imagen
- 🎨 Overlay de daños de U-Net (semántica) sobre la imagen
- ✅ Overlay del ground truth (anotación humana) para referencia visual
- 📊 mIoU de cada modelo vs ground truth y cuál se acerca más
- 📈 IoU por clase de cada modelo contra la verdad
