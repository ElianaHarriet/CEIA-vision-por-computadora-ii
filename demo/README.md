# Demo Gradio - Car Damage Detection

Interfaz gráfica para comparar los modelos YOLO vs U-Net en detección de daños de autos.
Consume el endpoint `/predict/compare` del servicio FastAPI.

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

Abre el navegador en: http://localhost:7860 (o el `DEMO_PORT` configurado).
Para que apunte al FastAPI de DigitalOcean, setear `FASTAPI_HOST=24.144.120.67`
en el `.env` de la raíz.

## Características

- 📤 Upload de imágenes
- 🔍 Análisis con YOLO (instance segmentation)
- 🎨 Análisis con U-Net (semantic segmentation)
- 📊 Comparación de resultados con métricas IoU
- 🖼️ Visualización de máscara semántica
