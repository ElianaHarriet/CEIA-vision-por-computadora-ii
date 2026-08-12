# Setup Local Airflow para Entrenamiento

Este documento explica cómo configurar Airflow localmente para entrenar modelos usando RunPod y la infraestructura compartida en DigitalOcean.

## Prerequisitos

- Docker Desktop instalado y corriendo
- Git instalado
- Acceso al repositorio (clonar desde GitHub)

## Paso 1: Clonar el repositorio

```bash
git clone https://github.com/ElianaHarriet/CEIA-vision-por-computadora-ii.git
cd CEIA-vision-por-computadora-ii
git checkout feat/multi-user-docker-workflow
```

## Paso 2: Crear archivo `.env`

**IMPORTANTE:** Necesitás credenciales del equipo para conectarte a la infraestructura compartida (DigitalOcean).

1. Copiá el template:
```bash
cp .env.template .env
```

2. Pedí al equipo:
   - IP del servidor de DigitalOcean
   - Credenciales de MinIO (AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY)
   - API Key de Roboflow
   - Workspace y proyecto de Roboflow
   - Endpoints de RunPod (luis y santiago)

3. Editá el archivo `.env` y reemplazá los valores `PEDIR_AL_EQUIPO` con los valores reales que te pasaron.

## Paso 3: Configurar perfil de RunPod

Editá el archivo `airflow/config/.runpod-profile` y poné tu nombre:

```bash
# Opciones: luis, santiago, eliana
luis
```

Cada persona usa su propio endpoint de RunPod.

## Paso 4: Configurar API Key de RunPod

Necesitás agregar tu API key de RunPod a los secrets de Airflow.

Editá el archivo `airflow/secrets/variables.yaml` y agregá tu API key:

```yaml
RUNPOD_API_KEY: "tu-api-key-de-runpod-aqui"
```

**Importante:** 
- Si tu perfil es `luis`, tu variable debe llamarse `RUNPOD_API_KEY_LUIS`
- Si tu perfil es `santiago`, tu variable debe llamarse `RUNPOD_API_KEY`
- Si tu perfil es `eliana`, podés usar cualquiera de los dos

## Paso 5: Levantar Airflow

```bash
docker compose --profile local up -d
```

Esto va a levantar:
- Airflow (scheduler, webserver, worker, etc.)
- PostgreSQL (base de datos local de Airflow)
- Redis (para Celery)
- MinIO local (para staging temporal)

**NO** levanta MLflow ni MinIO de producción porque esos están en DigitalOcean.

## Paso 6: Acceder a la UI de Airflow

Abrí tu navegador y andá a: http://localhost:8080

- **Usuario:** `airflow`
- **Password:** `airflow`

## Paso 7: Ver resultados en MLflow

Abrí tu navegador y andá a la URL de MLflow que te pasó el equipo (la IP de DigitalOcean en el puerto 5001).

Ahí podés ver todos los experimentos y comparar métricas entre entrenamientos de todo el equipo.

En la UI de Airflow vas a ver estos DAGs:

- **`training_instance_segmentation`** - Entrena YOLOv8 para instance segmentation
- **`training_semantic_segmentation`** - Entrena U-Net para semantic segmentation

Para ejecutar un DAG:
1. Hacé click en el nombre del DAG
2. Click en el botón **"Trigger"** (▶️) arriba a la derecha

El DAG va a:
1. Subir el dataset a MinIO en DigitalOcean
2. Disparar el entrenamiento en tu endpoint de RunPod
3. Guardar resultados en MLflow en DigitalOcean

## Paso 8: Ver resultados en MLflow

Abrí tu navegador y andá a: http://24.144.120.67:5001

Ahí podés ver todos los experimentos y comparar métricas entre entrenamientos de todo el equipo.

## Comandos útiles

### Ver logs de los contenedores
```bash
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-worker
```

### Reiniciar Airflow (si cambiaste el .env o el código)
```bash
docker compose --profile local down
docker compose --profile local up -d
```

### Detener Airflow
```bash
docker compose --profile local down
```

## Troubleshooting

### Error: "RUNPOD_API_KEY not found"
Revisá que pusiste tu API key en `airflow/secrets/variables.yaml` con el nombre correcto según tu perfil.

### Error: "Cannot connect to MinIO"
Verificá que tenés internet y que la IP de DigitalOcean (24.144.120.67) es accesible.

### Los DAGs no aparecen en la UI
Esperá 1-2 minutos después de levantar los contenedores. Airflow necesita tiempo para escanear los DAGs.

### El worker se queda sin memoria
Si tu máquina tiene poca RAM, podés reducir `AIRFLOW__CELERY__WORKER_CONCURRENCY` en el docker-compose.yaml.

## Arquitectura

```
Tu Máquina Local          DigitalOcean (24.144.120.67)      RunPod
┌──────────────────┐      ┌────────────────────────┐       ┌──────────────┐
│                  │      │                        │       │              │
│  Airflow         │─────>│  MLflow (5001)        │<──────│  GPU Worker  │
│  (orchestrator)  │      │  MinIO (9000)         │       │  (training)  │
│                  │      │  PostgreSQL           │       │              │
│  Dataset local   │──┐   │                        │   ┌───│              │
│                  │  │   └────────────────────────┘   │   └──────────────┘
└──────────────────┘  │                                │
                      │   Upload dataset to MinIO      │
                      └───────────────────────────────>│
                                                       │
                          Download dataset from MinIO  │
                          Train model                  │
                          Upload results to MLflow ────┘
```

Todo el equipo comparte la misma infraestructura en DigitalOcean (MLflow + MinIO), pero cada uno corre Airflow localmente y usa su propio endpoint de RunPod.
