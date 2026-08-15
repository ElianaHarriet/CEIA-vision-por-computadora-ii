# Setup DigitalOcean - Infraestructura Compartida

Este documento explica cómo configurar y mantener la infraestructura compartida en DigitalOcean (MLflow + MinIO + PostgreSQL).

## 📋 Información del Servidor

- **IP Pública:** 24.144.120.67
- **Hostname:** ubuntu-mlflow-minio
- **Plan:** Premium Intel - 4 GB RAM / 2 vCPUs
- **Sistema Operativo:** Ubuntu 22.04 LTS
- **Usuario:** root

## 🎯 Servicios Desplegados

| Servicio | Puerto | URL |
|----------|--------|-----|
| MLflow UI | 5001 | http://24.144.120.67:5001 |
| MinIO API | 9000 | http://24.144.120.67:9000 |
| MinIO Console | 9001 | http://24.144.120.67:9001 |
| FastAPI (serving) | 8800 | http://24.144.120.67:8800 (docs en `/docs`) |
| PostgreSQL | 5432 | (interno, no expuesto) |

## 🚀 Setup Inicial (Ya hecho, solo para referencia)

### 1. Conectarse al servidor

```bash
ssh root@24.144.120.67
```

### 2. Instalar Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### 3. Clonar el repositorio

```bash
cd /root
git clone https://github.com/ElianaHarriet/CEIA-vision-por-computadora-ii.git
cd CEIA-vision-por-computadora-ii
```

### 4. Crear archivo `.env` mínimo

El servidor de DigitalOcean solo necesita configuración básica de infraestructura:

```bash
cat > .env << 'EOF'
# MinIO
MINIO_ACCESS_KEY=minio
MINIO_SECRET_ACCESS_KEY=minio123
MINIO_PORT=9000
MINIO_PORT_UI=9001
MLFLOW_BUCKET_NAME=mlflow
DATA_REPO_BUCKET_NAME=data

# PostgreSQL
PG_USER=airflow
PG_PASSWORD=airflow
PG_DATABASE=airflow
PG_PORT=5432

# MLflow
MLFLOW_PORT=5001

# FastAPI (serving)
FASTAPI_PORT=8800
MLFLOW_EXPERIMENT_NAME=car-damage-segmentation
MODEL_NAME_INSTANCE=car-damage-instance-segmentation
MODEL_NAME_SEMANTIC=car-damage-semantic-segmentation
MODEL_STAGE=Production
MODEL_VERSION_INSTANCE=latest
MODEL_VERSION_SEMANTIC=latest
EOF
```

No seteamos acá `MLFLOW_TRACKING_URI`/`MLFLOW_S3_ENDPOINT_URL`: al quedar sin definir, el default `${VAR:-http://mlflow:5000}` de `docker-compose.yaml` resuelve al contenedor `mlflow`/`s3` sibling por red interna de Docker.

### 5. Configurar firewall (opcional)

El servidor actualmente tiene el firewall deshabilitado (`ufw status: inactive`). Si se desea habilitar:

```bash
# Habilitar puertos necesarios
ufw allow 22/tcp    # SSH
ufw allow 5001/tcp  # MLflow
ufw allow 9000/tcp  # MinIO API
ufw allow 9001/tcp  # MinIO Console
ufw enable
```

### 6. Levantar servicios

```bash
docker compose --profile infrastructure up -d
```

Esto levanta:
- **postgres**: Base de datos para MLflow
- **s3**: MinIO (almacenamiento de artefactos)
- **mlflow**: Servidor MLflow
- **create_buckets**: Inicialización de buckets (corre una sola vez y se detiene)
- **fastapi**: Serving de los modelos entrenados (instance + semantic + compare)

### 7. Verificar que todo está corriendo

```bash
docker compose ps
```

Deberías ver:
- `postgres` - running
- `s3` - running  
- `mlflow` - running
- `create_buckets` - exited (0) ← es normal, solo crea buckets y termina

## 🔧 Mantenimiento

### Ver logs de servicios

```bash
# Todos los servicios
docker compose logs -f

# Un servicio específico
docker compose logs -f mlflow
docker compose logs -f s3
docker compose logs -f postgres
```

### Reiniciar servicios

```bash
docker compose --profile infrastructure restart
```

### Detener servicios

```bash
docker compose --profile infrastructure down
```

### Actualizar código después de cambios

```bash
cd /root/CEIA-vision-por-computadora-ii
git pull origin main
docker compose --profile infrastructure down
docker compose --profile infrastructure up -d
```

### Verificar buckets en MinIO

```bash
# Entrar al contenedor de MinIO client
docker compose run --rm s3-client mc ls s3/

# Deberías ver:
# mlflow/
# data/
```

### Backup de base de datos PostgreSQL

```bash
# Crear backup
docker compose exec postgres pg_dump -U airflow airflow > mlflow_backup_$(date +%Y%m%d).sql

# Restaurar backup
docker compose exec -T postgres psql -U airflow airflow < mlflow_backup_20260812.sql
```

### Ver espacio en disco

```bash
df -h
docker system df  # Ver espacio usado por Docker
```

### Limpiar espacio (si es necesario)

```bash
# Limpiar contenedores, imágenes y volúmenes no usados
docker system prune -a --volumes
```

## 🔍 Troubleshooting

### MLflow no carga en el navegador

1. Verificar que el contenedor está corriendo:
```bash
docker compose ps mlflow
```

2. Ver logs de MLflow:
```bash
docker compose logs mlflow
```

3. Verificar conectividad:
```bash
curl http://localhost:5001
```

### MinIO no carga en el navegador

1. Verificar que el contenedor está corriendo:
```bash
docker compose ps s3
```

2. Ver logs de MinIO:
```bash
docker compose logs s3
```

3. Verificar puertos:
```bash
netstat -tlnp | grep -E '9000|9001'
```

### PostgreSQL no conecta

1. Ver logs:
```bash
docker compose logs postgres
```

### Conectarse manualmente a PostgreSQL

```bash
docker compose exec postgres psql -U airflow -d airflow
```

### "No space left on device"

Ver uso de disco y limpiar:
```bash
df -h
docker system df
docker system prune -a --volumes
```

### Contenedores se reinician constantemente

Ver logs para identificar el problema:
```bash
docker compose logs --tail=100
```

## 📊 Monitoreo

### Ver uso de recursos

```bash
# CPU y memoria de contenedores
docker stats

# Espacio en disco
df -h
du -sh /var/lib/docker
```

### Ver runs en MLflow

```bash
# Entrar a PostgreSQL y contar runs
docker compose exec postgres psql -U mlflow_user -d mlflow_db -c "SELECT COUNT(*) FROM runs;"
```

## 🔐 Seguridad

### Credenciales actuales (para desarrollo)

**MinIO:**
- Access Key: `minio`
- Secret Key: `minio123`

**PostgreSQL:**
- User: `airflow`
- Password: `airflow`
- Database: `airflow`

**⚠️ IMPORTANTE:** Estas credenciales son para desarrollo. Para producción se deben cambiar y usar secrets más seguros.

### Cambiar credenciales (si es necesario)

1. Editar `.env` en el servidor
2. Detener servicios: `docker compose --profile infrastructure down`
3. **IMPORTANTE:** Hacer backup de PostgreSQL antes
4. Borrar volúmenes: `docker compose down -v` (⚠️ esto borra TODOS los datos)
5. Levantar con nuevas credenciales: `docker compose --profile infrastructure up -d`

## 📞 Contacto

Si hay problemas con la infraestructura:
1. Revisar esta documentación
2. Ver logs de servicios
3. Contactar al administrador del servidor

## 🏗️ Arquitectura

```
DigitalOcean (24.144.120.67)
┌─────────────────────────────────────────────┐
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │  MLflow      │──────│  PostgreSQL  │   │
│  │  :5001       │      │  :5432       │   │
│  └──────┬───────┘      └──────────────┘   │
│         │                                   │
│         │ guarda artifacts                  │
│         ↓                                   │
│  ┌──────────────┐                          │
│  │  MinIO (s3)  │                          │
│  │  API:9000    │                          │
│  │  UI:9001     │                          │
│  └──────────────┘                          │
│                                             │
│  Volúmenes persistentes:                   │
│  - postgres_data                           │
│  - minio_data                              │
└─────────────────────────────────────────────┘
         ↑                    ↑
         │                    │
         │                    │
    Airflow Local        RunPod Workers
    (orchestrator)       (training)
```

## ✅ Checklist de Salud del Sistema

- [ ] MLflow UI accesible en http://24.144.120.67:5001
- [ ] MinIO Console accesible en http://24.144.120.67:9001
- [ ] Buckets `mlflow` y `data` existen en MinIO
- [ ] PostgreSQL acepta conexiones
- [ ] Todos los contenedores en estado "running" (excepto create_buckets)
- [ ] `/health` de FastAPI responde `{"status":"healthy", ...}` en http://24.144.120.67:8800/health
- [ ] `/docs` de FastAPI accesible en http://24.144.120.67:8800/docs
- [ ] Espacio en disco > 20% disponible
- [ ] Logs sin errores críticos

