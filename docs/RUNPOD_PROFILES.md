# Sistema de Perfiles RunPod

Este proyecto soporta múltiples cuentas de RunPod y Docker Hub mediante un **sistema de perfiles**.

## 📋 Perfiles Disponibles

- **`luis`**: Usa el RunPod y Docker Hub de Luis (endpoint: `zfk0phks97ucjj`)
- **`santiago`**: Usa el RunPod y Docker Hub de Santiago (endpoint: `lffm84c8mgc9k9`)

## 🎯 Cómo Usar

### 1️⃣ Seleccionar Perfil Localmente

Edita tu archivo `.env` local y cambia la variable `RUNPOD_PROFILE`:

```bash
# Para usar el RunPod de Luis (endpoint: zfk0phks97ucjj)
RUNPOD_PROFILE=luis

# Para usar el RunPod de Santiago (endpoint: lffm84c8mgc9k9)
RUNPOD_PROFILE=santiago
```

### 2️⃣ Configurar Credenciales

#### En `.env`:
```bash
# === PERFIL LUIS ===
RUNPOD_ENDPOINT_ID_LUIS=zfk0phks97ucjj

# === PERFIL SANTIAGO ===
RUNPOD_ENDPOINT_ID_SANTIAGO=lffm84c8mgc9k9
```

#### En `airflow/secrets/variables.yaml`:
```yaml
# API Keys por perfil
RUNPOD_API_KEY_LUIS: "rpa_..."
RUNPOD_API_KEY_SANTIAGO: "rpa_..."
```

### 3️⃣ Reiniciar Airflow

Después de cambiar el perfil:

```bash
docker-compose down
docker-compose up -d
```

### 4️⃣ Verificar Configuración

Al correr el DAG `training_instance_segmentation`, verás en los logs:

```
🎯 RunPod Profile: luis
   Endpoint ID: zfk0phks97ucjj
```

## 🚀 GitHub Actions

El workflow de CI/CD **automáticamente** selecciona el perfil correcto según quién hace el push:

- **LuisAli22** → Usa `DOCKERHUB_USERNAME_LUIS`, `RUNPOD_ENDPOINT_ID_LUIS`, `RUNPOD_API_KEY_LUIS`
- **Otros** → Usa `DOCKERHUB_USERNAME`, `RUNPOD_ENDPOINT_ID_SANTIAGO`, `RUNPOD_API_KEY_SANTIAGO`

### Secrets de GitHub Requeridos

#### Para Luis:
- `DOCKERHUB_USERNAME_LUIS` = `lali22`
- `DOCKERHUB_TOKEN_LUIS`
- `RUNPOD_API_KEY_LUIS`

#### Para Santiago:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `RUNPOD_API_KEY_SANTIAGO`

### Variables de GitHub Requeridas

- `RUNPOD_ENDPOINT_ID_LUIS` = `zfk0phks97ucjj`
- `RUNPOD_ENDPOINT_ID_SANTIAGO` = `lffm84c8mgc9k9`

## 👥 Para Eliana

Eliana puede elegir cualquier perfil editando `RUNPOD_PROFILE` en su `.env` local:

- `RUNPOD_PROFILE=santiago` → Usa crédito de Santiago
- `RUNPOD_PROFILE=luis` → Usa crédito de Luis

**No necesita configurar sus propios secrets**, solo cambiar la variable de perfil.

## 🔍 Troubleshooting

### Error: "No endpoint ID found for profile"
- Verifica que `RUNPOD_ENDPOINT_ID_<PROFILE>` exista en `.env`
- Verifica que `RUNPOD_PROFILE` sea `santiago` o `luis`

### Error: Variable RUNPOD_API_KEY not found
- Verifica que `RUNPOD_API_KEY_<PROFILE>` exista en `airflow/secrets/variables.yaml`
- Reinicia Airflow después de modificar `variables.yaml`

## 📝 Arquitectura

```
┌─────────────────┐
│  .env (local)   │
│  RUNPOD_PROFILE │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────┐
│  profile_config.py             │
│  get_endpoint_id()             │
│  get_dockerhub_image()         │
│  get_api_key() [via Variable]  │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  training_instance_dag.py      │
│  - Lee endpoint según perfil   │
│  - Imprime config activa       │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  runpod_client.py              │
│  - Lee API key según perfil    │
│  - Conecta al endpoint correcto│
└────────────────────────────────┘
```
