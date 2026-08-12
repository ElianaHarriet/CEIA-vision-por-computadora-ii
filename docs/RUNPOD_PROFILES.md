# Sistema de Perfiles RunPod

Este proyecto soporta múltiples cuentas de RunPod y Docker Hub mediante **perfiles**.

## 📋 Perfiles Disponibles

- **`luis`**: Docker Hub de Luis + RunPod de Luis
- **`santiago`**: Docker Hub de Santiago + RunPod de Santiago

## 🎯 Cómo Cambiar de Perfil

### 1. Editar el archivo de perfil

```bash
# Editar airflow/config/.runpod-profile
RUNPOD_PROFILE=luis   # o santiago
```

### 2. Commit y push

```bash
git add airflow/config/.runpod-profile
git commit -m "switch to luis profile"
git push
```

### 3. Reiniciar Airflow (solo si trabajás localmente)

```bash
docker compose --profile all down
docker compose --profile all up -d
```

**Listo.** Tanto tu Airflow local como GitHub Actions usan el perfil que pusiste en el archivo.

---

## 🚀 GitHub Actions

### Push Automático

Cuando hacés `git push`, GitHub Actions lee `airflow/config/.runpod-profile` y usa ese perfil para construir la imagen Docker.

### Ejecución Manual (Override Temporal)

Si querés construir con un perfil diferente **sin cambiar el archivo**:

1. GitHub → **Actions** → **"Build and push runpod_handler image"**
2. Click **"Run workflow"**
3. Seleccionar perfil del dropdown: `luis` o `santiago`
4. Click **"Run workflow"**

Esto **no modifica** el archivo, solo usa ese perfil para esa ejecución.

---

## 🔧 Configuración Inicial

### Variables Locales

Cada desarrollador necesita completar `airflow/secrets/variables.yaml`:

```yaml
# API Keys por perfil
RUNPOD_API_KEY_LUIS: "rpa_..."
RUNPOD_API_KEY: "rpa_..."
```

Los endpoint IDs ya están en `.env` (no tocar).

### GitHub Secrets/Variables

Ya configurados en el repositorio:

**Secrets:**
- `DOCKERHUB_USERNAME_LUIS` / `DOCKERHUB_TOKEN_LUIS`
- `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
- `RUNPOD_API_KEY_LUIS`
- `RUNPOD_API_KEY`

**Variables:**
- `DOCKERHUB_USERNAME_LUIS`
- `RUNPOD_ENDPOINT_ID_LUIS`
- `RUNPOD_ENDPOINT_ID_SANTIAGO`

---

## 🔍 Troubleshooting

### Error: "No endpoint ID found for profile"

Verificar que `airflow/config/.runpod-profile` tenga `RUNPOD_PROFILE=luis` o `RUNPOD_PROFILE=santiago`.

### Error: "Variable RUNPOD_API_KEY not found"

- Perfil `luis`: Necesita `RUNPOD_API_KEY_LUIS` en `airflow/secrets/variables.yaml`
- Perfil `santiago`: Necesita `RUNPOD_API_KEY` en `airflow/secrets/variables.yaml`

Reiniciar Airflow después de modificar `variables.yaml`.

---

## 💡 Ejemplos Rápidos

**Cambiar a perfil de Luis:**
```bash
echo "RUNPOD_PROFILE=luis" > airflow/config/.runpod-profile
git add airflow/config/.runpod-profile && git commit -m "use luis profile" && git push
docker compose --profile all restart  # solo si trabajás localmente
```

**Cambiar a perfil de Santiago:**
```bash
echo "RUNPOD_PROFILE=santiago" > airflow/config/.runpod-profile
git add airflow/config/.runpod-profile && git commit -m "use santiago profile" && git push
docker compose --profile all restart  # solo si trabajás localmente
```

**Ver perfil activo:**
```bash
cat airflow/config/.runpod-profile
```
