# Troubleshooting: Airflow-RunPod en Windows

## Problema: Pérdida de conexión entre Airflow y RunPod

Cuando Airflow corre en Docker Desktop sobre Windows, pueden aparecer problemas de red que causan que el DAG falle con "network errors" aunque el job en RunPod sigue ejecutándose normalmente.

### Causas raíz en Windows

1. **Docker Desktop usa virtualización**: Los contenedores corren en WSL2 o Hyper-V, no nativamente en Windows
2. **NAT y Firewall**: Windows Firewall o el NAT de Docker Desktop pueden cortar conexiones HTTP largas (>30 min)
3. **DNS inestable**: Resolución de nombres puede fallar intermitentemente desde contenedores
4. **Timeouts TCP agresivos**: Windows tiene timeouts más cortos que Linux para conexiones inactivas

### Síntomas

- El DAG en Airflow marca la tarea `train_yolov8_model` como `failed`
- Los logs muestran errores como:
  - `Network error X/20 (will retry in Xs): ...`
  - `ConnectionError`, `Timeout`, o `RequestException`
- **Pero** el job en RunPod sigue corriendo y eventualmente completa exitosamente
- El problema aparece después de 30-60 minutos de polling continuo

### Soluciones aplicadas en el código

El código en `runpod_client.py` ya implementa las siguientes mejoras:

1. **Reintentos aumentados**: 20 errores consecutivos permitidos (antes: 5)
2. **Backoff exponencial**: Espera progresiva entre reintentos (10s → 20s → 40s → 80s → 120s)
3. **Keep-alive headers**: Headers HTTP para mantener la conexión abierta
4. **Connection pooling**: Reutilización de conexiones TCP
5. **Timeouts configurables**: Ajustables vía variables de entorno
6. **Logs detallados**: Distingue entre timeout, connection error, y otros errores

### Configuración recomendada para Windows

Editá `.env` y ajustá estos valores:

```bash
# Intervalo de polling más largo para reducir requests
RUNPOD_POLL_INTERVAL_S=60  # Default: 45

# Timeout total aumentado (4 horas)
RUNPOD_POLL_TIMEOUT_S=14400  # Default: 10800 (3h)
```

Luego reiniciá Docker:
```bash
docker compose --profile all down
docker compose --profile all up -d
```

### Configuración de Docker Desktop

1. **Aumentar recursos asignados**:
   - Docker Desktop → Settings → Resources
   - Aumentar Memory a mínimo 4GB
   - Aumentar CPUs a mínimo 2

2. **Verificar backend**:
   - Docker Desktop → Settings → General
   - Asegurar que "Use WSL 2 based engine" esté **activado**
   - WSL2 es más estable que Hyper-V para networking

3. **Desactivar VPN o Proxy**:
   - Si usás VPN corporativa, puede interferir con el networking de Docker
   - Probá desactivarla temporalmente durante el entrenamiento

### Configuración de Windows Firewall

Si el problema persiste, puede ser necesario crear reglas en Windows Firewall:

1. Abrir PowerShell como Administrador
2. Ejecutar:
```powershell
# Permitir tráfico saliente de Docker a RunPod API
New-NetFirewallRule -DisplayName "Docker RunPod API" -Direction Outbound -Action Allow -RemoteAddress Any -RemotePort 443 -Protocol TCP
```

### Verificación de DNS

Verificá que Docker pueda resolver `api.runpod.ai`:

```bash
docker compose exec airflow-worker ping -c 4 api.runpod.ai
docker compose exec airflow-worker nslookup api.runpod.ai
```

Si falla, agregá DNS público a `docker-compose.yaml` (ya está configurado):
```yaml
dns:
  - 8.8.8.8  # Google DNS
  - 8.8.4.4
```

### Monitoreo durante entrenamiento

Para verificar que el polling sigue funcionando:

1. **Ver logs de Airflow Worker**:
```bash
docker compose logs -f airflow-worker
```

Deberías ver cada ~45 segundos:
```
RunPod job xyz status: IN_PROGRESS (elapsed: 1800s)
✓ RunPod polling healthy: 40 successful polls
```

2. **Ver status del job en RunPod Console**:
   - https://www.runpod.io/console/serverless
   - Verificar que el job está "Running" y no "Cancelled"

### Si el DAG falla

Si el DAG falla por red pero el job completó en RunPod:

1. **No relances el DAG** (crearía un segundo job y duplicaría costos)
2. Anotá el `run_id` de MLflow desde los logs del worker
3. Registrá manualmente el modelo:

```python
# Desde un notebook o script local
import mlflow
mlflow.set_tracking_uri("http://localhost:5001")
run_id = "el-run-id-del-log"
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "car-damage-instance-segmentation")
```

### Alternativas si el problema persiste

1. **Ejecutar Airflow en Linux**:
   - Usar WSL2 directamente (sin Docker Desktop)
   - O una VM Linux con VirtualBox/Hyper-V

2. **Reducir duración del training**:
   - Menos epochs: `YOLO_EPOCHS=50` en lugar de 150
   - Modelo más chico: `YOLO_MODEL=yolov8m-seg.pt` en lugar de `yolov8x-seg.pt`

3. **Async polling** (requiere cambio de arquitectura):
   - Submitear job y terminar tarea
   - Otra tarea periódica verifica status cada 5 minutos
   - Más complejo pero más resiliente a desconexiones

### Debugging avanzado

Si querés ver exactamente dónde falla la conexión:

```bash
# Logs detallados de requests HTTP
docker compose exec airflow-worker python << 'EOF'
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.DEBUG)

from training.runpod_client import RunPodClient
client = RunPodClient("tu-endpoint-id")
status = client.get_status("tu-job-id")
print(status)
EOF
```

Esto mostrará todos los detalles de la conexión TCP, SSL handshake, DNS resolution, etc.
