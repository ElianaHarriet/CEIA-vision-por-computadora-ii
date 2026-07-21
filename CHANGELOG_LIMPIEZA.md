# 🧹 Changelog - Limpieza de Dataset B

**Fecha:** 2026-07-19

## 📋 Resumen de Cambios

Se realizó una limpieza del proyecto para reflejar la metodología correcta: usar **un solo dataset** (Car Damages) en dos formatos diferentes para comparar arquitecturas de segmentación (instance vs semantic).

## ✅ Cambios Realizados

### 1. 📄 Documentación Actualizada

#### `README.md`
- ✂️ Removida sección completa de "Dataset B (Car Damage Detection)"
- 📝 Simplificada sección de "Datasets" a solo "Dataset" (singular)
- 🔄 Actualizada sección de configuración de Roboflow (solo Dataset A)
- 📊 Simplificada sección de "Preparación de Datos" (solo un DAG)
- ➕ Agregada referencia a `METODOLOGIA.md` para explicación detallada

#### `METODOLOGIA.md` (NUEVO)
- ✨ Creado archivo completo explicando la metodología del proyecto
- 🎯 Explica por qué se usa un solo dataset
- 📊 Detalla los dos formatos generados (instance + semantic)
- 🤖 Describe los dos modelos a entrenar
- ⚠️ Aclara por qué NO se usa Dataset B
- 📈 Propone metodología de comparación (multiclase vs binaria)

#### `airflow/dags/README_DAGS.md` (NUEVO)
- 📋 Documentación de DAGs disponibles
- ✅ Lista DAG activo (`data_preparation_semantic_dag.py`)
- ⚠️ Lista DAG desactivado (`data_preparation_instance_dag.py.disabled`)
- 🔧 Instrucciones para reactivar si es necesario

### 2. 🔧 Configuración Actualizada

#### `.env`
- ✂️ Removidas variables de Dataset B:
  - `ROBOFLOW_PROJECT_INSTANCE`
  - `ROBOFLOW_VERSION_INSTANCE`
- ✅ Mantenidas solo variables de Dataset A (Car Damages)

#### `docker-compose.yaml`
- ✂️ Removidas variables de entorno de Dataset B del servicio Airflow
- ✅ Mantenidas solo variables de Dataset A

### 3. 🔄 DAGs

#### `data_preparation_semantic_dag.py`
- ✅ Mantenido sin cambios (DAG principal activo)
- 📊 Descarga y procesa Dataset A en ambos formatos

#### `data_preparation_instance_dag.py` → `.disabled`
- 🔒 Renombrado a `data_preparation_instance_dag.py.disabled`
- 📦 Preservado por si se necesita en el futuro
- ⚠️ Airflow no lo cargará (extensión `.disabled`)

### 4. 📁 Estructura de Archivos

#### Archivos Nuevos
```
METODOLOGIA.md                          # Explicación detallada de la metodología
CHANGELOG_LIMPIEZA.md                   # Este archivo
airflow/dags/README_DAGS.md             # Documentación de DAGs
```

#### Archivos Modificados
```
README.md                               # Simplificado (solo Dataset A)
.env                                    # Removidas variables Dataset B
docker-compose.yaml                     # Removidas variables Dataset B
```

#### Archivos Renombrados
```
data_preparation_instance_dag.py → data_preparation_instance_dag.py.disabled
```

## 🎯 Estado Actual del Proyecto

### ✅ Dataset A (Car Damages)
- **Estado:** Configurado y listo
- **Fuente:** https://universe.roboflow.com/project-p5nyc/car-damages-v3gyz
- **Clases:** 4 (Minor Damage Dent, Minor Damage Scratch, No Damage, Severe Damage)
- **Formatos preparados:**
  - `instance/` → Para YOLOv8-seg
  - `semantic/` → Para U-Net/DeepLab

### ❌ Dataset B (Car Damage Detection)
- **Estado:** Desactivado - NO SE USA
- **Razón:** Incompatibilidad de clases con Dataset A
- **Código:** Preservado en `.disabled` por si se necesita

## 📖 Próximos Pasos

1. ✅ **Preparación de datos:** Ejecutar DAG `data_preparation_semantic` en Airflow
2. 🤖 **Entrenamiento Modelo 1:** YOLOv8-seg con `instance/`
3. 🤖 **Entrenamiento Modelo 2:** U-Net/DeepLab con `semantic/`
4. 📊 **Comparación:** Evaluar y comparar resultados

## 🔄 Cómo Revertir (Si es necesario)

Si se necesita reactivar Dataset B:

1. Renombrar DAG:
   ```bash
   mv airflow/dags/data_preparation_instance_dag.py.disabled \
      airflow/dags/data_preparation_instance_dag.py
   ```

2. Restaurar variables en `.env`:
   ```bash
   ROBOFLOW_PROJECT_INSTANCE=car-damage-detection-ha5mm-dlfts
   ROBOFLOW_VERSION_INSTANCE=1
   ```

3. Restaurar variables en `docker-compose.yaml`:
   ```yaml
   ROBOFLOW_PROJECT_INSTANCE: ${ROBOFLOW_PROJECT_INSTANCE}
   ROBOFLOW_VERSION_INSTANCE: ${ROBOFLOW_VERSION_INSTANCE:-1}
   ```

4. Actualizar README con secciones de Dataset B

## 📚 Referencias

- **Recomendación del profesor:** Usar Dataset A (Car Damages) para ambos modelos
- **Metodología completa:** Ver `METODOLOGIA.md`
- **Documentación de DAGs:** Ver `airflow/dags/README_DAGS.md`

---

**Nota:** Esta limpieza NO elimina datos descargados previamente. Si ya se descargó Dataset B, permanecerá en `car_damage_detection/car-damage-detection/` pero no se usará.
