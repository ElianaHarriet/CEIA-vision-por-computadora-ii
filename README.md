# CEIA - Visión por Computadora II

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 👥 Integrantes

| 👤 Nombre | 📧 Email |
|-----------|----------|
| **Santiago Bartolini Rizzo** | [santiagobartolini@gmail.com](mailto:santiagobartolini@gmail.com) |
| **Luis Ali** | [aliluis@gmail.com](mailto:aliluis@gmail.com) |
| **Eliana Harriet** | [eharriet@fi.uba.ar](mailto:eharriet@fi.uba.ar) |


## 📝 Descripción

Repositorio para el curso de Visión por Computadora II de la Carrera de Especialización en Inteligencia Artificial (CEIA) - FIUBA.

### ⚙️ Servicios
- [Apache Airflow](https://airflow.apache.org/)
- [MLflow](https://mlflow.org/)
- API Rest para servir modelos ([FastAPI](https://fastapi.tiangolo.com/))
- [MinIO](https://min.io/)
- Base de datos relacional [PostgreSQL](https://www.postgresql.org/)
- Base de datos key-value [ValKey](https://valkey.io/) 

![Diagrama de servicios](final_assign.png)

Por defecto, cuando se inician los contenedores con Docker Compose, se crean los siguientes buckets:

- `s3://data`
- `s3://mlflow` (utilizado por MLflow para guardar los artefactos).

y las siguientes bases de datos:

- `mlflow_db` (utilizada por MLflow).
- `airflow` (utilizada por Airflow).

## 📦 Submódulo

El proyecto incluye el siguiente submódulo:

```
vision_computadora_II/  (rama: VpC2_2026)
└─ https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/vision_computadora_II
```

Para inicializarlo, ejecutar:

```bash
git submodule update --init --recursive
```

## 📊 Dataset

El proyecto compara dos arquitecturas de segmentación (instance vs semantic) usando el **mismo dataset** en diferentes formatos.

### Dataset: Car Damages (Segmentación Semántica)

Dataset original en formato de segmentación semántica, que se convierte automáticamente a ambos formatos necesarios.

**Fuente:** https://universe.roboflow.com/project-p5nyc/car-damages-v3gyz

**Clases del dataset:**

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | Minor Damage (Dent) | Abolladuras leves |
| 1 | Minor Damage (Scratch) | Rayones superficiales |
| 2 | No Damage | Sin daños visibles |
| 3 | Severe Damage | Daños severos/graves |

**Estadísticas:**
- Total: 2,320 imágenes
- Train: 1,970 imágenes
- Valid: 231 imágenes
- Test: 119 imágenes

**Formatos generados:**
- `instance/` → Para entrenar modelos de instance segmentation (YOLOv8-seg)
- `semantic/` → Para entrenar modelos de semantic segmentation (U-Net, DeepLab)

> 📖 **Ver [METODOLOGIA.md](METODOLOGIA.md)** para entender la metodología completa del proyecto y por qué se usa un solo dataset.


### 🔄 Configuración de Roboflow (Prerrequisito)

⚠️ **IMPORTANTE:** Completar esta configuración ANTES de la instalación. Los valores obtenidos aquí serán necesarios para el archivo `.env`.

Es necesario configurar una cuenta propia de Roboflow y forkear el dataset. Seguir estos pasos:

#### 1️⃣ Crear cuenta en Roboflow (si no tienes una)

- Ir a https://roboflow.com y registrarse
- Se creará automáticamente un workspace personal

#### 2️⃣ Obtener la API Key

- Ir a **Settings → API** o directamente a https://app.roboflow.com/settings/api
- Copiar la **Private API Key**
- ⚠️ **No compartir esta key con nadie** - es personal y privada
- Guardar este valor para el paso de instalación

#### 3️⃣ Forkear el Dataset (Car Damages)

- Ir al dataset original: https://universe.roboflow.com/project-p5nyc/car-damages-v3gyz
- Hacer clic en el botón **"Fork Dataset"** (arriba a la derecha)
- Esto crea una copia del dataset en el workspace personal
- El dataset forkeado aparecerá en el dashboard

#### 4️⃣ Generar la versión 1

⚠️ **IMPORTANTE:** Sin generar una versión, no se puede descargar el dataset programáticamente.

Después de forkear, el dataset no tiene versiones generadas. Es necesario crear la versión 1:

1. En el workspace personal, abrir el proyecto recién forkeado (ej: `car-damages-v3gyz-XXXXX`)
2. Ir a la pestaña **"Versions"** (en el menú lateral izquierdo bajo "DATA") o **"Generate"**
3. Hacer clic en **"Create New Version"**
4. Configurar preprocessing (opcional - usar valores por defecto):
   - Auto-Orient: ✅
   - Resize: 640x640 (recomendado para YOLO)
5. Configurar augmentations (opcional - se puede omitir o agregar según necesidad)
6. Hacer clic en **"Generate"**
7. Esperar a que se genere la versión (puede tardar unos minutos)
8. Una vez generada, aparecerá **"Version 1"** en la lista de versiones

#### 5️⃣ Obtener los identificadores del dataset

Una vez generada la versión, se necesitan 4 valores para configurar el `.env`:

**a) ROBOFLOW_WORKSPACE:**
- Es el nombre de usuario de Roboflow
- Se encuentra en la URL cuando se está en el dashboard
- Ejemplo: `https://app.roboflow.com/TU_WORKSPACE/...`
- También está visible en la esquina superior izquierda de Roboflow

**b) ROBOFLOW_PROJECT:**
- Es el ID único del proyecto forkeado
- Se encuentra en la URL cuando se abre el proyecto:
  ```
  https://app.roboflow.com/tu_workspace/car-damages-v3gyz-XXXXX/...
                                         ^^^^^^^^^^^^^^^^^^^^^^
                                         Este es el PROJECT_ID
  ```
- El sufijo `-XXXXX` es único para cada fork

**c) ROBOFLOW_VERSION:**
- Es el número de versión generada (normalmente `1`)
- Se puede ver en la pestaña "Versions" del proyecto

---

#### 🎯 Ejemplo de valores obtenidos

Si el workspace es `juan-perez-abc` y el dataset forkeado es `car-damages-v3gyz-9z8x7`:

```bash
# API Key
ROBOFLOW_API_KEY=AbCdEf123456GhIjKl789012

# Workspace
ROBOFLOW_WORKSPACE=juan-perez-abc

# Dataset (Car Damages)
ROBOFLOW_PROJECT=car-damages-v3gyz-9z8x7
ROBOFLOW_VERSION=1
```

⚠️ **Guardar estos 4 valores** - serán necesarios en el paso 3 de la instalación.

## 🚀 Instalación

1. Instalar [Docker](https://docs.docker.com/engine/install/) en la computadora (o en el servidor a utilizar).

2. Clonar este repositorio.

3. **Configurar el archivo `.env`:**
   - Solicitar el archivo `.env` al administrador del proyecto (se comparte por canal seguro)
   - Ubicar el archivo `.env` en la raíz del proyecto
   - Editar el archivo y completar las variables de Roboflow con los valores obtenidos en la sección [Configuración de Roboflow](#-configuración-de-roboflow-prerrequisito):
     ```bash
     # API Key y Workspace
     ROBOFLOW_API_KEY=tu_api_key_personal_aqui
     ROBOFLOW_WORKSPACE=tu_username_de_roboflow
     
     # Dataset (Car Damages)
     ROBOFLOW_PROJECT=car-damages-v3gyz-XXXXX
     ROBOFLOW_VERSION=1
     ```
   
   ⚠️ **Recordatorio:** Cada miembro usa su propio workspace y fork del dataset. No compartir API keys.
   
4. Crear las carpetas necesarias:
   ```bash
   # Carpetas de Airflow (dags ya existe en el repositorio)
   mkdir -p airflow/config airflow/logs airflow/plugins
   ```
   
   Nota: En Windows usar `mkdir` sin `-p`:
   ```cmd
   mkdir airflow\config
   mkdir airflow\logs
   mkdir airflow\plugins
   ```
   
   ℹ️ **Sobre `car_damage_detection/`:** Esta carpeta ya existe en el repositorio (con `.gitkeep`). Docker la montará como volumen y los DAGs de preparación de datos crearán automáticamente las subcarpetas `car-damages/` y `car-damage-detection/` cuando se ejecuten por primera vez.

5. En Linux o MacOS, en el archivo `.env`, reemplazar `AIRFLOW_UID` por el del usuario a utilizar (para encontrar el UID, utilizar el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no será posible subir DAGs (en `airflow/dags`) o plugins, etc.

6. En la carpeta raíz de este repositorio, ejecutar:

```bash
docker compose --profile all up
```

7. (Opcional, recomendado) Verificar que todos los servicios estén funcionando con el comando `docker ps -a` o revisar en Docker Desktop.

8. Acceder a los diferentes servicios mediante:
   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (ventana de administración de Buckets)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

Todos los puertos y otras configuraciones se pueden modificar en el archivo `.env`.

## 📂 Preparación de Datos

Una vez levantados los servicios, el primer paso es descargar y preparar el dataset ejecutando el DAG correspondiente:

### Descarga y Preparación del Dataset

1. **Acceder a Airflow**: http://localhost:8080
   Solicitar usuario y contraseña al administrador

2. **Ejecutar el DAG `data_preparation_semantic`**:
   - Buscar el DAG `data_preparation_semantic` en la lista
   - Hacer clic en el botón "▶️ Trigger DAG"
   - Esperar a que se complete (tarda varios minutos dependiendo de la conexión a internet)

3. **Verificar los datos descargados**:
   - Los datos se guardan automáticamente en `car_damage_detection/car-damages/`
   - Estructura generada:
     ```
     car_damage_detection/car-damages/
     ├── car-damages-forked/      # Datos raw de Roboflow (COCO format)
     └── car-damages-ready/        # Datos procesados
         ├── instance/             # YOLOv8 instance segmentation format
         │   ├── train/, valid/, test/
         │   │   ├── images/
         │   │   └── labels/
         │   └── data.yaml
         └── semantic/             # Máscaras PNG para semantic segmentation
             ├── train/, valid/, test/
             │   ├── images/
             │   └── masks/
     ```

**El DAG automáticamente:**
- ✅ Descarga el dataset desde Roboflow
- ✅ Convierte las anotaciones COCO a formato YOLOv8 (para instance segmentation)
- ✅ Genera máscaras PNG (para semantic segmentation)
- ✅ Valida la integridad de los datos
- ✅ Crea toda la estructura de carpetas necesaria

**Resultado:** Los datos quedan listos en **DOS formatos** para entrenar ambos modelos:
- `instance/` → Para entrenar YOLOv8-seg (instance segmentation)
- `semantic/` → Para entrenar U-Net/DeepLab (semantic segmentation)

> 📖 **Ver [METODOLOGIA.md](METODOLOGIA.md)** para entender cómo se usan estos dos formatos en el proyecto.

## 🧪 Ejemplos

Para verificar que la infraestructura funciona correctamente, se puede ejecutar el notebook de ejemplo:

- **[mlflow_quickstart.ipynb](examples/mlflow_quickstart.ipynb)** - Ejemplo introductorio que muestra cómo conectarse a MLflow, entrenar un modelo simple y registrar experimentos.

Ver [examples/README.md](examples/README.md) para más detalles sobre cómo usar los ejemplos.

## 🛑 Detener los servicios

Estos servicios ocupan cierta cantidad de memoria RAM y procesamiento, por lo que cuando no se utilizan, se recomienda detenerlos. Para hacerlo, ejecutar:

```bash
docker compose --profile all down
```

Si se desea no solo detenerlos, sino también eliminar toda la infraestructura (liberando espacio en disco), utilizar el siguiente comando:

```bash
docker compose down --rmi all --volumes
```

⚠️ **Nota:** Al ejecutar este comando, se perderá todo el contenido de los buckets y bases de datos.

## 📄 Licencia

Este proyecto está licenciado bajo Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
