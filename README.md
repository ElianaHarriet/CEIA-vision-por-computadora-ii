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

```
vision_computadora_II/  (rama: VpC2_2026)
└─ https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/vision_computadora_II
```

```bash
git submodule update --init --recursive
```

## 🚀 Instalación

1. Instalar [Docker](https://docs.docker.com/engine/install/) en la computadora (o en el servidor a utilizar).

2. Clonar este repositorio.

3. **Configurar el archivo `.env`:**
   - Solicitar el archivo `.env` al administrador del proyecto (se comparte por canal seguro)
   - Ubicar el archivo `.env` en la raíz del proyecto
   - **Cada miembro del equipo debe configurar sus propias credenciales de Roboflow:**
     
     a. Obtener tu API key personal:
        - Ir a https://app.roboflow.com/settings/api
        - Copiar tu API key
     
     b. Crear tu workspace (si no tenés uno):
        - El workspace es tu usuario de Roboflow
        - Si es tu primera vez, se crea automáticamente al registrarte
     
     c. Forkear el dataset original:
        - Ir a https://universe.roboflow.com/project-p5nyc/car-damages-v3gyz
        - Hacer clic en "Fork Dataset" para crear una copia en tu workspace
        - Una vez forkeado, generar la versión 1 aplicando preprocessing y augmentations
     
     d. Editar el archivo `.env` con tus valores:
        ```bash
        ROBOFLOW_API_KEY=tu_api_key_personal_aqui
        ROBOFLOW_WORKSPACE=tu_username_de_roboflow
        ROBOFLOW_PROJECT=car-damages-v3gyz-XXXXX  # ID del proyecto forkeado (ver en la URL)
        ROBOFLOW_VERSION=1
        ```
   
   - El `ROBOFLOW_PROJECT` se encuentra en la URL cuando abrís tu proyecto forkeado:
     `https://app.roboflow.com/tu_workspace/car-damages-v3gyz-XXXXX/...`
   
   ⚠️ **Nota importante:** Cada miembro usa su propio workspace y fork del dataset. No compartir API keys.
   
4. Crear las carpetas necesarias:
   ```bash
   # Carpetas de Airflow
   mkdir -p airflow/config airflow/dags airflow/logs airflow/plugins
   
   # Carpeta para datos (necesaria para el volumen de Docker)
   mkdir -p car_damage_detection
   ```
   
   Nota: En Windows usar `mkdir` sin `-p`:
   ```cmd
   mkdir airflow\config
   mkdir airflow\dags
   mkdir airflow\logs
   mkdir airflow\plugins
   mkdir car_damage_detection
   ```

5. En Linux o MacOS, en el archivo `.env`, reemplazar `AIRFLOW_UID` por el del usuario a utilizar (para encontrar el UID, utilizar el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no se podrá subir DAGs (en `airflow/dags`) o plugins, etc.

6. En la carpeta raíz de este repositorio, ejecutar:

```bash
docker compose --profile all up
```

7. (Opcional, recomendado): Verificar que todos los servicios estén funcionando con el comando `docker ps -a` o revisar en Docker Desktop.

8. Acceder a los diferentes servicios mediante:
   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (ventana de administración de Buckets)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

Todos los puertos y otras configuraciones se pueden modificar en el archivo `.env`.

## 📂 Preparación de Datos

Una vez levantados los servicios, el primer paso es descargar y preparar el dataset:

1. **Acceder a Airflow**: http://localhost:8080
   - Usuario: `airflow`
   - Contraseña: `airflow`

2. **Ejecutar el DAG `data_preparation`**:
   - Buscar el DAG `data_preparation` en la lista
   - Hacer clic en el botón "▶️ Trigger DAG"
   - Esperar a que se complete (tarda varios minutos dependiendo de la conexión a internet)

3. **Verificar los datos descargados**:
   - Los datos se guardan automáticamente en `car_damage_detection/data/`
   - Estructura generada:
     ```
     car_damage_detection/data/
     ├── car-damages-forked/      # Datos raw de Roboflow (COCO format)
     └── car-damages-ready/        # Datos procesados
         ├── instance/             # YOLOv8 instance segmentation format
         │   ├── train/, valid/, test/
         │   └── data.yaml
         └── semantic/             # Máscaras PNG para semantic segmentation
             ├── train/, valid/, test/
     ```

**El DAG automáticamente:**
- ✅ Descarga el dataset desde Roboflow
- ✅ Convierte las anotaciones COCO a formato YOLOv8
- ✅ Genera máscaras PNG para semantic segmentation
- ✅ Valida la integridad de los datos
- ✅ Crea toda la estructura de carpetas necesaria

⚠️ **Importante:** No es necesario crear manualmente ninguna carpeta. El DAG se encarga de todo.

## 🧪 Ejemplos

Para verificar que la infraestructura funciona correctamente, se puede ejecutar el notebook de ejemplo:

- **[mlflow_quickstart.ipynb](examples/mlflow_quickstart.ipynb)** - Ejemplo introductorio que muestra cómo conectarse a MLflow, entrenar un modelo simple y registrar experimentos.

Ver [examples/README.md](examples/README.md) para más detalles sobre cómo usar los ejemplos.

## 🛑 Apagar los servicios

Estos servicios ocupan cierta cantidad de memoria RAM y procesamiento, por lo que cuando no se están utilizando, se recomienda detenerlos. Para hacerlo, ejecutar:

```bash
docker compose --profile all down
```

Si se desea no solo detenerlos, sino también eliminar toda la infraestructura (liberando espacio en disco), utilizar el siguiente comando:

```bash
docker compose down --rmi all --volumes
```

Nota: Al hacer esto, se perderá todo en los buckets y bases de datos.

## 📊 Dataset

Car Damages forkeado en Roboflow. 4 clases:

| ID | Clase |
|----|-------|
| 0 | Minor Damage (Dent) |
| 1 | Minor Damage (Scratch) |
| 2 | No Damage |
| 3 | Severe Damage |



## 📄 Licencia

Este proyecto está licenciado bajo Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
