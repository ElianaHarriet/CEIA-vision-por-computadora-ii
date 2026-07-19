# Ejemplos

Esta carpeta contiene notebooks de ejemplo para aprender a usar la infraestructura de MLOps del proyecto.

## 📓 mlflow_quickstart.ipynb

Notebook introductorio que demuestra cómo:

- **Configurar las credenciales** de MinIO (S3) para que MLflow pueda guardar artefactos
- **Conectarse a MLflow** ejecutándose en `http://localhost:5001`
- **Crear un experimento** y configurar el tracking
- **Entrenar un modelo simple** (RandomForestRegressor con el dataset de diabetes)
- **Usar autolog** para registrar automáticamente métricas, parámetros y modelos

### Prerrequisitos

Antes de ejecutar este notebook, asegurarse de:

1. Tener todos los servicios ejecutándose con `docker compose --profile all up`
2. Verificar que MLflow esté accesible en http://localhost:5001
3. Verificar que MinIO esté accesible en http://localhost:9001

### Variables de entorno requeridas

El notebook configura automáticamente las credenciales de acceso a MinIO y la URL del endpoint S3. Estas variables están definidas en el archivo `.env` del proyecto.

### Ejecución del Notebook

1. **Crear y activar un entorno virtual (desde la raíz del proyecto):**
   ```bash
   # Crear entorno virtual
   python -m venv venv-car-damage
   
   # Activar en Windows
   venv-car-damage\Scripts\activate
   
   # Activar en Linux/Mac
   source venv-car-damage/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install jupyter mlflow scikit-learn boto3 python-dotenv
   ```

3. **Iniciar Jupyter:**
   ```bash
   # Opción 1: Abrir directamente el notebook
   jupyter notebook examples/mlflow_quickstart.ipynb
   
   # Opción 2: Abrir Jupyter en la carpeta examples
   jupyter notebook examples/
   ```

4. **Ejecutar el notebook:**
   - Si usaste la opción 1, el notebook se abre directamente
   - Si usaste la opción 2, navegar a `mlflow_quickstart.ipynb` en el navegador
   - Ejecutar las celdas en orden

5. **Detener Jupyter y desactivar el entorno (cuando se termine):**
   ```bash
   # Presionar Ctrl+C en la terminal donde se ejecuta Jupyter para detener el servidor
   # Luego desactivar el entorno virtual
   deactivate
   ```

**Nota:** El entorno virtual se crea en la raíz del proyecto para que pueda ser reutilizado si se agregan más notebooks o scripts en el futuro.

### Uso

**Nota:** También se puede abrir y ejecutar el notebook directamente en VS Code con la extensión de Jupyter.

1. Abrir el notebook con Jupyter o VS Code
2. Ejecutar las celdas en orden
3. Ver el experimento registrado en la interfaz web de MLflow

### Resultado esperado

Después de ejecutar el notebook, se podrá ver:

- Un experimento llamado `test_experiment` en MLflow
- Un run con métricas, parámetros y el modelo registrado
- Los artefactos del modelo guardados en el bucket `s3://mlflow/` en MinIO
