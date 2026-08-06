# Build context = raíz del repo (necesario para poder copiar airflow/dags/training).
#
# IMPORTANT: la imagen base trae torch preinstalado ya compilado contra su
# propio CUDA — no lo reinstalamos con un --index-url fijo para evitar un
# mismatch entre el CUDA de la imagen y el driver del host de RunPod (eso
# produce errores silenciosos o pesos incompatibles en runtime, no solo
# errores de build). Si se cambia de template en RunPod, ajustar solo el tag
# de este FROM (ver tags vigentes en https://hub.docker.com/r/runpod/pytorch/tags).
FROM runpod/pytorch:1.1.0-rc.154-cu1281-torch260-ubuntu2204

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         libgl1-mesa-glx \
         libglib2.0-0 \
  && (apt-get purge -yqq --autoremove python3-blinker || true) \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY runpod_handler/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY airflow/dags/training /app/training
COPY runpod_handler/handler.py /app/handler.py
COPY runpod_handler/dataset_fetcher.py /app/dataset_fetcher.py

# Test that Python can at least import the handler module
RUN python -c "import sys; sys.path.insert(0, '/app'); import handler; print('Handler module imported successfully')"

CMD ["python", "-u", "/app/handler.py"]
