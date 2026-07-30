FROM apache/airflow:3.0.2

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
         libgl1-mesa-glx \
         libglib2.0-0 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
# CPU-only wheels: este worker solo valida YOLO en CPU, el training real
# corre en RunPod. El índice default de PyPI para torch no es CPU-only.
RUN pip install --no-cache-dir torch~=2.0 torchvision~=0.15 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt
