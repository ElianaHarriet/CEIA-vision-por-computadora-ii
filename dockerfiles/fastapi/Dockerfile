FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         curl \
         libgl1 \
         libglib2.0-0 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY dockerfiles/fastapi/requirements.txt ./requirements.txt

RUN pip install --upgrade pip
# CPU-only wheels: el droplet de DigitalOcean es CPU-only.
RUN pip install --no-cache-dir torch~=2.0 torchvision~=0.15 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY dockerfiles/fastapi/app.py ./app.py
COPY airflow/dags/evaluation ./evaluation

EXPOSE 8800
