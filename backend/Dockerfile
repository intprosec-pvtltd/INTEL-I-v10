FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl openssl \
    ffmpeg libmagic1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    postgresql postgresql-client redis-server \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/models /app/temp_uploads /app/outbox /workspace \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
