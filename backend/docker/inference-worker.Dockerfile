FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INTEL_I_PROCESS_ROLE=inference-worker \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       software-properties-common \
       ca-certificates \
       curl \
       ffmpeg \
       libgl1 \
       libglib2.0-0 \
       libmagic1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 \
       python3.11-dev \
       python3.11-venv \
    && python3.11 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

RUN python --version \
    && python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/all-requirements.txt

RUN sed \
      -e '/^torch==/d' \
      -e '/^torchvision==/d' \
      -e '/^torchaudio==/d' \
      -e '/^paddleocr==/d' \
      -e '/^paddlepaddle==/d' \
      -e '/^paddlex==/d' \
      -e '/^--extra-index-url/d' \
      /tmp/all-requirements.txt > /tmp/requirements.txt

RUN python -m pip install --no-cache-dir \
       torch==2.5.1 \
       torchvision==0.20.1 \
       torchaudio==2.5.1 \
       --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/all-requirements.txt /tmp/requirements.txt

COPY . .

RUN python -m compileall -q .

USER 65532:65532

EXPOSE 9300

CMD ["python", "-m", "workers.inference_worker"]
