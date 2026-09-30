FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INTEL_I_PROCESS_ROLE=anpr-worker

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       libgl1 \
       libglib2.0-0 \
       curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/all-requirements.txt

RUN sed \
      -e '/^torch==/d' \
      -e '/^torchvision==/d' \
      -e '/^torchaudio==/d' \
      -e '/^ultralytics==/d' \
      -e '/^onnxruntime/d' \
      -e '/^paddlepaddle==/d' \
      -e '/^--extra-index-url/d' \
      /tmp/all-requirements.txt > /tmp/requirements.txt \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && (python3 -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true) \
    && python3 -m pip install --no-cache-dir \
       paddlepaddle-gpu==3.3.1 \
       --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && rm -f /tmp/all-requirements.txt /tmp/requirements.txt

COPY . .

RUN python3 -m compileall -q .

USER 65532:65532

EXPOSE 9201

CMD ["python3", "-m", "workers.anpr_ocr_worker"]

