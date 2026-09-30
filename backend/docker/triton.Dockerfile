# Pin a reviewed NVIDIA Triton image digest/tag at build time.
ARG TRITON_IMAGE
FROM ${TRITON_IMAGE}
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 INTEL_I_BACKEND_PATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libmagic1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/source-requirements.txt
# Keep one OpenCV distribution. Windows magic package cannot install on Linux.
RUN sed '/^python-magic-bin/d;/^opencv-python==/d;/^paddlepaddle==/d' /tmp/source-requirements.txt | awk '!seen[$0]++' > /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -r /tmp/requirements.txt python-magic 'tritonclient[grpc]'
# Provision a reviewed GPU Paddle wheel matching the selected CUDA/Python image.
# The original CPU paddlepaddle dependency cannot satisfy strict CUDA OCR.
COPY docker/wheels/ /opt/intel-i-wheels/
ARG PADDLE_GPU_WHEEL
RUN test -n "$PADDLE_GPU_WHEEL" && python3 -m pip install --no-cache-dir "/opt/intel-i-wheels/$PADDLE_GPU_WHEEL"
COPY . /app
RUN python3 -m compileall -q /app && mkdir -p /app/object-cache /app/temp_uploads /app/outbox && chown -R 65532:65532 /app/object-cache /app/temp_uploads /app/outbox
ENV PYTHONPATH=/app
USER 65532:65532
EXPOSE 8000 8001 8002
ENTRYPOINT []
CMD ["tritonserver", "--model-repository=/app/triton_models", "--strict-readiness=true", "--model-control-mode=explicit", "--load-model=awiros_anpr_ocr", "--load-model=intel_i_frs", "--load-model=intel_i_pipeline"]
