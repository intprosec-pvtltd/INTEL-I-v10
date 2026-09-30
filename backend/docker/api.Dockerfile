FROM python:3.11-slim AS requirements
WORKDIR /build
COPY requirements.txt .
RUN sed '/^torch/d;/^torchvision/d;/^torchaudio/d;/^ultralytics/d;/^paddle/d;/^paddlex/d;/^onnxruntime-gpu/d;/^--extra-index-url/d' requirements.txt > api-requirements.txt \
    && printf '\nonnxruntime==1.20.2\n' >> api-requirements.txt

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 INTEL_I_PROCESS_ROLE=api
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libmagic1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=requirements /build/api-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY . .
RUN python -m compileall -q .
USER 65532:65532
EXPOSE 3000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
