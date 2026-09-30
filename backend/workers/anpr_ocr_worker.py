"""Isolated GPU worker for Awiros Indian ANPR OCR.

Run with:
    python -m workers.anpr_ocr_worker

Security/runtime constraints:
- Never import ``main``, ``vehicleANPR``, Ultralytics, or PyTorch here.
- This process owns PaddlePaddle and its CUDA/cuDNN runtime.
- Bind to loopback for local development. In Kubernetes, expose only through an
  internal ClusterIP/NetworkPolicy and keep the token in a Secret.
"""
from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status

from services.awirosOcr import AwirosOCR

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("intel_i.anpr_ocr_worker")

HOST = os.getenv("ANPR_OCR_WORKER_HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.getenv("ANPR_OCR_WORKER_PORT", "9201"))
TOKEN = os.getenv("ANPR_OCR_INTERNAL_TOKEN", "").strip()
MAX_IMAGE_BYTES = max(64 * 1024, min(10 * 1024 * 1024, int(os.getenv("ANPR_OCR_MAX_IMAGE_BYTES", str(2 * 1024 * 1024)))))

WEIGHTS = os.getenv("ANPR_AWIROS_MODEL_PATH", "models/awiros_anpr_ocr/model.safetensors")
DICTIONARY = os.getenv("ANPR_AWIROS_DICT_PATH", "models/awiros_anpr_ocr/en_dict.txt")
PADDLEOCR_DIR = os.getenv("ANPR_AWIROS_PADDLEOCR_DIR", "vendor/PaddleOCR")
DEVICE = os.getenv("ANPR_OCR_DEVICE", "cuda").strip().lower() or "cuda"
STRICT_DEVICE = os.getenv("ANPR_AWIROS_STRICT_DEVICE", "true").strip().lower() in {"1", "true", "yes", "on"}

_engine: AwirosOCR | None = None
_startup_error: str | None = None


def _authorize(provided: str | None) -> None:
    if len(TOKEN) < 32:
        raise HTTPException(status_code=503, detail="OCR worker token is not configured")
    if not provided or not hmac.compare_digest(provided, TOKEN):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _startup_error
    if len(TOKEN) < 32:
        raise RuntimeError("ANPR_OCR_INTERNAL_TOKEN must be at least 32 characters")

    logger.info(
        "INTEL-I isolated Awiros OCR worker starting | host=%s port=%s device=%s",
        HOST,
        PORT,
        DEVICE,
    )
    try:
        _engine = AwirosOCR(
            weights_path=WEIGHTS,
            dictionary_path=DICTIONARY,
            paddleocr_dir=PADDLEOCR_DIR,
            device=DEVICE,
            strict_device=STRICT_DEVICE,
        ).load()
        _startup_error = None
        logger.info("Awiros OCR worker READY | health=%s", _engine.health())
    except Exception as exc:
        _startup_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Awiros OCR worker failed to initialize")
        raise

    yield
    _engine = None


app = FastAPI(
    title="INTEL-I Internal Awiros OCR Worker",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/internal/health")
def health(x_intel_i_ocr_token: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(x_intel_i_ocr_token)
    if _engine is None:
        return {
            "status": "NOT_READY",
            "loaded": False,
            "actual_device": "uninitialized",
            "error": _startup_error,
        }
    return {"status": "READY", **_engine.health()}


@app.post("/internal/anpr/ocr")
async def recognize(
    request: Request,
    x_intel_i_ocr_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(x_intel_i_ocr_token)
    if _engine is None:
        raise HTTPException(status_code=503, detail="OCR engine is not ready")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Plate crop is too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")

    body_buffer = bytearray()
    async for chunk in request.stream():
        body_buffer.extend(chunk)
        if len(body_buffer) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Plate crop is too large")
    body = bytes(body_buffer)
    if not body:
        raise HTTPException(status_code=400, detail="Empty plate crop")
    if len(body) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Plate crop is too large")

    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid JPEG/PNG plate crop")
    if image.shape[0] * image.shape[1] > 8_000_000:
        raise HTTPException(status_code=413, detail="Plate image dimensions are too large")

    try:
        text, confidence = _engine.recognize(image)
    except Exception as exc:
        logger.exception("Awiros OCR inference failed")
        raise HTTPException(status_code=500, detail="OCR inference failed") from exc

    return {
        "text": text,
        "confidence": float(confidence),
        "engine": "Awiros-ANPR-OCR",
        "device": _engine.actual_device,
    }


def main() -> None:
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.getenv("ANPR_OCR_WORKER_LOG_LEVEL", "info").lower(),
        access_log=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
