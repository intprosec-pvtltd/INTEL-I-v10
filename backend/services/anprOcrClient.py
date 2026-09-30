"""Internal client for the isolated Awiros/Paddle GPU OCR worker.

This module intentionally has no Paddle, PyTorch, Ultralytics, or ONNX Runtime
imports. It lets the camera/analytics process use the same ``recognize`` and
``health`` shape as the in-process Awiros adapter without loading Paddle into a
process that already owns a different cuDNN runtime.
"""
from __future__ import annotations

import hmac
import logging
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class RemoteAwirosOCR:
    version = "Awiros-ANPR-OCR/remote-v1"

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_seconds: float = 2.0,
        jpeg_quality: int = 92,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.token = str(token or "").strip()
        self.timeout_seconds = max(0.1, min(30.0, float(timeout_seconds)))
        self.jpeg_quality = max(70, min(100, int(jpeg_quality)))
        self.actual_device = "remote-uninitialized"
        self._health: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        return {"X-Intel-I-OCR-Token": self.token}

    def _validate_config(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("ANPR_OCR_REMOTE_URL must be a valid http(s) URL")
        if len(self.token) < 32:
            raise RuntimeError("ANPR_OCR_INTERNAL_TOKEN must be at least 32 characters")

    def load(self) -> "RemoteAwirosOCR":
        self._validate_config()
        try:
            response = requests.get(
                f"{self.base_url}/internal/health",
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Isolated Awiros OCR worker is unavailable at {self.base_url}"
            ) from exc

        if str(payload.get("status", "")).upper() != "READY":
            raise RuntimeError(f"Awiros OCR worker is not READY: {payload}")
        if not bool(payload.get("loaded", False)):
            raise RuntimeError(f"Awiros OCR worker did not load its model: {payload}")

        self.actual_device = str(payload.get("actual_device") or "remote")
        self._health = dict(payload)
        return self

    def recognize(self, plate_crop: np.ndarray) -> tuple[str | None, float]:
        if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
            return None, 0.0
        if plate_crop.ndim != 3 or plate_crop.shape[2] != 3:
            return None, 0.0

        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(plate_crop),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None, 0.0

        try:
            response = requests.post(
                f"{self.base_url}/internal/anpr/ocr",
                headers={
                    **self._headers(),
                    "Content-Type": "image/jpeg",
                },
                data=encoded.tobytes(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Remote Awiros OCR request failed: %s", exc)
            return None, 0.0

        text = str(payload.get("text") or "").strip().upper() or None
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        self.actual_device = str(payload.get("device") or self.actual_device)
        return text, confidence

    def health(self) -> dict[str, Any]:
        data = dict(self._health)
        data.update(
            {
                "engine": "awiros_remote",
                "model": "Awiros-ANPR-OCR",
                "loaded": bool(data.get("loaded", False)),
                "requested_device": "remote",
                "actual_device": self.actual_device,
                "remote_url": self.base_url,
            }
        )
        return data


__all__ = ["RemoteAwirosOCR"]
