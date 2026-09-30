from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np

from services.plateFusion import normalize_plate

logger = logging.getLogger("ollama-vision")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class OllamaPlateVerifier:
    """Bounded advisory verifier. Never blocks the camera worker."""

    def __init__(self):
        self.enabled = _env_bool("OLLAMA_VISION_ENABLED", False)
        self.model = os.getenv("OLLAMA_VISION_MODEL", "gemma4:12b").strip()[:150]
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        self.timeout = max(2.0, min(60.0, float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "15"))))
        self.connect_timeout = max(1.0, min(10.0, float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "3"))))
        self.result_ttl = max(10.0, min(600.0, float(os.getenv("OLLAMA_RESULT_TTL_SECONDS", "90"))))
        self.max_image_dimension = max(256, min(4096, int(os.getenv("OLLAMA_MAX_IMAGE_DIMENSION", "1600"))))
        self.failure_threshold = max(2, min(20, int(os.getenv("OLLAMA_CIRCUIT_FAILURES", "5"))))
        self.circuit_seconds = max(5.0, min(300.0, float(os.getenv("OLLAMA_CIRCUIT_SECONDS", "30"))))
        self.low_threshold = max(0.0, min(1.0, float(os.getenv("OLLAMA_ANPR_LOW_CONFIDENCE", "0.72"))))
        self.cooldown = max(2.0, min(300.0, float(os.getenv("OLLAMA_ANPR_COOLDOWN_SECONDS", "15"))))
        workers = max(1, min(4, int(os.getenv("OLLAMA_WORKERS", "1"))))
        queue_limit = max(workers, min(64, int(os.getenv("OLLAMA_QUEUE_LIMIT", "8"))))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ollama-anpr")
        self._slots = threading.BoundedSemaphore(queue_limit)
        self._lock = threading.RLock()
        self._last_submit: dict[str, float] = {}
        self._results: dict[str, dict] = {}
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._last_health_error: str | None = None
        if self.enabled:
            self._validate_endpoint()

    def _validate_endpoint(self):
        parsed = urlparse(self.base_url)
        allowed = {item.strip().lower() for item in os.getenv("OLLAMA_ALLOWED_HOSTS", "127.0.0.1,localhost,ollama").split(",") if item.strip()}
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.lower() not in allowed:
            raise RuntimeError("OLLAMA_BASE_URL host is not allowed")

    def probe_model(self) -> dict:
        """Truthful non-fatal startup check for Ollama and configured model."""
        if not self.enabled:
            return {"enabled": False, "available": False, "model": self.model, "reason": "disabled"}
        try:
            timeout = httpx.Timeout(min(self.timeout, 8.0), connect=self.connect_timeout)
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
            names = {str(item.get("name") or item.get("model") or "").strip() for item in (payload.get("models") or []) if isinstance(item, dict)}
            configured = self.model
            # Deployment readiness is intentionally exact. A different tag can
            # represent different weights/quantization and must not satisfy the
            # configured production model contract.
            available = configured in names
            self._last_health_error = None if available else "MODEL_NOT_FOUND"
            return {"enabled": True, "available": available, "model": configured, "reason": None if available else "model_not_found"}
        except Exception as exc:
            self._last_health_error = type(exc).__name__
            return {"enabled": True, "available": False, "model": self.model, "reason": type(exc).__name__}

    def should_verify(self, confidence: float, *, uncertain: bool = False) -> bool:
        if not self.enabled or time.monotonic() < self._circuit_open_until:
            return False
        confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        return bool(uncertain or confidence < self.low_threshold)

    def submit(self, key: str, image: np.ndarray, *, current_candidates: list[dict] | None = None) -> bool:
        if not self.enabled or not isinstance(image, np.ndarray) or image.size == 0:
            return False
        if time.monotonic() < self._circuit_open_until:
            return False
        h, w = image.shape[:2]
        if max(h, w) > self.max_image_dimension:
            scale = self.max_image_dimension / float(max(h, w))
            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        self._cleanup()
        safe_key = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(key))[:200]
        now = time.monotonic()
        with self._lock:
            if now - self._last_submit.get(safe_key, 0.0) < self.cooldown:
                return False
        if not self._slots.acquire(blocking=False):
            return False
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok or len(encoded) > 2_000_000:
            self._slots.release()
            return False
        with self._lock:
            self._last_submit[safe_key] = now
        future = self._executor.submit(self._verify, safe_key, encoded.tobytes(), current_candidates or [])
        future.add_done_callback(lambda _: self._slots.release())
        return True

    def get_result(self, key: str, *, max_age_seconds: float | None = None) -> dict | None:
        safe_key = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(key))[:200]
        max_age_seconds = self.result_ttl if max_age_seconds is None else max(1.0, float(max_age_seconds))
        with self._lock:
            value = self._results.get(safe_key)
            if not value or time.time() - float(value.get("received_at", 0.0)) > max_age_seconds:
                return None
            return dict(value)

    def _verify(self, key: str, image: bytes, candidates: list[dict]):
        prompt = (
            "Read only the Indian vehicle registration plate in this crop. "
            "Do not guess invisible characters. Return strict JSON with keys "
            "plate_text, confidence, readable, alternatives, uncertain_characters. "
            f"Other OCR candidates for comparison: {json.dumps(candidates[:8], ensure_ascii=True)}"
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image).decode("ascii")]}],
            "options": {"temperature": 0, "num_predict": 220},
        }
        try:
            timeout = httpx.Timeout(self.timeout, connect=self.connect_timeout)
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                outer = response.json()
            raw = outer.get("message", {}).get("content", "")
            if not isinstance(raw, str) or len(raw) > 5000:
                return
            parsed = json.loads(raw)
            plate = normalize_plate(parsed.get("plate_text"))
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            if not parsed.get("readable") or len(plate) < 6:
                return
            result = {
                "text": plate,
                "confidence": confidence,
                "source": "ollama",
                "model": self.model,
                "alternatives": [normalize_plate(v) for v in (parsed.get("alternatives") or [])[:5]],
                "uncertain_characters": [str(v)[:2] for v in (parsed.get("uncertain_characters") or [])[:8]],
                "received_at": time.time(),
            }
            with self._lock:
                self._results[key] = result
                self._consecutive_failures = 0
                self._last_health_error = None
        except Exception as exc:
            with self._lock:
                self._consecutive_failures += 1
                self._last_health_error = type(exc).__name__
                if self._consecutive_failures >= self.failure_threshold:
                    self._circuit_open_until = time.monotonic() + self.circuit_seconds
            logger.warning("Ollama plate verification failed | key=%s error=%s", key, type(exc).__name__)

    def _cleanup(self) -> None:
        cutoff = time.time() - self.result_ttl
        now_mono = time.monotonic()
        with self._lock:
            for key in [k for k, v in self._results.items() if float(v.get("received_at", 0.0)) < cutoff]:
                self._results.pop(key, None)
            for key in [k for k, ts in self._last_submit.items() if now_mono - ts > max(self.result_ttl, self.cooldown * 4)]:
                self._last_submit.pop(key, None)

    def health(self) -> dict:
        self._cleanup()
        return {"enabled": self.enabled, "model": self.model, "cached_results": len(self._results),
                "circuit_open": time.monotonic() < self._circuit_open_until,
                "consecutive_failures": self._consecutive_failures, "last_error": self._last_health_error}

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


ollama_plate_verifier = OllamaPlateVerifier()
