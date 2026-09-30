"""Database-backed, model-compatible watchlist synchronization for FRS pods."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np

# Import the primary model registry before advanced models so SQLAlchemy can
# resolve string relationships such as Incident during mapper configuration.
import db.model  # noqa: F401
from db.advanced_intelligence_model import PersonWatchlistEntry
from db.database import SessionLocal
from services.personRecognition import decrypt_bytes

logger = logging.getLogger(__name__)


class DatabaseWatchlistSynchronizer:
    """Periodically rebuild an FRS worker's in-memory index from PostgreSQL.

    Reference images are embedded by the worker's active recognition model.
    This avoids mixing legacy SFace vectors with AdaFace/other ONNX vectors.
    """

    def __init__(self, service: Any) -> None:
        self.service = service
        self.interval = max(2.0, min(float(os.getenv("FRS_WATCHLIST_SYNC_SECONDS", "15")), 3600.0))
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._fingerprint = ""
        self._lock = threading.Lock()
        self.last_sync_at: str | None = None
        self.last_error: str | None = None
        self.generation = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="frs-watchlist-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def request_reload(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                logger.exception("FRS watchlist synchronization failed")
            self._wake.wait(self.interval)
            self._wake.clear()

    @staticmethod
    def _row_fingerprint(rows: list[PersonWatchlistEntry]) -> str:
        digest = hashlib.sha256()
        for row in rows:
            updated = row.updated_at.isoformat() if isinstance(row.updated_at, datetime) else str(row.updated_at or "")
            digest.update(f"{row.id}:{row.user_id}:{row.status}:{updated}:{row.reference_image_size or 0}".encode())
        return digest.hexdigest()

    def refresh(self, *, force: bool = False) -> bool:
        if self.service.embedder is None:
            raise RuntimeError("FRS embedder is not loaded")
        with self._lock, SessionLocal() as db:
            rows = (
                db.query(PersonWatchlistEntry)
                .filter(PersonWatchlistEntry.status == "ACTIVE")
                .order_by(PersonWatchlistEntry.id)
                .all()
            )
            fingerprint = self._row_fingerprint(rows)
            if not force and fingerprint == self._fingerprint:
                self.last_sync_at = datetime.utcnow().isoformat() + "Z"
                self.last_error = None
                return False

            entries: list[tuple[str, np.ndarray, dict[str, Any]]] = []
            for row in rows:
                try:
                    if not row.reference_image_data:
                        continue
                    raw = decrypt_bytes(row.reference_image_data)
                    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.size == 0:
                        continue
                    face = self.service.reference_face_crop(image)
                    vector = self.service.embedder.embed(face)
                    entries.append((str(row.id), vector, {
                        "user_id": int(row.user_id),
                        "full_name": row.full_name,
                        "category": row.category,
                    }))
                except Exception:
                    logger.exception("FRS watchlist entry sync failed | entry_id=%s", row.id)

            self.service.watchlist.replace(entries)
            self._fingerprint = fingerprint
            self.generation += 1
            self.last_sync_at = datetime.utcnow().isoformat() + "Z"
            self.last_error = None
            logger.info("FRS watchlist synchronized | entries=%s generation=%s", len(entries), self.generation)
            return True

    def status(self) -> dict[str, Any]:
        return {
            "entries": self.service.watchlist.size(),
            "generation": self.generation,
            "last_sync_at": self.last_sync_at,
            "last_error": self.last_error,
            "sync_seconds": self.interval,
        }
