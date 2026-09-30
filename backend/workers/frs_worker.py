"""Isolated ONNX Runtime GPU face-recognition service."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("intel_i.frs_worker")


# ============================================================================
# Environment helpers
# ============================================================================


def _env_flag(name: str, default: bool = False) -> bool:
    default_value = "true" if default else "false"

    return os.getenv(name, default_value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================================
# Windows CUDA / cuDNN DLL bootstrap
#
# IMPORTANT:
# This block MUST execute before importing faceEmbedder because faceEmbedder
# imports/creates ONNX Runtime sessions.
#
# PyTorch already ships the CUDA 12 + cuDNN 9 DLLs required by the current
# Intel-I Windows environment. We expose torch\lib to Windows DLL discovery
# before ONNX Runtime initializes CUDAExecutionProvider.
# ============================================================================


_DLL_DIRECTORY_HANDLES: list[Any] = []


def _bootstrap_cuda_runtime() -> dict[str, Any]:
    state: dict[str, Any] = {
        "platform": sys.platform,
        "enabled": True,
        "torch_available": False,
        "cuda_available": False,
        "gpu_name": None,
        "torch_cuda_version": None,
        "cudnn_version": None,
        "torch_lib": None,
        "dll_directory_added": False,
        "path_added": False,
        "error": None,
    }

    if not _env_flag("FRS_CUDA_DLL_BOOTSTRAP_ENABLED", True):
        state["enabled"] = False
        return state

    try:
        # Import torch BEFORE anything that can initialize ONNX Runtime.
        import torch

        state["torch_available"] = True
        state["torch_cuda_version"] = torch.version.cuda

        try:
            state["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            state["cudnn_version"] = None

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        state["torch_lib"] = str(torch_lib)

        if sys.platform == "win32" and torch_lib.is_dir():

            # --------------------------------------------------------------
            # Windows Python >= 3.8 DLL search directory
            #
            # IMPORTANT:
            # Keep the returned handle alive for the lifetime of the process.
            # Otherwise Windows may remove the directory from DLL search.
            # --------------------------------------------------------------

            if hasattr(os, "add_dll_directory"):
                handle = os.add_dll_directory(str(torch_lib))
                _DLL_DIRECTORY_HANDLES.append(handle)

                state["dll_directory_added"] = True

            # --------------------------------------------------------------
            # PATH fallback
            #
            # Some transitive CUDA/cuDNN dependencies still depend on the
            # regular Windows DLL lookup mechanism.
            # --------------------------------------------------------------

            current_path = os.environ.get("PATH", "")

            existing_paths = {
                entry.strip().rstrip("\\/").lower()
                for entry in current_path.split(os.pathsep)
                if entry.strip()
            }

            normalized_torch_lib = str(torch_lib).rstrip("\\/").lower()

            if normalized_torch_lib not in existing_paths:
                os.environ["PATH"] = (
                    str(torch_lib)
                    + os.pathsep
                    + current_path
                )

                state["path_added"] = True

        # --------------------------------------------------------------
        # Verify PyTorch itself can see CUDA.
        # --------------------------------------------------------------

        state["cuda_available"] = bool(torch.cuda.is_available())

        if state["cuda_available"]:
            try:
                state["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                state["gpu_name"] = "CUDA device 0"

    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"

    return state


CUDA_BOOTSTRAP_STATE = _bootstrap_cuda_runtime()


# ============================================================================
# Third-party imports
#
# These imports deliberately occur AFTER CUDA bootstrap.
# ============================================================================

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)


# ============================================================================
# INTEL-I FRS imports
#
# faceEmbedder may initialize/use ONNX Runtime. It therefore MUST remain
# below _bootstrap_cuda_runtime().
# ============================================================================

from services.frs.faceEmbedder import OnnxFaceEmbedder
from services.frs.faceQuality import FaceQualityGate
from services.frs.schemas import FaceEmbedding
from services.frs.trackFusion import TrackEmbeddingFusion
from services.frs.watchlistSearch import (
    TemporalConfirmation,
    WatchlistIndex,
)
from services.frs.watchlistSync import DatabaseWatchlistSynchronizer


# ============================================================================
# FRS Service
# ============================================================================


class FRSService:
    def __init__(self) -> None:
        self.embedder: OnnxFaceEmbedder | None = None

        self.quality = FaceQualityGate(
            min_size=int(
                os.getenv(
                    "FRS_MIN_FACE_SIZE",
                    "40",
                )
            ),
            min_blur=float(
                os.getenv(
                    "FRS_MIN_BLUR",
                    "20",
                )
            ),
            min_detector_confidence=float(
                os.getenv(
                    "FRS_MIN_DETECTOR_CONFIDENCE",
                    "0.70",
                )
            ),
        )

        self.fusion = TrackEmbeddingFusion(
            max_observations=int(
                os.getenv(
                    "FRS_FUSION_MAX_OBSERVATIONS",
                    "8",
                )
            ),
            ttl_seconds=float(
                os.getenv(
                    "FRS_TRACK_TTL_SECONDS",
                    "15",
                )
            ),
        )

        self.watchlist = WatchlistIndex()

        self.confirmation = TemporalConfirmation(
            min_similarity=float(
                os.getenv(
                    "FRS_MATCH_THRESHOLD",
                    "0.55",
                )
            ),
            confirmations=int(
                os.getenv(
                    "FRS_CONFIRMATIONS_REQUIRED",
                    "3",
                )
            ),
            window_seconds=float(
                os.getenv(
                    "FRS_CONFIRMATION_WINDOW_SECONDS",
                    "5",
                )
            ),
        )

        self.error: str | None = None

        self.sync = DatabaseWatchlistSynchronizer(self)

        self.detector = None
        self._detector_lock = threading.Lock()

        self.gpu_validation: dict[str, Any] = {
            "validated": False,
            "bootstrap": CUDA_BOOTSTRAP_STATE,
        }

    # ========================================================================
    # Startup / model loading
    # ========================================================================

    def load(self) -> None:
        logger.info(
            "FRS CUDA bootstrap | platform=%s torch=%s cuda=%s gpu=%s "
            "torch_cuda=%s cudnn=%s torch_lib=%s dll_directory=%s path_added=%s",
            CUDA_BOOTSTRAP_STATE.get("platform"),
            CUDA_BOOTSTRAP_STATE.get("torch_available"),
            CUDA_BOOTSTRAP_STATE.get("cuda_available"),
            CUDA_BOOTSTRAP_STATE.get("gpu_name"),
            CUDA_BOOTSTRAP_STATE.get("torch_cuda_version"),
            CUDA_BOOTSTRAP_STATE.get("cudnn_version"),
            CUDA_BOOTSTRAP_STATE.get("torch_lib"),
            CUDA_BOOTSTRAP_STATE.get("dll_directory_added"),
            CUDA_BOOTSTRAP_STATE.get("path_added"),
        )

        bootstrap_error = CUDA_BOOTSTRAP_STATE.get("error")

        if bootstrap_error:
            logger.warning(
                "FRS CUDA bootstrap reported an error | %s",
                bootstrap_error,
            )

        # --------------------------------------------------------------------
        # Face embedding model
        # --------------------------------------------------------------------

        path = Path(
            os.getenv(
                "FACE_RECOGNIZER_MODEL_PATH",
                "models/face_recognition.onnx",
            )
        )

        if not path.is_file():
            raise RuntimeError(
                f"FRS embedding model is missing: {path}"
            )

        logger.info(
            "Loading FRS embedding model | path=%s",
            path,
        )

        self.embedder = OnnxFaceEmbedder(
            str(path),
            [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        # --------------------------------------------------------------------
        # IMPORTANT:
        # get active providers from the ACTUAL model session.
        # --------------------------------------------------------------------

        providers = list(
            self.embedder.active_providers or []
        )

        logger.info(
            "FRS ONNX active providers | providers=%s",
            providers,
        )

        cuda_required = _env_flag(
            "FRS_CUDA_REQUIRED",
            True,
        )

        if cuda_required:
            if not providers:
                raise RuntimeError(
                    "FRS ONNX session has no active execution providers"
                )

            if providers[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    "CUDAExecutionProvider is not primary: "
                    f"{providers}"
                )

        # --------------------------------------------------------------------
        # Face detector
        # --------------------------------------------------------------------

        detector_path = Path(
            os.getenv(
                "FACE_DETECTOR_MODEL_PATH",
                "models/face_detection_yunet_2023mar.onnx",
            )
        )

        if detector_path.is_file():
            detector_score_threshold = float(
                os.getenv(
                    "FRS_DETECTOR_SCORE_THRESHOLD",
                    "0.70",
                )
            )

            detector_nms_threshold = float(
                os.getenv(
                    "FRS_DETECTOR_NMS_THRESHOLD",
                    "0.30",
                )
            )

            detector_top_k = int(
                os.getenv(
                    "FRS_DETECTOR_TOP_K",
                    "5000",
                )
            )

            self.detector = cv2.FaceDetectorYN.create(
                str(detector_path),
                "",
                (320, 320),
                detector_score_threshold,
                detector_nms_threshold,
                detector_top_k,
            )

            logger.info(
                "FRS YuNet detector loaded | path=%s",
                detector_path,
            )

        else:
            self.detector = None

            logger.warning(
                "FRS face detector model not found | path=%s",
                detector_path,
            )

        # --------------------------------------------------------------------
        # Actual CUDA inference validation
        # --------------------------------------------------------------------

        self._validate_inference()

        logger.info(
            "FRS inference validation passed | %s",
            self.gpu_validation,
        )

        # --------------------------------------------------------------------
        # Watchlist synchronization
        # --------------------------------------------------------------------
        #
        # Perform an initial synchronous refresh so readiness never reports
        # stale or empty state simply because the background synchronization
        # thread has not executed yet.
        # --------------------------------------------------------------------

        self.sync.refresh(force=True)

        self.sync.start()

        self.error = None

        logger.info(
            "FRS service loaded successfully | providers=%s",
            providers,
        )

    # ========================================================================
    # Optional NPZ watchlist loader
    # ========================================================================

    def _load_watchlist(self) -> None:
        source = os.getenv(
            "FRS_WATCHLIST_NPZ",
            "",
        ).strip()

        if not source:
            return

        path = Path(source)

        if not path.is_file():
            raise RuntimeError(
                f"FRS watchlist NPZ does not exist: {path}"
            )

        data = np.load(
            path,
            allow_pickle=False,
        )

        ids = data["ids"].astype(str).tolist()

        vectors = np.asarray(
            data["embeddings"],
            dtype=np.float32,
        )

        if len(ids) != len(vectors):
            raise RuntimeError(
                "watchlist ids/embeddings length mismatch"
            )

        self.watchlist.replace(
            [
                (
                    identity_id,
                    vector,
                    {},
                )
                for identity_id, vector in zip(
                    ids,
                    vectors,
                )
            ]
        )

    # ========================================================================
    # GPU inference validation
    # ========================================================================

    def _validate_inference(self) -> None:
        if self.embedder is None:
            raise RuntimeError(
                "FRS embedder is not loaded"
            )

        # Neutral deterministic dummy face tensor/image.
        sample = np.full(
            (112, 112, 3),
            127,
            dtype=np.uint8,
        )

        started = time.perf_counter()

        vector = self.embedder.embed(sample)

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        vector = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size == 0:
            raise RuntimeError(
                "FRS GPU validation returned an empty embedding"
            )

        if not np.all(np.isfinite(vector)):
            raise RuntimeError(
                "FRS GPU validation returned non-finite embedding values"
            )

        providers = list(
            self.embedder.active_providers or []
        )

        cuda_primary = bool(
            providers
            and providers[0] == "CUDAExecutionProvider"
        )

        if _env_flag("FRS_CUDA_REQUIRED", True):
            if not cuda_primary:
                raise RuntimeError(
                    "FRS inference validation failed because CUDA "
                    f"is not primary. Active providers: {providers}"
                )

        self.gpu_validation = {
            "validated": True,
            "providers": providers,
            "cuda_primary": cuda_primary,
            "embedding_dimension": int(vector.size),
            "inference_ms": round(
                elapsed_ms,
                3,
            ),
            "bootstrap": CUDA_BOOTSTRAP_STATE,
        }

    # ========================================================================
    # Reference face extraction
    # ========================================================================

    def reference_face_crop(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Return the strongest detected face.

        If the YuNet detector is unavailable or no face is detected,
        the original image is returned.
        """

        if self.detector is None:
            return image

        if image is None or image.size == 0:
            return image

        height, width = image.shape[:2]

        if width <= 0 or height <= 0:
            return image

        with self._detector_lock:
            self.detector.setInputSize(
                (
                    int(width),
                    int(height),
                )
            )

            _status, faces = self.detector.detect(
                np.ascontiguousarray(image)
            )

        if faces is None or len(faces) == 0:
            return image

        best = max(
            faces,
            key=lambda row: float(row[-1]),
        )

        x, y, w, h = (
            float(value)
            for value in best[:4]
        )

        padding = float(
            os.getenv(
                "FRS_REFERENCE_FACE_PADDING",
                "0.15",
            )
        )

        x1 = max(
            0,
            int(x - w * padding),
        )

        y1 = max(
            0,
            int(y - h * padding),
        )

        x2 = min(
            width,
            int(
                x
                + w * (1.0 + padding)
            ),
        )

        y2 = min(
            height,
            int(
                y
                + h * (1.0 + padding)
            ),
        )

        if x2 <= x1 or y2 <= y1:
            return image

        crop = image[
            y1:y2,
            x1:x2,
        ]

        return (
            crop
            if crop.size
            else image
        )

    # ========================================================================
    # Health
    # ========================================================================

    def health(self) -> dict[str, Any]:
        providers = (
            list(self.embedder.active_providers or [])
            if self.embedder
            else []
        )

        sync_state = self.sync.status()

        cuda_primary = bool(
            providers
            and providers[0] == "CUDAExecutionProvider"
        )

        cuda_required = _env_flag(
            "FRS_CUDA_REQUIRED",
            True,
        )

        gpu_validated = bool(
            self.gpu_validation.get(
                "validated"
            )
        )

        ready = (
            self.embedder is not None
            and bool(providers)
            and gpu_validated
            and sync_state.get("last_error") is None
            and (
                cuda_primary
                or not cuda_required
            )
            and self.error is None
        )

        return {
            "status": (
                "READY"
                if ready
                else "NOT_READY"
            ),
            "loaded": self.embedder is not None,
            "providers": providers,
            "cuda_primary": cuda_primary,
            "cuda_required": cuda_required,
            "gpu_validation": self.gpu_validation,
            "cuda_bootstrap": CUDA_BOOTSTRAP_STATE,
            "watchlist": sync_state,
            "error": self.error,
        }

    # ========================================================================
    # Face observation / watchlist matching
    # ========================================================================

    def observe(
        self,
        user_id: int,
        camera_id: str,
        track_id: str,
        timestamp: float,
        image: np.ndarray,
        detector_confidence: float,
    ) -> dict[str, Any]:

        if self.embedder is None:
            raise RuntimeError(
                "FRS model is not loaded"
            )

        if image is None or image.size == 0:
            raise RuntimeError(
                "FRS received an empty face image"
            )

        metrics = self.quality.evaluate(
            image,
            detector_confidence,
        )

        if not metrics["accepted"]:
            return {
                "status": "rejected",
                "track_id": str(track_id),
                "confirmed": False,
                "quality": metrics,
            }

        vector = self.embedder.embed(
            image
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size == 0:
            raise RuntimeError(
                "FRS embedding model returned an empty vector"
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise RuntimeError(
                "FRS embedding model returned non-finite values"
            )

        observation = FaceEmbedding(
            str(camera_id),
            str(track_id),
            float(timestamp),
            vector,
            float(metrics["quality"]),
            metrics,
        )

        fused = self.fusion.add(
            observation
        )

        match = self.watchlist.search(
            fused,
            user_id=int(user_id),
        )

        if match is None:
            return {
                "status": "no_match",
                "track_id": str(track_id),
                "confirmed": False,
                "observations": self.fusion.count(
                    camera_id,
                    track_id,
                ),
                "quality": metrics,
            }

        result = self.confirmation.observe(
            camera_id,
            track_id,
            match,
            timestamp,
        )

        return {
            **result,
            "track_id": str(track_id),
            "quality": metrics,
        }


# ============================================================================
# Internal authentication configuration
# ============================================================================


TOKEN = os.getenv(
    "FRS_INTERNAL_TOKEN",
    "",
).strip()

ENV = os.getenv(
    "ENV",
    "dev",
).strip().lower()


if ENV == "prod" and len(TOKEN) < 32:
    raise RuntimeError(
        "FRS_INTERNAL_TOKEN must be at least "
        "32 characters in production"
    )


if not TOKEN:
    TOKEN = "intel-i-local-frs-token-32-characters"


# ============================================================================
# Service instance
# ============================================================================


service = FRSService()


# ============================================================================
# Authentication
# ============================================================================


def _authorize(
    supplied: str | None,
) -> None:

    if (
        not supplied
        or not hmac.compare_digest(
            supplied,
            TOKEN,
        )
    ):
        # Deliberately return 404 rather than exposing existence
        # of the internal FRS service.
        raise HTTPException(
            status_code=404,
            detail="Not found",
        )


# ============================================================================
# FastAPI lifespan
#
# Replaces deprecated:
#
# @app.on_event("startup")
# @app.on_event("shutdown")
# ============================================================================


@asynccontextmanager
async def lifespan(
    _app: FastAPI,
):
    try:
        service.load()

    except Exception as exc:
        service.error = (
            f"{type(exc).__name__}: "
            f"{str(exc)[:500]}"
        )

        logger.exception(
            "FRS startup failed"
        )

        if _env_flag(
            "FRS_REQUIRED",
            False,
        ):
            raise

    try:
        yield

    finally:
        try:
            service.sync.stop()

        except Exception:
            logger.exception(
                "FRS watchlist synchronizer shutdown failed"
            )


# ============================================================================
# FastAPI application
# ============================================================================


app = FastAPI(
    title="INTEL-I Internal FRS",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


# ============================================================================
# Health endpoint
# ============================================================================


@app.get("/internal/health")
def health(
    x_intel_i_frs_token: str | None = Header(
        default=None
    ),
) -> dict[str, Any]:

    _authorize(
        x_intel_i_frs_token
    )

    return service.health()


# ============================================================================
# Face observation endpoint
# ============================================================================


@app.post("/internal/frs/observe")
async def observe(
    user_id: int = Form(...),
    camera_id: str = Form(...),
    track_id: str = Form(...),
    timestamp: float = Form(...),
    detector_confidence: float = Form(1.0),
    face_image: UploadFile = File(...),
    x_intel_i_frs_token: str | None = Header(
        default=None
    ),
) -> dict[str, Any]:

    _authorize(
        x_intel_i_frs_token
    )

    max_image_bytes = int(
        os.getenv(
            "FRS_MAX_FACE_IMAGE_BYTES",
            str(2 * 1024 * 1024),
        )
    )

    raw = await face_image.read(
        max_image_bytes + 1
    )

    if len(raw) > max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail="Face image is too large",
        )

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Empty face image",
        )

    image = cv2.imdecode(
        np.frombuffer(
            raw,
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid face image",
        )
    if image.shape[0] * image.shape[1] > 8_000_000:
        raise HTTPException(status_code=413, detail="Face image dimensions are too large")

    if int(user_id) <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id",
        )

    if not camera_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid camera_id",
        )

    if not track_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid track_id",
        )

    if not np.isfinite(
        float(timestamp)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid timestamp",
        )

    if not np.isfinite(
        float(detector_confidence)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid detector_confidence",
        )

    try:
        return service.observe(
            user_id=int(user_id),
            camera_id=camera_id,
            track_id=track_id,
            timestamp=float(timestamp),
            image=image,
            detector_confidence=float(
                detector_confidence
            ),
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


# ============================================================================
# Watchlist reload endpoint
# ============================================================================


@app.post("/internal/watchlist/reload")
def reload_watchlist(
    x_intel_i_frs_token: str | None = Header(
        default=None
    ),
) -> dict[str, Any]:

    _authorize(
        x_intel_i_frs_token
    )

    try:
        changed = service.sync.refresh(
            force=True
        )

    except Exception as exc:
        logger.exception(
            "FRS watchlist reload failed"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Watchlist reload failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    return {
        "status": "READY",
        "changed": changed,
        "watchlist": service.sync.status(),
    }


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    import uvicorn

    host = os.getenv(
        "FRS_HOST",
        "0.0.0.0",
    )

    port = int(
        os.getenv(
            "FRS_PORT",
            "9202",
        )
    )

    logger.info(
        "Starting INTEL-I FRS worker | "
        "host=%s port=%s",
        host,
        port,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
