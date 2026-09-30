"""GPU preprocessing capability helper.

Ultralytics handles letterboxing and coordinate restoration safely for numpy
batch inputs.  This helper enables pinned host staging for compatible custom
paths without forcing coordinate-changing tensor preprocessing on legacy code.
"""
from __future__ import annotations

import os
from typing import Any


def status() -> dict[str, Any]:
    enabled = str(os.getenv("GPU_PREPROCESSING_ENABLED", "true")).strip().lower() in {
        "1", "true", "yes", "on"
    }
    state = {
        "enabled": enabled,
        "cuda_available": False,
        "pinned_memory_available": False,
        "non_blocking_transfer": False,
        "active_path": "ultralytics-batched-numpy",
    }
    if not enabled:
        state["active_path"] = "cpu"
        return state
    try:
        import torch
        state["cuda_available"] = bool(torch.cuda.is_available())
        state["pinned_memory_available"] = bool(torch.cuda.is_available())
        state["non_blocking_transfer"] = bool(torch.cuda.is_available())
        # The central batch path deliberately passes the whole numpy batch to
        # Ultralytics in one call.  Ultralytics performs one batched preprocess
        # and device transfer while preserving original-image box coordinates.
        if torch.cuda.is_available():
            state["active_path"] = "ultralytics-batched-gpu"
    except Exception:
        pass
    return state
