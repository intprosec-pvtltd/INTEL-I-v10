"""Hardware decode capability detection and safe FFmpeg/PyAV option selection."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

from .settings import SETTINGS

_lock = threading.RLock()
_cache: tuple[float, dict[str, Any]] | None = None


def _run(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=3, check=False)
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout or ""


def detect_capability(force: bool = False) -> dict[str, Any]:
    global _cache
    now = time.monotonic()
    with _lock:
        if _cache and not force and now - _cache[0] < 30.0:
            return dict(_cache[1])
    ffmpeg = shutil.which("ffmpeg")
    output = _run([ffmpeg, "-hide_banner", "-hwaccels"]) if ffmpeg else ""
    accelerators = [
        line.strip().lower()
        for line in output.splitlines()
        if line.strip() and "hardware acceleration methods" not in line.lower()
    ]
    cuda = "cuda" in accelerators
    requested = str(SETTINGS.decode_device or "auto").lower()
    enabled = bool(SETTINGS.hw_decode_enabled)
    selected = "cpu"
    if enabled and cuda and requested in {"auto", "cuda", "nvdec", "gpu", "0"}:
        selected = "cuda"
    result = {
        "enabled": enabled,
        "requested_device": requested,
        "selected_device": selected,
        "ffmpeg_present": bool(ffmpeg),
        "ffmpeg_path": ffmpeg,
        "hwaccels": sorted(set(accelerators)),
        "nvdec_available": cuda,
        "fallback": selected == "cpu",
    }
    with _lock:
        _cache = (now, dict(result))
    return result


def pyav_options() -> dict[str, str]:
    """Options attempted when PyAV/FFmpeg exposes CUDA hwaccel input flags.

    PyAV/FFmpeg builds differ.  The capture layer retries without these options
    if they are rejected, so local/CPU deployments remain supported.
    """
    state = detect_capability()
    if state.get("selected_device") != "cuda":
        return {}
    return {
        "hwaccel": "cuda",
        "hwaccel_device": str(os.getenv("VIDEO_HW_DEVICE", "0")),
    }


def make_pyav_hwaccel(av_module: Any):
    """Return a real PyAV CUDA HWAccel object when supported.

    INTEL-I pins PyAV 16.x, whose public ``HWAccel`` API passes a hardware
    device context into libavcodec. Returning ``None`` keeps older/local builds
    on the safe software path. ``allow_software_fallback=False`` is deliberate:
    if CUDA cannot actually be activated, the caller catches the open failure
    and explicitly retries CPU decode, so telemetry never reports a false GPU
    decode state.
    """
    state = detect_capability()
    if state.get("selected_device") != "cuda":
        return None
    try:
        hwaccel_module = getattr(getattr(av_module, "codec", None), "hwaccel", None)
        factory = getattr(hwaccel_module, "HWAccel", None)
        if factory is None:
            return None
        return factory(
            device_type="cuda",
            device=str(os.getenv("VIDEO_HW_DEVICE", "0")),
            allow_software_fallback=False,
        )
    except TypeError:
        # PyAV builds may not expose an explicit ``device`` keyword.
        try:
            return av_module.codec.hwaccel.HWAccel(
                device_type="cuda", allow_software_fallback=False
            )
        except Exception:
            return None
    except Exception:
        return None
