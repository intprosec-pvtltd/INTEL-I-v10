from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import cv2
import numpy as np

@dataclass(frozen=True)
class QualityThresholds:
    min_quality: float = 0.40
    enhance_below: float = 0.72
    dark_mean: float = 55.0
    very_dark_mean: float = 35.0
    bright_mean: float = 205.0
    contrast_min: float = 28.0
    blur_min: float = 65.0
    severe_blur: float = 25.0
    glare_ratio: float = 0.035
    fog_edge_density: float = 0.015

DEFAULT_THRESHOLDS = QualityThresholds()

def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))

def _normalize(v: float, lo: float, hi: float) -> float:
    return _clip01((float(v)-lo)/(hi-lo)) if hi > lo else 0.0

def assess_frame_quality(frame: np.ndarray, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> Dict[str, Any]:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return {"ok": False, "quality_score": 0.0, "mode": "INVALID", "needs_enhancement": False,
                "flags": ["invalid_frame"], "reason": "invalid_frame", "metrics": {}}
    try:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        metric = gray
        if w > 640:
            s = 640.0 / w
            metric = cv2.resize(gray, (640, max(1, int(h*s))), interpolation=cv2.INTER_AREA)

        brightness = float(np.mean(metric))
        contrast = float(np.std(metric))
        blur = float(cv2.Laplacian(metric, cv2.CV_64F).var())
        edges = cv2.Canny(metric, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size or 1)
        median = cv2.medianBlur(metric, 3)
        noise = float(np.std(metric.astype(np.float32)-median.astype(np.float32)))
        dark_ratio = float(np.mean(metric < 45))
        bright_ratio = float(np.mean(metric > 235))

        fog = contrast < 42 and edge_density < thresholds.fog_edge_density and dark_ratio < 0.55
        flags: List[str] = []
        if brightness <= thresholds.very_dark_mean: flags.append("very_dark")
        elif brightness <= thresholds.dark_mean: flags.append("low_light")
        if brightness >= thresholds.bright_mean or bright_ratio >= thresholds.glare_ratio: flags.append("overexposed_or_glare")
        if contrast < thresholds.contrast_min: flags.append("low_contrast")
        if blur < thresholds.severe_blur: flags.append("severe_blur")
        elif blur < thresholds.blur_min: flags.append("blur")
        if noise >= 18: flags.append("high_noise")
        if fog: flags.append("possible_fog_haze")

        brightness_score = _clip01(1 - abs(brightness-128)/128)
        contrast_score = _normalize(contrast, 15, 75)
        blur_score = _normalize(blur, 25, 450)
        noise_score = 1 - _normalize(noise, 4, 28)
        glare_penalty = _normalize(bright_ratio, thresholds.glare_ratio, min(0.2, thresholds.glare_ratio*4))
        quality = (brightness_score*0.27 + contrast_score*0.20 + blur_score*0.23 +
                   noise_score*0.12 + max(0, 1-glare_penalty)*0.18)
        if fog: quality -= 0.22
        quality = _clip01(quality)

        if quality >= thresholds.enhance_below: mode = "GOOD"
        elif "very_dark" in flags: mode = "LOW_LIGHT"
        elif fog: mode = "FOG_HAZE"
        elif "overexposed_or_glare" in flags: mode = "GLARE"
        elif "severe_blur" in flags: mode = "SEVERE_BLUR"
        elif "blur" in flags: mode = "BLUR"
        elif "high_noise" in flags: mode = "NOISY"
        elif "low_contrast" in flags: mode = "LOW_CONTRAST"
        else: mode = "MODERATE"

        needs = quality < thresholds.enhance_below and bool(set(flags) & {
            "very_dark","low_light","possible_fog_haze","overexposed_or_glare","low_contrast","high_noise"
        })
        return {
            "ok": quality >= thresholds.min_quality,
            "quality_score": round(quality,4),
            "mode": mode,
            "needs_enhancement": bool(needs),
            "flags": flags,
            "reason": "good" if quality >= thresholds.enhance_below else ",".join(flags) if flags else "moderate_quality",
            "metrics": {
                "width": int(w), "height": int(h), "mean_brightness": round(brightness,3),
                "contrast_std": round(contrast,3), "blur_laplacian": round(blur,3),
                "edge_density": round(edge_density,5), "noise_estimate": round(noise,3),
                "dark_ratio": round(dark_ratio,4), "bright_ratio": round(bright_ratio,4),
            },
        }
    except Exception as exc:
        return {"ok": False, "quality_score": 0.0, "mode": "ERROR", "needs_enhancement": False,
                "flags": ["quality_check_error"], "reason": str(exc)[:160], "metrics": {}}

def camera_quality(frame: np.ndarray) -> Dict[str, Any]:
    return assess_frame_quality(frame)
