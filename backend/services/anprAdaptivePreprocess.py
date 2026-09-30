"""Bounded ANPR preprocessing for difficult Indian CCTV capture conditions.

This module does not invent plate characters. It creates conservative OCR views
for blur, low-light, glare/haze and modest camera skew; temporal consensus and
plateFusion remain authoritative.
"""
from __future__ import annotations

import math
import os
from typing import Any

import cv2
import numpy as np


def _f(name: str, default: float, lo: float, hi: float) -> float:
    try: value = float(os.getenv(name, str(default)))
    except Exception: value = default
    return max(lo, min(hi, value if math.isfinite(value) else default))


def analyze_conditions(image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    p95 = float(np.percentile(gray, 95))
    return {
        "brightness": round(mean, 3),
        "contrast": round(contrast, 3),
        "blur_score": round(blur, 3),
        "low_light": mean < _f("ANPR_LOW_LIGHT_MEAN", 72.0, 10.0, 160.0),
        "low_contrast": contrast < _f("ANPR_LOW_CONTRAST_STD", 38.0, 5.0, 100.0),
        "blurred": blur < _f("ANPR_BLUR_VARIANCE_MIN", 75.0, 1.0, 1000.0),
        "glare": p95 > _f("ANPR_GLARE_P95", 245.0, 180.0, 255.0),
    }


def _rotate_keep(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def build_adaptive_variants(image: np.ndarray, *, max_variants: int | None = None) -> tuple[list[np.ndarray], dict[str, Any]]:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return [], {"invalid": True}
    limit = max(1, min(12, int(max_variants or os.getenv("ANPR_MAX_ADAPTIVE_VARIANTS", "8"))))
    conditions = analyze_conditions(image)
    variants: list[np.ndarray] = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    contrast = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    variants.append(contrast)

    if conditions["low_light"]:
        # Gamma < 1 brightens shadows without hallucinating high-frequency detail.
        gamma = _f("ANPR_LOW_LIGHT_GAMMA", 0.62, 0.35, 0.95)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        variants.append(cv2.LUT(image, table))

    if conditions["blurred"]:
        # Mild unsharp mask only. Strong deconvolution can create false glyph strokes.
        smooth = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        variants.append(cv2.addWeighted(contrast, 1.65, smooth, -0.65, 0))

    if conditions["low_contrast"] or conditions["glare"]:
        denoised = cv2.bilateralFilter(contrast, 5, 35, 35)
        variants.append(denoised)

    if os.getenv("ANPR_ANGLE_VARIANTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}:
        # CCTV plates often have modest roll. Large perspective distortion should be
        # handled by better plate detection/camera placement, not guessed warps.
        for angle in (-8.0, -4.0, 4.0, 8.0):
            variants.append(_rotate_keep(contrast, angle))

    unique: list[np.ndarray] = []
    signatures: set[tuple[int, int, int]] = set()
    for item in variants:
        if len(unique) >= limit:
            break
        sig = (item.shape[0], item.shape[1], int(item.mean()))
        if sig in signatures:
            continue
        signatures.add(sig)
        unique.append(np.ascontiguousarray(item))
    conditions["variant_count"] = len(unique)
    return unique, conditions
