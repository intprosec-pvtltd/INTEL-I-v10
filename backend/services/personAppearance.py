from __future__ import annotations
from typing import Any, Dict, Optional
import cv2
import numpy as np


def dominant_color_name(crop: np.ndarray) -> str:
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])

    if v < 45:
        return "black"
    if s < 28 and v > 190:
        return "white"
    if s < 35:
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 160:
        return "purple"
    return "red"


def appearance_features(person_crop: np.ndarray) -> Dict[str, Any]:
    if not isinstance(person_crop, np.ndarray) or person_crop.size == 0:
        return {"valid": False}

    h, w = person_crop.shape[:2]
    upper = person_crop[:max(1, int(h * 0.60))]
    lower = person_crop[max(0, int(h * 0.45)):]

    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    descriptor = histogram.flatten().astype(float).tolist()

    return {
        "valid": True,
        "width": int(w),
        "height": int(h),
        "upper_color": dominant_color_name(upper),
        "lower_color": dominant_color_name(lower),
        "mean_brightness": round(float(np.mean(cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY))), 3),
        "appearance_descriptor": [round(value, 6) for value in descriptor],
    }
