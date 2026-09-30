from __future__ import annotations

import cv2
import numpy as np


class FaceQualityGate:
    def __init__(self, *, min_size: int = 40, min_blur: float = 20.0, min_brightness: float = 20.0, max_brightness: float = 235.0, min_contrast: float = 12.0, min_detector_confidence: float = 0.70) -> None:
        self.min_size = int(min_size)
        self.min_blur = float(min_blur)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)
        self.min_contrast = float(min_contrast)
        self.min_detector_confidence = float(min_detector_confidence)

    def evaluate(self, face: np.ndarray, detector_confidence: float, *, pose: dict | None = None, crop_completeness: float = 1.0) -> dict:
        if not isinstance(face, np.ndarray) or face.size == 0:
            return {"accepted": False, "quality": 0.0, "reason": "empty_face"}
        height, width = face.shape[:2]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        pose = pose or {}
        yaw, pitch, roll = (abs(float(pose.get(k, 0.0))) for k in ("yaw", "pitch", "roll"))
        checks = {
            "size": min(width, height) >= self.min_size,
            "confidence": float(detector_confidence) >= self.min_detector_confidence,
            "blur": blur >= self.min_blur,
            "brightness": self.min_brightness <= brightness <= self.max_brightness,
            "contrast": contrast >= self.min_contrast,
            "pose": yaw <= 45.0 and pitch <= 35.0 and roll <= 35.0,
            "complete": float(crop_completeness) >= 0.80,
        }
        scores = [
            min(1.0, min(width, height) / max(1.0, self.min_size * 2.0)),
            max(0.0, min(1.0, float(detector_confidence))),
            min(1.0, blur / max(1.0, self.min_blur * 3.0)),
            min(1.0, contrast / max(1.0, self.min_contrast * 3.0)),
            max(0.0, min(1.0, float(crop_completeness))),
        ]
        return {
            "accepted": all(checks.values()), "quality": float(np.mean(scores)),
            "width": width, "height": height, "detector_confidence": float(detector_confidence),
            "blur": blur, "brightness": brightness, "contrast": contrast,
            "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "crop_completeness": float(crop_completeness), "checks": checks,
        }
