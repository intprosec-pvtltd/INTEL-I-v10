"""Paddle remains in a separate Triton Python stub, never in the Torch stub."""
import os

import cv2
import numpy as np

from inference.triton_client import InferenceUnavailable, TritonInferenceClient


class TritonOCR:
    version = "Awiros-ANPR-OCR/triton-v1"
    actual_device = "triton"

    def __init__(self):
        self.client = TritonInferenceClient()
        self.client.model = os.getenv("INTEL_I_TRITON_OCR_MODEL", "awiros_anpr_ocr")

    def load(self):
        self.client.health()
        return self

    def health(self):
        return {**self.client.health(), "loaded": True, "actual_device": "triton"}

    def recognize(self, crop):
        ok, image = cv2.imencode(".png", crop)
        if not ok:
            raise ValueError("invalid OCR crop")
        result = self.client.infer_batch([{"camera_id": "ocr", "jpeg": image.tobytes()}])[0]
        if not result.get("ok"):
            raise InferenceUnavailable(result.get("error", "OCR failed"))
        return result.get("text"), float(result.get("confidence", 0))
