"""Isolated Paddle process preserving the uploaded Awiros adapter."""
import json
import os
import sys

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        sys.path.insert(0, os.getenv("INTEL_I_BACKEND_PATH", "/app"))
        from services.awirosOcr import AwirosOCR
        from inference.triton_client import decode_payload
        self.decode = decode_payload
        self.engine = AwirosOCR(
            weights_path=os.getenv("ANPR_AWIROS_MODEL_PATH", "models/awiros_anpr_ocr/model.safetensors"),
            dictionary_path=os.getenv("ANPR_AWIROS_DICT_PATH", "models/awiros_anpr_ocr/en_dict.txt"),
            paddleocr_dir=os.getenv("ANPR_AWIROS_PADDLEOCR_DIR", "vendor/PaddleOCR"),
            device=os.getenv("ANPR_OCR_DEVICE", "cuda"), strict_device=True,
        ).load()

    def execute(self, requests):
        responses = []
        for request in requests:
            output = []
            for value in pb_utils.get_input_tensor_by_name(request, "REQUEST").as_numpy():
                row = {}
                try:
                    row = self.decode(value[0])
                    image = cv2.imdecode(np.frombuffer(row["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("invalid OCR image")
                    text, confidence = self.engine.recognize(image)
                    result = {"ok": True, "text": text, "confidence": float(confidence)}
                except Exception as exc:
                    result = {"ok": False, "error": type(exc).__name__}
                result.update(camera_id=row.get("camera_id"), request_id=row.get("request_id"))
                output.append([json.dumps(result).encode()])
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("RESPONSE", np.asarray(output, dtype=object))]))
        return responses
