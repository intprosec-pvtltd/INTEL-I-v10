"""Optional existing FRSService, with its watchlist sync and track fusion intact."""
import json
import os
import sys
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        sys.path.insert(0, os.getenv("INTEL_I_BACKEND_PATH", "/app"))
        from workers.frs_worker import FRSService
        from inference.triton_client import decode_payload, json_default
        self.decode, self.json_default = decode_payload, json_default
        self.service = FRSService()
        self.service.load()

    def execute(self, requests):
        responses = []
        for request in requests:
            output = []
            for value in pb_utils.get_input_tensor_by_name(request, "REQUEST").as_numpy():
                row = {}
                try:
                    row = self.decode(value[0])
                    if row.get("operation") == "reload":
                        self.service._load_watchlist()
                        result = {"ok": True}
                    else:
                        image = cv2.imdecode(np.frombuffer(row["jpeg"], np.uint8), cv2.IMREAD_COLOR)
                        result = {"ok": True, "result": self.service.observe(
                            user_id=int(row["user_id"]), camera_id=row["camera_id"],
                            track_id=row["track_id"], timestamp=float(row["timestamp"]),
                            image=image, detector_confidence=float(row["detector_confidence"]))}
                except Exception as exc:
                    result = {"ok": False, "error": type(exc).__name__}
                result.update(camera_id=row.get("camera_id"), request_id=row.get("request_id"))
                output.append([json.dumps(result, default=self.json_default).encode()])
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("RESPONSE", np.asarray(output, dtype=object))]))
        return responses

    def finalize(self):
        self.service.sync.stop()
