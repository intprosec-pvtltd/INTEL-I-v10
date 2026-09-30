"""Compatibility serving: one stateful analytics owner, existing model semantics.

The Python backend batches JPEG requests across cameras before the existing
YOLO batch call. Downstream track/ROI/correlation/evidence code stays intact.
Keep count=1: replicated instances would split camera/global identity state.
"""
import json
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        os.environ["INTEL_I_TRITON_EMBEDDED"] = "1"
        os.environ["INTEL_I_PROCESS_ROLE"] = "inference-worker"
        sys.path.insert(0, os.getenv("INTEL_I_BACKEND_PATH", "/app"))
        from workers import inference_worker
        from inference.triton_client import decode_payload, json_default
        self.worker = inference_worker
        self.decode = decode_payload
        self.json_default = json_default
        self.worker.startup()

    def execute(self, requests):
        rows, groups = [], []
        responses = [None] * len(requests)
        for index, request in enumerate(requests):
            try:
                values = pb_utils.get_input_tensor_by_name(request, "REQUEST").as_numpy()
                decoded = [self.decode(value[0]) for value in values]
                groups.append((index, len(rows), len(decoded)))
                rows.extend(decoded)
            except Exception as exc:
                responses[index] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
        if rows:
            results = [None] * len(rows)
            frames, indices = [], []
            for i, row in enumerate(rows):
                if row.get("operation") in {"face_embeddings", "face_evidence"}:
                    try:
                        from services.personRecognition import extract_embeddings, detect_face_evidence_candidates
                        fn = extract_embeddings if row["operation"] == "face_embeddings" else detect_face_evidence_candidates
                        faces = fn(self.worker._decode_item(row), **row.get("options", {}))
                        results[i] = {"ok": True, "camera_id": row["camera_id"], "faces": faces}
                    except Exception as exc:
                        results[i] = {"ok": False, "camera_id": row["camera_id"], "error": type(exc).__name__}
                elif row.get("operation") == "reset":
                    self.worker.runtime.reset_camera(str(row["camera_id"]))
                    results[i] = {"ok": True, "camera_id": row["camera_id"]}
                else:
                    indices.append(i)
                    frames.append(row)
            if frames:
                try:
                    for i, result in zip(indices, self.worker.handle_batch(frames)):
                        results[i] = result
                except Exception as exc:
                    for i in indices:
                        results[i] = {"ok": False, "camera_id": rows[i].get("camera_id"), "error": type(exc).__name__}
            for row, result in zip(rows, results):
                result["request_id"] = row.get("request_id")
            for index, start, count in groups:
                output = np.asarray([[json.dumps(r, allow_nan=False, default=self.json_default).encode()] for r in results[start:start + count]], dtype=object)
                responses[index] = pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("RESPONSE", output)])
        return responses

    def finalize(self):
        self.worker.shutdown()
