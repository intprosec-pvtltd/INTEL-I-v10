"""Strict opt-in engine selection. Never silently substitute missing engines."""
import hashlib
import json
import os
from pathlib import Path


def resolve_engine(source: str) -> str:
    manifest = os.getenv("INTEL_I_TENSORRT_MANIFEST", "").strip()
    if not manifest:
        return source
    document = json.loads(Path(manifest).read_text())
    if document.get("precision") != "fp16":
        raise ValueError("Only validated FP16 build manifests are supported")
    entry = document["models"].get(Path(source).name)
    if entry is None:
        raise ValueError(f"Model absent from TensorRT manifest: {Path(source).name}")
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Engine integrity failed: {path.name}")
    # The build tool intentionally leaves accuracy_validated false. Operator
    # must explicitly opt in to unvalidated engines for benchmark/evaluation.
    if not entry.get("accuracy_validated") and os.getenv("INTEL_I_ALLOW_UNVALIDATED_ENGINES") != "true":
        raise ValueError("Engine accuracy is not validated; run comparison before production")
    return str(path)
