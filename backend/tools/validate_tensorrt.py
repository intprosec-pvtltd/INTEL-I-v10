#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed TensorRT artifact and runtime validation."""

import argparse
import json
import subprocess
from pathlib import Path

from ai.tensorrt.manifest import validate_manifest


def run(command: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode == 0, (p.stdout + "\n" + p.stderr).strip()
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("models/tensorrt/manifest.json"))
    ap.add_argument("--runtime", action="store_true", help="Deserialize each engine with TensorRT Python bindings")
    ap.add_argument("--trtexec", action="store_true", help="Run trtexec --loadEngine sanity checks when available")
    args = ap.parse_args()

    report = {"ok": True, "manifest": str(args.manifest), "engines": []}
    result = validate_manifest(args.manifest, require_engine=True)
    report["manifest_validation"] = result
    if not result["ok"]:
        report["ok"] = False

    if args.runtime and result["ok"]:
        try:
            import tensorrt as trt  # type: ignore
            logger = trt.Logger(trt.Logger.ERROR)
            rt = trt.Runtime(logger)
            data = json.loads(args.manifest.read_text(encoding="utf-8"))
            for entry in data["engines"]:
                blob = Path(entry["engine_path"]).read_bytes()
                engine = rt.deserialize_cuda_engine(blob)
                ok = engine is not None
                report["engines"].append({"role": entry["role"], "deserialize_ok": ok})
                report["ok"] = report["ok"] and ok
        except Exception as exc:
            report["ok"] = False
            report["runtime_error"] = str(exc)

    if args.trtexec:
        trtexec = "trtexec"
        data = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.is_file() else {"engines": []}
        for entry in data.get("engines", []):
            ok, output = run([trtexec, f"--loadEngine={entry['engine_path']}", "--skipInference"], timeout=60)
            report["engines"].append({"role": entry["role"], "trtexec_ok": ok, "output_tail": output[-1000:]})
            report["ok"] = report["ok"] and ok

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
