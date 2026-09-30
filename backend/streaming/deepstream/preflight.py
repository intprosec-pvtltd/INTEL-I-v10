#!/usr/bin/env python3
from __future__ import annotations

"""Fail-fast DeepStream/NVIDIA preflight for the INTEL-I GPU server."""

import os
import shutil
import subprocess
from pathlib import Path

from ai.tensorrt.manifest import validate_manifest, tensorrt_version

from ai.tensorrt.manifest import validate_manifest, tensorrt_version

REQUIRED_PLUGINS = [
    "nvstreammux",
    "nvinfer",
    "nvtracker",
]


def run(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def main():
    report = {"ok": True, "checks": {}}
    gst = shutil.which("gst-inspect-1.0")
    report["checks"]["gst-inspect"] = bool(gst)
    if not gst:
        report["ok"] = False
    else:
        for plugin in REQUIRED_PLUGINS:
            code, out, err = run([gst, plugin])
            report["checks"][plugin] = code == 0
            if code != 0:
                report["ok"] = False

    nvidia = shutil.which("nvidia-smi")
    report["checks"]["nvidia-smi"] = bool(nvidia)
    if nvidia:
        code, out, err = run([nvidia, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        report["gpu"] = out if code == 0 else err
        report["ok"] &= code == 0
    else:
        report["ok"] = False

    manifest = Path(os.getenv("TENSORRT_MANIFEST", "/opt/intel-i/models/tensorrt/manifest.json"))
    manifest_result = validate_manifest(manifest, require_engine=True) if manifest.is_file() else {"ok": False, "errors": [f"missing manifest: {manifest}"], "engines": []}
    report["checks"]["tensorrt_manifest"] = bool(manifest_result.get("ok"))
    report["tensorrt"] = {
        "manifest": str(manifest),
        "validation": manifest_result,
        "python_version": tensorrt_version(),
    }
    if not manifest_result.get("ok"):
        report["ok"] = False

    print(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
