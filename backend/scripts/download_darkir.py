#!/usr/bin/env python3
"""Download and verify the official DarkIR_384.pt checkpoint.

The checkpoint is downloaded only over HTTPS and is accepted only when both
size and SHA-256 match the pinned official artifact.
"""
from __future__ import annotations

import hashlib
import os
import ssl
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://huggingface.co/Cidaut/DarkIR/resolve/main/DarkIR_384.pt?download=true"
EXPECTED_SHA256 = "61eee7d5cfb593d408fba4c716874449b324ed1f49cbff97c2a25ce2fcbe4fde"
EXPECTED_SIZE = 13_397_397
MAX_BYTES = 20 * 1024 * 1024

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
TARGET = MODEL_DIR / "DarkIR_384.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> None:
    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"DarkIR checkpoint size mismatch: {size} != {EXPECTED_SIZE}")
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"DarkIR checkpoint SHA-256 mismatch: {digest} != {EXPECTED_SHA256}"
        )


def download() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        verify(TARGET)
        print(f"Verified existing checkpoint: {TARGET}")
        return

    request = Request(
        URL,
        headers={"User-Agent": "INTEL-I-DarkIR-installer/1.0"},
        method="GET",
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=".DarkIR_384.", suffix=".part", dir=str(MODEL_DIR)
    )
    os.close(fd)
    temp = Path(temp_name)

    try:
        with urlopen(request, timeout=120, context=ssl.create_default_context()) as response, temp.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise RuntimeError("DarkIR checkpoint exceeds the safety size limit")
                out.write(chunk)

        verify(temp)
        os.replace(temp, TARGET)
        print(f"Installed and verified: {TARGET}")
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:
        print(f"DarkIR installation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
