"""Secure, manifest-driven INTEL-I model provisioning.

Downloads are optional and must be explicitly configured per model through the
manifest/env. Existing valid artifacts are never replaced. Downloads are HTTPS
only, host allowlisted, size bounded, checksum verified and atomically installed.
"""
from __future__ import annotations

import argparse
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from model_manifest import BACKEND_ROOT, load_manifest, sha256_file

load_dotenv(BACKEND_ROOT / ".env")


def _allowed_hosts(env_name: str) -> set[str]:
    return {
        x.strip().lower()
        for x in os.getenv(env_name, "").split(",")
        if x.strip()
    }


def _validate_source(url: str, allowed_hosts: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError("model source must use HTTPS")
    if not parsed.hostname:
        raise RuntimeError("model source hostname missing")
    if parsed.username or parsed.password:
        raise RuntimeError("credentials in model URL are forbidden")
    if parsed.hostname.lower() not in allowed_hosts:
        raise RuntimeError(f"model source host not allowlisted: {parsed.hostname}")


def _download(url: str, target: Path, max_bytes: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "INTEL-I-model-provisioner/1"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise RuntimeError("remote model exceeds configured maximum size")
        total = 0
        with target.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError("download exceeded configured maximum size")
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())


def provision_one(spec, *, force: bool = False) -> str:
    if not spec.enabled:
        return f"SKIP {spec.id}: disabled"
    expected = spec.expected_sha256
    if not expected:
        raise RuntimeError(
            f"{spec.id}: expected SHA-256 is not pinned. Set the model-specific SHA env before provisioning."
        )
    path = spec.path
    if path.is_file() and not force:
        actual = sha256_file(path)
        if actual.lower() == expected.lower():
            return f"OK   {spec.id}: already verified at {path}"
        raise RuntimeError(f"{spec.id}: existing artifact checksum mismatch: {path}")
    url = spec.url
    if not url:
        raise RuntimeError(f"{spec.id}: no approved download URL configured")
    allowed = _allowed_hosts(spec.allowed_hosts_env)
    if not allowed:
        raise RuntimeError(f"{spec.id}: {spec.allowed_hosts_env} is empty")
    _validate_source(url, allowed)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _download(url, tmp, spec.max_bytes)
        actual = sha256_file(tmp)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{spec.id}: SHA-256 mismatch after download")
        if spec.size_bytes is not None and tmp.stat().st_size != spec.size_bytes:
            raise RuntimeError(
                f"{spec.id}: size mismatch expected={spec.size_bytes} actual={tmp.stat().st_size}"
            )
        os.replace(tmp, path)
        return f"OK   {spec.id}: installed {path}"
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace only after a valid verified download")
    parser.add_argument("--model", action="append", default=[], help="provision only one or more model IDs")
    args = parser.parse_args()

    wanted = set(args.model)
    failures = []
    for spec in load_manifest():
        if wanted and spec.id not in wanted:
            continue
        try:
            print(provision_one(spec, force=args.force))
        except Exception as exc:
            failures.append(f"{spec.id}: {exc}")
            print(f"FAIL {spec.id}: {exc}")
    if failures:
        raise SystemExit("Model provisioning failed:\n- " + "\n- ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
