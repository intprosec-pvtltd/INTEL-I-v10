from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
from model_manifest import BACKEND_ROOT, load_manifest, sha256_file

load_dotenv(BACKEND_ROOT / ".env")

fail = 0
for spec in load_manifest():
    if not spec.enabled:
        print(f"SKIP {spec.id}: disabled")
        continue
    path = spec.path
    expected = spec.expected_sha256
    present = path.is_file()
    actual = sha256_file(path) if present else None
    valid = bool(expected) and actual == expected
    print(f"{spec.id}: present={present} pinned={bool(expected)} checksum_valid={valid} path={path}")
    if not (present and expected and valid):
        fail += 1
if fail:
    raise SystemExit(f"FAIL: {fail} enabled controlled model(s) not ready")
print("PASS: all enabled controlled model artifacts are checksum-pinned and valid")
