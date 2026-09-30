#!/usr/bin/env python3
"""Acceptance smoke test for INTEL-I distributed camera ownership.

This script does not kill processes. It starts requested cameras through the
existing API, waits for unique worker ownership, reports capacity/runtime state,
and optionally stops them in a finally block.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _request(client: httpx.Client, method: str, path: str, token: str) -> Any:
    response = client.request(method, path, headers=_headers(token))
    response.raise_for_status()
    if response.content:
        return response.json()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--camera", action="append", dest="cameras", required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    started: list[str] = []
    client = httpx.Client(base_url=base, timeout=15.0)
    try:
        for cam_id in args.cameras:
            result = _request(client, "POST", f"/camera/{cam_id}/start", args.api_token)
            print(f"START {cam_id}: {result}")
            started.append(cam_id)

        deadline = time.monotonic() + max(5.0, args.timeout)
        last_inventory = None
        while time.monotonic() < deadline:
            inventory = _request(client, "GET", "/api/system/workers", args.api_token)
            last_inventory = inventory
            runtimes = [
                item for item in (inventory.get("camera_runtimes") or [])
                if item.get("cam_id") in set(args.cameras)
                and str(item.get("state") or "").upper() in {"CLAIMED", "RUNNING", "RECONNECTING"}
            ]
            cam_ids = [str(item.get("cam_id")) for item in runtimes]
            camera_pks = [int(item.get("camera_pk")) for item in runtimes if item.get("camera_pk") is not None]
            unique = len(camera_pks) == len(set(camera_pks)) == len(args.cameras)
            all_present = all(cam in cam_ids for cam in args.cameras)
            if unique and all_present:
                print("PASS: every requested camera has one unique distributed runtime owner")
                for item in runtimes:
                    print(
                        f"  {item.get('cam_id')} pk={item.get('camera_pk')} "
                        f"worker={item.get('worker_id')} state={item.get('state')} "
                        f"fps={item.get('processing_fps')}"
                    )
                print(
                    f"Workers={inventory.get('worker_count')} "
                    f"capacity={inventory.get('total_capacity')} "
                    f"owned={inventory.get('owned_cameras')} "
                    f"available={inventory.get('available_capacity')}"
                )
                return 0
            time.sleep(1.0)

        print("FAIL: cameras did not reach unique distributed ownership before timeout", file=sys.stderr)
        print(last_inventory, file=sys.stderr)
        return 2
    finally:
        if not args.keep_running:
            for cam_id in started:
                try:
                    result = _request(client, "POST", f"/camera/{cam_id}/stop", args.api_token)
                    print(f"STOP {cam_id}: {result}")
                except Exception as exc:
                    print(f"STOP FAILED {cam_id}: {exc}", file=sys.stderr)
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
