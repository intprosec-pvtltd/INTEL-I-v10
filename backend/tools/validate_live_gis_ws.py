"""Authenticated deployed WebSocket validator.

Credentials are read only from INTELI_TEST_EMAIL and INTELI_TEST_PASSWORD.
Never pass passwords on the command line.
"""
import asyncio
import json
import os
from urllib.parse import urlparse

import httpx
import websockets


async def main():
    base = os.environ["INTELI_TEST_BASE_URL"].rstrip("/")
    email = os.environ["INTELI_TEST_EMAIL"]
    password = os.environ["INTELI_TEST_PASSWORD"]
    parsed = urlparse(base)
    if parsed.scheme != "https" and os.getenv("ENV") == "prod":
        raise RuntimeError("Production validation requires HTTPS")
    async with httpx.AsyncClient(base_url=base, timeout=20, follow_redirects=False) as client:
        csrf = (await client.get("/auth/csrf")).json()["csrf_token"]
        login = await client.post("/auth/login", json={"email": email, "password": password}, headers={"X-CSRF-Token": csrf})
        login.raise_for_status()
        cookies = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
        health = await client.get("/api/system/health"); health.raise_for_status()
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    async with websockets.connect(ws_url, additional_headers={"Cookie": cookies}, open_timeout=15) as socket:
        await socket.send("ping")
        observed = []
        for _ in range(30):
            raw = await asyncio.wait_for(socket.recv(), timeout=5)
            payload = json.loads(raw)
            if payload.get("type") in {"vehicle_position", "person_position"}:
                observed.append(payload.get("type"))
                if len(set(observed)) == 2: break
        print(json.dumps({"ok": bool(observed), "observed_event_types": sorted(set(observed))}))


if __name__ == "__main__":
    asyncio.run(main())
