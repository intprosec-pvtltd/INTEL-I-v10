import os
from urllib.parse import quote


def get_sentinel_rtsp_url(camera_id: str) -> str:
    host = os.getenv(
        "SENTINEL_RTSP_HOST",
        "103.250.160.189",
    )

    port = os.getenv(
        "SENTINEL_RTSP_PORT",
        "8554",
    )

    username = os.getenv("SENTINEL_USERNAME")
    password = os.getenv("SENTINEL_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Sentinel username/password not configured"
        )

    username = quote(username, safe="")
    password = quote(password, safe="")

    camera_id = camera_id.strip().lower()

    return (
        f"rtsp://{username}:{password}"
        f"@{host}:{port}/stream/{camera_id}"
    )