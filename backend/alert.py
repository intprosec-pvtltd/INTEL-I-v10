import cv2
import os
import time
from datetime import datetime
from pathlib import Path
from config import SNAPSHOT_COOLDOWN, SNAPSHOT_RETENTION_DAYS

alertMemory = {}

def cleanAlertMemory(timeout: float = 300):
    now = time.time()
    removeKeys = []

    for key, value in alertMemory.items():
        if now - value > timeout:
            removeKeys.append(key)

    for key in removeKeys:
        del alertMemory[key]


def should_snapshot(camID: str, track_id, rule_name: str) -> bool:
    now = time.time()
    key = f"{camID}_{track_id}_{rule_name}_snapshot"

    if key not in alertMemory:
        alertMemory[key] = now
        return True

    if now - alertMemory[key] > SNAPSHOT_COOLDOWN:
        alertMemory[key] = now
        return True

    return False


def snapShot(frame, box, camID, rule_name="critical") -> str | None:
    try:
        x1, y1, x2, y2 = map(int, box)
        padding = 50
        h, w, _ = frame.shape

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        path = f"alerts/{camID}/{rule_name}"
        os.makedirs(path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fileName = f"{path}/{timestamp}.jpg"

        success = cv2.imwrite(fileName, crop)

        return fileName if success else None

    except Exception:
        return None


def cleanAlertSnapshots(days: int = SNAPSHOT_RETENTION_DAYS):
    try:
        alerts_dir = Path("alerts")

        if not alerts_dir.exists():
            return
        threshold_time = time.time() - (days * 24 * 60 * 60)

        for jpg_file in alerts_dir.rglob("*.jpg"):
            if jpg_file.stat().st_mtime < threshold_time:
                jpg_file.unlink()

    except Exception:
        pass