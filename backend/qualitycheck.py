import cv2
import numpy as np
from config import MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_BLUR_SCORE, MAX_DARK_RATIO


def camera_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_ratio = float(np.mean(gray < 30))

    ok = (
        MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
        and blur >= MIN_BLUR_SCORE
        and dark_ratio <= MAX_DARK_RATIO
    )

    reason = "OK"

    if brightness < MIN_BRIGHTNESS:
        reason = "TOO_DARK"
    elif brightness > MAX_BRIGHTNESS:
        reason = "TOO_BRIGHT"
    elif blur < MIN_BLUR_SCORE:
        reason = "BLURRY"
    elif dark_ratio > MAX_DARK_RATIO:
        reason = "HIGH_DARK_AREA"

    return {
        "ok": ok,
        "reason": reason,
        "brightness": round(brightness, 2),
        "blur": round(blur, 2),
        "dark_ratio": round(dark_ratio, 2),
    }