"""Conservative, track-scoped DarkIR gate for verified face crops.

This does not own a model. It invokes the same AdvancedIntelligenceEngine.darkir
instance already used by ANPR inside the stateful Triton analytics process.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np
from prometheus_client import Counter, Histogram

ATTEMPTS = Counter('intel_i_frs_darkir_attempts_total', 'DarkIR face crop attempts')
SKIPPED = Counter('intel_i_frs_darkir_skipped_total', 'DarkIR face crop skip reason', ['reason'])
FAILED = Counter('intel_i_frs_darkir_failures_total', 'DarkIR face crop errors', ['reason'])
SELECTED = Counter('intel_i_frs_darkir_confirmed_total', 'Enhanced-only face identities confirmed')
LATENCY = Histogram('intel_i_frs_darkir_seconds', 'DarkIR face crop latency')


def _positive(name: str, default: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be numeric')
    if not 0 < value <= maximum:
        raise ValueError(f'{name} out of range')
    return value


class FaceDarkIRGate:
    def __init__(self, restorer: Callable[[np.ndarray], Any], *, clock=time.monotonic):
        self.restorer = restorer
        self.clock = clock
        self.cooldown = _positive('FRS_DARKIR_COOLDOWN_MS', 800, 60000) / 1000
        self.max_attempts = int(_positive('FRS_DARKIR_MAX_ATTEMPTS_PER_TRACK', 3, 30))
        self.min_size = int(_positive('FRS_DARKIR_MIN_FACE_SIZE', 40, 1000))
        self.min_blur = _positive('FRS_DARKIR_MIN_BLUR', 20, 10000)
        self.min_brightness = _positive('FRS_DARKIR_MIN_BRIGHTNESS', 9, 255)
        self.max_brightness = _positive('FRS_DARKIR_BRIGHTNESS_THRESHOLD', 65, 255)
        self.max_contrast = _positive('FRS_DARKIR_CONTRAST_THRESHOLD', 80, 255)
        self.timeout = _positive('FRS_DARKIR_TIMEOUT_MS', 1000, 30000) / 1000
        self._lock = threading.RLock()
        self._semaphore = threading.BoundedSemaphore(int(_positive('FRS_DARKIR_QUEUE_LIMIT', 1, 8)))
        self._attempts: dict[tuple[str,str], tuple[int,float,float]] = {}
        self._evidence: dict[tuple[str,str,str], deque] = {}

    @staticmethod
    def quality(crop):
        if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.size == 0:
            return {'usable': False}
        h,w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return {'usable': True, 'width': w, 'height': h, 'brightness': float(gray.mean()),
                'contrast': float(gray.std()), 'blur': float(cv2.Laplacian(gray, cv2.CV_64F).var())}

    def maybe_enhance(self, camera_id: str, track_id: str, crop: np.ndarray,
                      *, detected: bool, detection_score: float, enabled: bool = True):
        original = self.quality(crop)
        if not enabled:
            SKIPPED.labels('disabled').inc(); return None
        if not detected or detection_score < .7 or not original['usable']:
            SKIPPED.labels('face_not_verified').inc(); return None
        if min(original['width'],original['height']) < self.min_size or original['blur'] < self.min_blur or original['brightness'] < self.min_brightness:
            SKIPPED.labels('unusable').inc(); return None
        if original['brightness'] >= self.max_brightness and original['contrast'] >= self.max_contrast / 2:
            SKIPPED.labels('good_quality').inc(); return None
        if not self._semaphore.acquire(blocking=False):
            SKIPPED.labels('busy').inc(); return None
        key = (str(camera_id),str(track_id))
        try:
            now = self.clock()
            with self._lock:
                count,last,started = self._attempts.get(key,(0,0,now))
                if now - started > 30:
                    count,last,started=0,0,now
                if now - last < self.cooldown or count >= self.max_attempts:
                    SKIPPED.labels('cooldown').inc(); return None
                self._attempts[key]=(count+1,now,started)
                for stale in [k for k,v in self._attempts.items() if now - v[2] > 60]:
                    self._attempts.pop(stale,None)
            ATTEMPTS.inc()
            begin=time.perf_counter()
            try:
                result=self.restorer(crop)
            except Exception:
                FAILED.labels('exception').inc(); return None
            elapsed=time.perf_counter()-begin
            LATENCY.observe(elapsed)
            if elapsed > self.timeout:
                FAILED.labels('timeout').inc(); return None
            enhanced=result.get('image') if isinstance(result,dict) and result.get('model_used') else None
            if not isinstance(enhanced,np.ndarray) or enhanced.dtype!=np.uint8 or enhanced.shape!=crop.shape:
                FAILED.labels('invalid_output').inc(); return None
            new=self.quality(enhanced)
            if not new['usable'] or new['blur'] < self.min_blur or new['brightness'] <= original['brightness']:
                FAILED.labels('quality').inc(); return None
            return enhanced
        finally:
            self._semaphore.release()

    def confirmed_enhanced_only(self, camera_id, track_id, identity_id, timestamp):
        """Require independent enhanced observations before an enhanced-only alert."""
        now=float(timestamp)
        key=(str(camera_id),str(track_id),str(identity_id))
        with self._lock:
            seq=self._evidence.setdefault(key,deque(maxlen=3))
            while seq and now-seq[0] > 5:
                seq.popleft()
            if not seq or now-seq[-1] >= .3:
                seq.append(now)
            accepted=len(seq)>=3
            if accepted:
                SELECTED.inc()
            for old in [k for k,v in self._evidence.items() if not v or now-v[-1]>15]:
                self._evidence.pop(old,None)
            return accepted

    def clear_camera(self,camera_id):
        with self._lock:
            for key in [key for key in self._attempts if key[0]==str(camera_id)]:self._attempts.pop(key,None)
            for key in [key for key in self._evidence if key[0]==str(camera_id)]:self._evidence.pop(key,None)
