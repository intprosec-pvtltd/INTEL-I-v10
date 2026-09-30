"""Independent RTSP capture, preview and adaptive analytics loops for one camera."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import threading
import time
from typing import Any, Callable

from services.latestFrameBuffer import LatestFrameBuffer
from services.playbackClock import PlaybackClock
try:
    from inference.client import InferenceBackpressureError
except Exception:
    class InferenceBackpressureError(RuntimeError):
        pass

logger = logging.getLogger(__name__)


def is_upload_source(camera):
    return str(getattr(camera, "source_type", "")).lower() in {"upload", "uploaded", "file"}



class CameraPipeline:
    def __init__(
        self,
        *,
        camera: Any,
        runtime: Any,
        capture_factory: Callable[[Any], Any],
        analytics_fps: float = 5.0,
        preview_fps: float = 12.0,
        on_preview: Callable[[Any, Any], None] | None = None,
        on_processed: Callable[[Any, Any], None] | None = None,
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
    ) -> None:
        self.camera = camera
        self.runtime = runtime
        self.capture_factory = capture_factory
        self.analytics_fps = max(0.1, min(60.0, float(analytics_fps)))
        self.preview_fps = max(0.1, min(60.0, float(preview_fps)))
        self.on_preview = on_preview
        self.on_processed = on_processed
        self.reconnect_initial_seconds = max(0.1, float(reconnect_initial_seconds))
        self.reconnect_max_seconds = max(self.reconnect_initial_seconds, float(reconnect_max_seconds))
        self.buffer = LatestFrameBuffer()
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.analytics_thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.frames_captured = 0
        self.preview_frames = 0
        self.frames_processed = 0
        self.analytics_skipped = 0
        self.reconnect_count = 0
        self.last_frame_timestamp = None
        self._last_capture_mono = time.monotonic()
        self.frame_ttl = max(0.01, float(os.getenv("INFERENCE_FRAME_TTL_MS", "500")) / 1000)
        self.stuck_seconds = max(5.0, float(os.getenv("CAMERA_STUCK_SECONDS", "30")))
        self._started_mono = time.monotonic()
        self._last_processed_mono = self._started_mono
        self._last_consumed_sequence = 0

    @property
    def camera_id(self) -> str:
        return str(getattr(self.camera, "cam_id", None) or getattr(self.camera, "id", None) or self.camera)

    def start(self) -> None:
        if self.capture_thread and self.capture_thread.is_alive():
            return
        if not getattr(self.runtime, "loaded", False):
            raise RuntimeError("Analytics runtime must be loaded before starting a camera pipeline")
        if self.analytics_thread and self.analytics_thread.is_alive():
            raise RuntimeError("Previous analytics thread is still stopping")
        self.buffer = LatestFrameBuffer()
        self.stop_event.clear()
        self._started_mono = time.monotonic()
        self.capture_thread = threading.Thread(target=self.capture_loop, name=f"capture-{self.camera_id}", daemon=True)
        self.analytics_thread = threading.Thread(target=self.analytics_loop, name=f"analytics-{self.camera_id}", daemon=True)
        self.capture_thread.start(); self.analytics_thread.start()

    def capture_loop(self) -> None:
        delay = self.reconnect_initial_seconds
        capture = None
        next_preview = 0.0
        playback = PlaybackClock()
        is_upload = str(getattr(self.camera, "source_type", "")).lower() in {"upload", "uploaded", "file"}
        try:
            while not self.stop_event.is_set():
                if capture is None:
                    try:
                        capture = self.capture_factory(self.camera)
                    except Exception as exc:
                        self.last_error = f"capture_open: {type(exc).__name__}: {exc}"
                        self.reconnect_count += 1
                        self.stop_event.wait(delay); delay=min(self.reconnect_max_seconds,delay*2.0); continue
                try:
                    decoded = capture.read_timestamped() if hasattr(capture, "read_timestamped") else capture.read()
                    if hasattr(decoded, "ok"):
                        ok, frame = decoded.ok, decoded.frame; pts=getattr(decoded,"pts_seconds",None)
                    else:
                        ok, frame = decoded[0], decoded[1]; pts=None
                    if not ok or frame is None:
                        if is_upload:
                            # EOF is terminal, not a reconnect/replay request.
                            self.last_error = None
                            self.stop_event.wait()
                            break
                        raise RuntimeError("camera read failed")
                    if is_upload and playback.wait(self.stop_event, pts, (getattr(capture, "metadata", {}) or {}).get("fps")):
                        break
                    if self.stop_event.is_set(): break
                    now_wall=time.time(); now_mono=time.monotonic(); self.frames_captured += 1
                    self.last_frame_timestamp = now_wall
                    self._last_capture_mono = now_mono
                    item=self.buffer.put(frame,now_wall,pts_seconds=pts, captured_monotonic=now_mono)
                    # Preview is emitted directly from decoded frames and never waits for AI.
                    if self.on_preview is not None and now_mono >= next_preview:
                        try:
                            self.on_preview(frame,item); self.preview_frames += 1
                        except Exception:
                            logger.exception("Preview publication failed | camera_id=%s",self.camera_id)
                        next_preview=now_mono+(1.0/self.preview_fps)
                    delay=self.reconnect_initial_seconds; self.last_error=None
                except Exception as exc:
                    if self.stop_event.is_set(): break
                    self.last_error=f"capture_read: {type(exc).__name__}: {exc}"; self.reconnect_count += 1
                    try: capture.release()
                    except Exception: pass
                    capture=None; self.stop_event.wait(delay); delay=min(self.reconnect_max_seconds,delay*2.0)
        finally:
            if capture is not None:
                try: capture.release()
                except Exception: logger.debug("Capture release failed",exc_info=True)

    def analytics_loop(self) -> None:
        sequence=0
        fallback_next_run=time.monotonic()
        fallback_interval=1.0/self.analytics_fps
        while not self.stop_event.is_set():
            item=self.buffer.get(after_sequence=sequence,timeout=0.5)
            if item is None: continue
            # Consume newest sequence immediately; stale frames are intentionally skipped.
            sequence=item.sequence
            self._last_consumed_sequence=sequence
            adaptive=hasattr(self.runtime,"should_submit")
            if adaptive:
                try:
                    if not self.runtime.should_submit(self.camera):
                        self.analytics_skipped += 1; continue
                except Exception:
                    logger.debug("Adaptive scheduler check failed; using fallback cadence",exc_info=True)
                    adaptive=False
            if not adaptive:
                wait=fallback_next_run-time.monotonic()
                if wait>0 and self.stop_event.wait(wait): break
                fallback_next_run=max(fallback_next_run+fallback_interval,time.monotonic())
            if time.monotonic() - item.metadata.get("captured_monotonic", time.monotonic()) > self.frame_ttl:
                self.analytics_skipped += 1
                continue
            try:
                priority=self.runtime.priority_for(self.camera) if hasattr(self.runtime,"priority_for") else 5
                processed=self.runtime.process_frame(
                    self.camera,item.frame,
                    item.metadata.get("pts_seconds") if is_upload_source(self.camera) and item.metadata.get("pts_seconds") is not None else item.timestamp,
                    frame_count=self.frames_processed,
                    source_pts_seconds=item.metadata.get("pts_seconds"),
                    frame_timestamp=datetime.fromtimestamp(item.timestamp, timezone.utc),
                    timestamp_source="SERVER_RECEIVE_TIME",
                    timestamp_quality="ESTIMATED",
                    priority=priority,
                )
                self.frames_processed += 1; self._last_processed_mono=time.monotonic()
                if self.on_processed is not None: self.on_processed(processed,item)
                self.last_error=None
            except InferenceBackpressureError:
                # Expected graceful degradation: the frame is expendable, the
                # preview and stream stay healthy, and the adaptive scheduler
                # will reduce subsequent non-critical submissions.
                self.analytics_skipped += 1
                self.last_error=None
                logger.debug("Inference frame shed under pressure | camera_id=%s",self.camera_id)
            except Exception as exc:
                self.last_error=f"analytics: {type(exc).__name__}: {exc}"
                logger.exception("Camera analytics failed | camera_id=%s",self.camera_id)

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set(); self.buffer.close(); deadline=time.monotonic()+max(0.0,float(timeout))
        for thread in (self.capture_thread,self.analytics_thread):
            if thread and thread.is_alive(): thread.join(max(0.0,deadline-time.monotonic()))
        self.runtime.reset_camera(self.camera_id)

    def status(self) -> dict[str, Any]:
        elapsed=max(1e-6,time.monotonic()-self._started_mono)
        target=self.runtime.target_fps(self.camera) if hasattr(self.runtime,"target_fps") else self.analytics_fps
        return {
            "camera_id":self.camera_id,
            "running":bool(self.capture_thread and self.capture_thread.is_alive() and self.analytics_thread and self.analytics_thread.is_alive()),
            "frames_captured":self.frames_captured,"preview_frames":self.preview_frames,"frames_processed":self.frames_processed,
            "analytics_skipped":self.analytics_skipped,"dropped_frames":self.buffer.dropped_frames,
            "queue_depth":self.buffer.pending_after(self._last_consumed_sequence),"queue_capacity":1,
            "effective_detection_fps":round(self.frames_processed/elapsed,3),"target_detection_fps":round(float(target),3),
            "last_successful_frame_timestamp": self.last_frame_timestamp,
            "stream_stuck": not is_upload_source(self.camera) and time.monotonic() - self._last_capture_mono > self.stuck_seconds,
            "reconnect_count":self.reconnect_count,"last_error":self.last_error,
        }
