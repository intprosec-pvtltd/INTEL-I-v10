"""Decoder abstraction that exposes presentation timestamps with each frame."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import re
import socket
from typing import Any, Callable
from urllib.parse import quote, urlparse


logger = logging.getLogger(__name__)
try:
    from inference.hardware_decode import detect_capability, make_pyav_hwaccel, pyav_options
except Exception:
    detect_capability=lambda force=False: {"selected_device":"cpu","nvdec_available":False,"fallback":True}
    pyav_options=lambda: {}
    make_pyav_hwaccel=lambda av_module: None


# ============================================================
# Sentinel configuration
# ============================================================

_SENTINEL_DEFAULT_HOST = "103.250.160.189"
_SENTINEL_DEFAULT_PORT = 8554

_SENTINEL_CAMERA_RE = re.compile(
    r"^cam(?:0[1-9]|[12][0-9]|30)$",
    re.IGNORECASE,
)


# ============================================================
# Errors / data model
# ============================================================


class CameraSourceError(RuntimeError):
    """A safe, classified camera-source connection error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class DecodedFrame:
    ok: bool
    frame: Any = None
    pts_seconds: float | None = None
    time_base: str | None = None
    key_frame: bool = False


# ============================================================
# Generic helpers
# ============================================================


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        ).strip()

    return str(value or "").strip()


def _endpoint_port(parsed) -> int:
    if parsed.port:
        return int(parsed.port)

    if parsed.scheme.lower() in {
        "rtsp",
        "rtsps",
    }:
        return 554

    if parsed.scheme.lower() == "https":
        return 443

    return 80


# ============================================================
# Sentinel helpers
# ============================================================


def _sentinel_host() -> str:
    return os.getenv(
        "SENTINEL_RTSP_HOST",
        _SENTINEL_DEFAULT_HOST,
    ).strip()


def _sentinel_port() -> int:
    raw = os.getenv(
        "SENTINEL_RTSP_PORT",
        str(_SENTINEL_DEFAULT_PORT),
    ).strip()

    try:
        port = int(raw)

    except ValueError as exc:
        raise CameraSourceError(
            "SENTINEL_CONFIG_INVALID",
            "SENTINEL_RTSP_PORT must be a valid integer",
        ) from exc

    if not (1 <= port <= 65535):
        raise CameraSourceError(
            "SENTINEL_CONFIG_INVALID",
            "SENTINEL_RTSP_PORT must be between 1 and 65535",
        )

    return port


def _extract_sentinel_camera_id(
    source: Any,
) -> str:
    """
    Supported input formats:

        cam01

        /stream/cam01

        rtsp://103.250.160.189:8554/stream/cam01

        rtsp://user:password@
        103.250.160.189:8554/stream/cam01
    """

    text = _as_text(source)

    if not text:
        raise CameraSourceError(
            "INVALID_SENTINEL_SOURCE",
            "Sentinel camera source is empty",
        )

    candidate = text

    if "://" in text:
        parsed = urlparse(text)

        candidate = (
            parsed.path
            .rstrip("/")
            .split("/")[-1]
        )

    elif "/" in text:
        candidate = (
            text
            .rstrip("/")
            .split("/")[-1]
        )

    candidate = candidate.strip().lower()

    if not _SENTINEL_CAMERA_RE.fullmatch(
        candidate
    ):
        raise CameraSourceError(
            "INVALID_SENTINEL_SOURCE",
            (
                "Sentinel camera ID must be "
                "between cam01 and cam30"
            ),
        )

    return candidate


def _build_sentinel_rtsp_url(
    source: Any,
) -> str:
    """
    Build the authenticated Sentinel RTSP URL.

    Credentials are taken only from backend environment
    variables and are never required from the frontend.
    """

    camera_id = _extract_sentinel_camera_id(
        source
    )

    username = os.getenv(
        "SENTINEL_USERNAME",
        "",
    ).strip()

    password = os.getenv(
        "SENTINEL_PASSWORD",
        "",
    )

    if not username or not password:
        raise CameraSourceError(
            "SENTINEL_CREDENTIALS_MISSING",
            (
                "Sentinel RTSP credentials are "
                "not configured on the Intel-I backend"
            ),
        )

    encoded_username = quote(
        username,
        safe="",
    )

    encoded_password = quote(
        password,
        safe="",
    )

    return (
        f"rtsp://"
        f"{encoded_username}:"
        f"{encoded_password}"
        f"@{_sentinel_host()}:"
        f"{_sentinel_port()}"
        f"/stream/{camera_id}"
    )


def _is_sentinel_rtsp_source(
    source: Any,
) -> bool:
    """
    Detect a Sentinel URL even when the database currently
    stores the camera as source_type='rtsp'.

    Example:

        rtsp://103.250.160.189:8554/stream/cam01

    INTEL-I will automatically rebuild that URL with the
    backend Sentinel credentials before PyAV opens it.
    """

    text = _as_text(source)

    if not text.lower().startswith(
        (
            "rtsp://",
            "rtsps://",
        )
    ):
        return False

    try:
        parsed = urlparse(text)

        if not parsed.hostname:
            return False

        configured_host = (
            _sentinel_host().lower()
        )

        host_matches = (
            parsed.hostname.lower()
            == configured_host
        )

        port_matches = (
            _endpoint_port(parsed)
            == _sentinel_port()
        )

        path = (
            parsed.path or ""
        ).rstrip("/")

        camera_id = (
            path.split("/")[-1].lower()
            if path
            else ""
        )

        return (
            host_matches
            and port_matches
            and path.lower().startswith(
                "/stream/"
            )
            and bool(
                _SENTINEL_CAMERA_RE.fullmatch(
                    camera_id
                )
            )
        )

    except (
        ValueError,
        CameraSourceError,
    ):
        return False


# ============================================================
# Network preflight
# ============================================================


def preflight_stream_endpoint(
    source: Any,
    source_type: str,
    timeout_seconds: float,
) -> None:
    """
    Verify that the decoder host/port is reachable before
    opening media.

    This is intentionally a TCP-only probe.

    It does not:
      - consume the camera stream
      - expose authentication credentials
      - perform HTTP HEAD requests
    """

    normalized = str(
        source_type or ""
    ).strip().lower()

    if normalized not in {
        "rtsp",
        "http",
        "https",
        "hls",
    }:
        return

    parsed = urlparse(
        _as_text(source)
    )

    if parsed.scheme.lower() not in {
        "rtsp",
        "rtsps",
        "http",
        "https",
    }:
        raise CameraSourceError(
            "INVALID_SOURCE_URL",
            (
                "Camera source must use "
                "rtsp://, rtsps://, "
                "http://, or https://"
            ),
        )

    if not parsed.hostname:
        raise CameraSourceError(
            "INVALID_SOURCE_URL",
            "Camera source URL has no host",
        )

    timeout = max(
        1.0,
        min(
            float(timeout_seconds),
            10.0,
        ),
    )

    try:
        with socket.create_connection(
            (
                parsed.hostname,
                _endpoint_port(parsed),
            ),
            timeout=timeout,
        ):
            return

    except socket.gaierror as exc:
        raise CameraSourceError(
            "SOURCE_DNS_FAILED",
            (
                "Camera hostname could not "
                "be resolved"
            ),
        ) from exc

    except (
        TimeoutError,
        socket.timeout,
    ) as exc:
        raise CameraSourceError(
            "SOURCE_CONNECT_TIMEOUT",
            "Camera connection timed out",
        ) from exc

    except OSError as exc:
        raise CameraSourceError(
            "SOURCE_UNREACHABLE",
            (
                "Camera host or port is "
                "unreachable from this "
                "Intel-I server"
            ),
        ) from exc


# ============================================================
# Decoder error classification
# ============================================================


def _classify_decoder_error(
    exc: Exception,
) -> CameraSourceError:

    message = str(exc).lower()

    if any(
        token in message
        for token in (
            "401",
            "403",
            "unauthorized",
            "authentication",
            "authorization failed",
        )
    ):
        return CameraSourceError(
            "SOURCE_AUTH_FAILED",
            (
                "Camera authentication "
                "was rejected"
            ),
        )

    if any(
        token in message
        for token in (
            "timed out",
            "timeout",
        )
    ):
        return CameraSourceError(
            "SOURCE_READ_TIMEOUT",
            (
                "Camera stream did not respond "
                "before the timeout"
            ),
        )

    if any(
        token in message
        for token in (
            "404",
            "not found",
        )
    ):
        return CameraSourceError(
            "SOURCE_NOT_FOUND",
            (
                "Camera stream path "
                "was not found"
            ),
        )

    if any(
        token in message
        for token in (
            "invalid data",
            "invalid argument",
            "no video stream",
            "could not find codec parameters",
        )
    ):
        return CameraSourceError(
            "INVALID_MEDIA_STREAM",
            (
                "Camera source did not provide "
                "a supported video stream"
            ),
        )

    return CameraSourceError(
        "DECODER_OPEN_FAILED",
        (
            "Camera stream could not be "
            "opened by the media decoder"
        ),
    )


# ============================================================
# PyAV timestamped capture
# ============================================================


class PyAVTimestampedCapture:

    def __init__(
        self,
        source: Any,
        source_type: str,
        timeout_seconds: float,
    ):

        try:
            import av

        except ImportError as exc:
            raise RuntimeError(
                (
                    "PyAV is required for "
                    "PTS-based media decoding"
                )
            ) from exc

        self._av = av

        self.source = source

        self.source_type = str(
            source_type or ""
        ).strip().lower()

        self.timeout_seconds = max(
            1.0,
            min(
                60.0,
                float(timeout_seconds),
            ),
        )

        self.container = None

        # ----------------------------------------------------
        # Network reachability check
        # ----------------------------------------------------

        preflight_stream_endpoint(
            source=self.source,
            source_type=self.source_type,
            timeout_seconds=self.timeout_seconds,
        )

        # ----------------------------------------------------
        # FFmpeg / PyAV options
        # ----------------------------------------------------

        options: dict[str, str] = {}

        timeout_us = str(
            int(
                self.timeout_seconds
                * 1_000_000
            )
        )

        if self.source_type == "rtsp":

            options.update(
                {
                    # Sentinel requires TCP.
                    "rtsp_transport": "tcp",

                    # Read/write timeout.
                    "rw_timeout": timeout_us,

                    # RTSP socket timeout for
                    # compatible FFmpeg builds.
                    "stimeout": timeout_us,
                }
            )

        elif self.source_type in {
            "http",
            "https",
            "hls",
        }:

            options.update(
                {
                    "rw_timeout": timeout_us,
                    "reconnect": "1",
                    "reconnect_streamed": "1",
                    "reconnect_delay_max": "5",
                }
            )

        # Hardware decode is attempted only when FFmpeg advertises CUDA.
        # Any option/open failure retries with the original CPU options.
        base_options=dict(options)
        # Modern PyAV exposes a real HWAccel device-context API. Use it instead
        # of merely passing ffmpeg CLI-style dictionary options. The old option
        # helper remains a compatibility fallback for non-standard builds.
        hwaccel=make_pyav_hwaccel(av)
        hw_options={} if hwaccel is not None else pyav_options()
        if hw_options:
            options.update(hw_options)
        self.hardware_decode_requested=bool(hwaccel is not None or hw_options)
        self.hardware_decode_active=False

        # ----------------------------------------------------
        # Open stream
        # ----------------------------------------------------

        try:

            def _open_with(open_options, *, use_hwaccel=False):
                kwargs={
                    "mode":"r",
                    "options":open_options,
                    "timeout":(self.timeout_seconds,self.timeout_seconds),
                }
                if use_hwaccel and hwaccel is not None:
                    kwargs["hwaccel"]=hwaccel
                try:
                    return av.open(self.source,**kwargs)
                except TypeError:
                    # Older/non-standard PyAV builds may reject tuple timeout.
                    kwargs.pop("timeout",None)
                    return av.open(self.source,**kwargs)

            try:
                self.container=_open_with(options,use_hwaccel=hwaccel is not None)
                video_streams=list(self.container.streams.video)
                if hwaccel is not None and video_streams:
                    self.hardware_decode_active=bool(
                        getattr(getattr(video_streams[0],"codec_context",None),"is_hwaccel",False)
                    )
                    if not self.hardware_decode_active:
                        raise RuntimeError("PyAV opened stream but CUDA hardware decode is not active")
                else:
                    # Dictionary fallback cannot be trusted as proof of HWAccel;
                    # report active only when codec_context confirms it later.
                    self.hardware_decode_active=False
            except Exception as hw_exc:
                if hwaccel is None and not hw_options:
                    raise
                logger.warning(
                    "Hardware decode unavailable; retrying CPU decoder | source_type=%s error=%s",
                    self.source_type,type(hw_exc).__name__,
                )
                try:
                    if self.container is not None:
                        self.container.close()
                except Exception:
                    pass
                self.container=_open_with(base_options,use_hwaccel=False)
                self.hardware_decode_active=False

        except CameraSourceError:
            raise

        except Exception as exc:
            raise _classify_decoder_error(
                exc
            ) from exc

        # ----------------------------------------------------
        # Find video stream
        # ----------------------------------------------------

        video_streams = list(
            self.container.streams.video
        )

        if not video_streams:

            self.container.close()
            self.container = None

            raise CameraSourceError(
                "INVALID_MEDIA_STREAM",
                (
                    "Media source has "
                    "no video stream"
                ),
            )

        self.stream = video_streams[0]
        try:
            self.hardware_decode_active=bool(
                self.hardware_decode_active
                or getattr(getattr(self.stream,"codec_context",None),"is_hwaccel",False)
            )
        except Exception:
            pass

        # ----------------------------------------------------
        # Decoder threading
        # ----------------------------------------------------

        try:
            self.stream.thread_type = "AUTO"

        except Exception:
            pass

        # ----------------------------------------------------
        # Frame generator
        # ----------------------------------------------------

        self._frames = (
            self.container.decode(
                video=0
            )
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.metadata = {
            "decoder": "pyav",
            "hardware_decode_requested": bool(self.hardware_decode_requested),
            "hardware_decode_active": bool(self.hardware_decode_active),
            "hardware_decode": detect_capability(),

            "codec": getattr(
                getattr(
                    self.stream,
                    "codec_context",
                    None,
                ),
                "name",
                None,
            ),

            "width": getattr(
                self.stream,
                "width",
                None,
            ),

            "height": getattr(
                self.stream,
                "height",
                None,
            ),

            "fps": self._rate(),

            "transport": (
                "tcp"
                if self.source_type == "rtsp"
                else None
            ),
        }

    # ========================================================
    # FPS
    # ========================================================

    def _rate(
        self,
    ) -> float | None:

        for value in (
            getattr(
                self.stream,
                "average_rate",
                None,
            ),
            getattr(
                self.stream,
                "base_rate",
                None,
            ),
        ):

            try:
                result = float(value)

            except (
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                continue

            if (
                math.isfinite(result)
                and 0.1 <= result <= 240
            ):
                return result

        return None

    # ========================================================
    # Open status
    # ========================================================

    def isOpened(
        self,
    ) -> bool:

        return self.container is not None

    # ========================================================
    # Read timestamped frame
    # ========================================================

    def read_timestamped(
        self,
    ) -> DecodedFrame:

        if self.container is None:
            return DecodedFrame(
                False
            )

        try:
            decoded = next(
                self._frames
            )

        except (
            StopIteration,
            EOFError,
        ):
            return DecodedFrame(
                False
            )

        except Exception as exc:
            raise _classify_decoder_error(
                exc
            ) from exc

        # ----------------------------------------------------
        # Extract PTS
        # ----------------------------------------------------

        pts_seconds = None

        if (
            decoded.pts is not None
            and decoded.time_base
            is not None
        ):

            try:
                pts_seconds = float(
                    decoded.pts
                    * decoded.time_base
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                pts_seconds = None

        # ----------------------------------------------------
        # Convert to OpenCV BGR frame
        # ----------------------------------------------------

        try:
            frame = decoded.to_ndarray(
                format="bgr24"
            )

        except Exception as exc:
            raise CameraSourceError(
                "FRAME_CONVERSION_FAILED",
                (
                    "Decoded camera frame "
                    "could not be converted "
                    "to BGR format"
                ),
            ) from exc

        return DecodedFrame(
            ok=frame is not None,
            frame=frame,
            pts_seconds=pts_seconds,

            time_base=(
                str(
                    decoded.time_base
                )
                if decoded.time_base
                is not None
                else None
            ),

            key_frame=bool(
                getattr(
                    decoded,
                    "key_frame",
                    False,
                )
            ),
        )

    # ========================================================
    # OpenCV-compatible read()
    # ========================================================

    def read(
        self,
    ):
        packet = (
            self.read_timestamped()
        )

        return (
            packet.ok,
            packet.frame,
        )

    # ========================================================
    # Release
    # ========================================================

    def release(
        self,
    ) -> None:

        if self.container is not None:

            try:
                self.container.close()

            finally:
                self.container = None


# ============================================================
# OpenCV timestamp wrapper
# ============================================================


class OpenCVTimestampedCapture:

    def __init__(
        self,
        capture: Any,
        source_type: str,
    ):

        self.capture = capture

        self.source_type = str(
            source_type or ""
        ).strip().lower()

        self.metadata = {
            "decoder": "opencv",
            "fps": self._fps(),
        }

    # ========================================================
    # FPS
    # ========================================================

    def _fps(
        self,
    ) -> float | None:

        try:
            import cv2

            value = float(
                self.capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

        except Exception:
            return None

        return (
            value
            if (
                math.isfinite(value)
                and 0.1 <= value <= 240
            )
            else None
        )

    # ========================================================
    # Open status
    # ========================================================

    def isOpened(
        self,
    ) -> bool:

        return bool(
            self.capture
            and self.capture.isOpened()
        )

    # ========================================================
    # Read timestamped
    # ========================================================

    def read_timestamped(
        self,
    ) -> DecodedFrame:

        ok, frame = (
            self.capture.read()
        )

        if not ok or frame is None:
            return DecodedFrame(
                False
            )

        pts_seconds = None

        try:
            import cv2

            pos_msec = float(
                self.capture.get(
                    cv2.CAP_PROP_POS_MSEC
                )
            )

            if (
                math.isfinite(pos_msec)
                and pos_msec >= 0
                and (
                    pos_msec > 0
                    or self.source_type
                    in {
                        "upload",
                        "recorded_video",
                    }
                )
            ):
                pts_seconds = (
                    pos_msec / 1000.0
                )

            elif hasattr(
                cv2,
                "CAP_PROP_PTS",
            ):

                pts_units = float(
                    self.capture.get(
                        cv2.CAP_PROP_PTS
                    )
                )

                fps = (
                    self.metadata.get(
                        "fps"
                    )
                )

                if (
                    math.isfinite(
                        pts_units
                    )
                    and pts_units >= 0
                    and fps
                ):
                    pts_seconds = (
                        pts_units
                        / float(fps)
                    )

        except Exception:
            pts_seconds = None

        return DecodedFrame(
            ok=True,
            frame=frame,
            pts_seconds=pts_seconds,
        )

    # ========================================================
    # OpenCV-compatible read()
    # ========================================================

    def read(
        self,
    ):
        packet = (
            self.read_timestamped()
        )

        return (
            packet.ok,
            packet.frame,
        )

    # ========================================================
    # Release
    # ========================================================

    def release(
        self,
    ) -> None:

        if self.capture is not None:
            self.capture.release()


# ============================================================
# Source resolver
# ============================================================


def _resolve_source(
    source: Any,
    source_type: str,
) -> tuple[Any, str]:
    """
    Resolve connector-backed camera sources into concrete
    media sources.

    Sentinel is resolved here BEFORE PyAV opens the source.

    This prevents:

        401 Unauthorized

    when INTEL-I stores:

        source_type = rtsp

    and:

        rtsp://103.250.160.189:8554/stream/cam01

    without credentials.
    """

    normalized = str(
        source_type or ""
    ).strip().lower()

    # ========================================================
    # Sentinel-native mode
    #
    # Recommended database representation:
    #
    #   source_type = sentinel
    #   source      = cam01
    #
    # ========================================================

    if normalized == "sentinel":

        resolved_source = (
            _build_sentinel_rtsp_url(
                source
            )
        )

        logger.info(
            (
                "Sentinel camera source resolved "
                "| camera=%s "
                "| host=%s "
                "| port=%s "
                "| transport=tcp"
            ),
            _extract_sentinel_camera_id(
                source
            ),
            _sentinel_host(),
            _sentinel_port(),
        )

        return (
            resolved_source,
            "rtsp",
        )

    # ========================================================
    # Backward compatibility
    #
    # Existing INTEL-I records may already contain:
    #
    # source_type = rtsp
    #
    # source =
    # rtsp://103.250.160.189:8554/stream/cam01
    #
    # We detect that automatically and securely inject
    # authentication here.
    # ========================================================

    if (
        normalized == "rtsp"
        and _is_sentinel_rtsp_source(
            source
        )
    ):

        camera_id = (
            _extract_sentinel_camera_id(
                source
            )
        )

        resolved_source = (
            _build_sentinel_rtsp_url(
                source
            )
        )

        logger.info(
            (
                "Sentinel RTSP source detected "
                "| camera=%s "
                "| host=%s "
                "| port=%s "
                "| transport=tcp"
            ),
            camera_id,
            _sentinel_host(),
            _sentinel_port(),
        )

        return (
            resolved_source,
            "rtsp",
        )

    # ========================================================
    # Existing connectors
    # ========================================================

    if normalized in {
        "onvif",
        "nvr",
        "vms",
        "vendor_api",
        "vendor_sdk",
        "generic_vms",
    }:

        from connectors.manager import (
            resolve_connector,
        )

        result = resolve_connector(
            normalized,
            source,
        )

        return (
            result.source,
            str(
                result.source_type or ""
            ).strip().lower(),
        )

    # ========================================================
    # Normal RTSP / HLS / upload / etc.
    # ========================================================

    return (
        source,
        normalized,
    )


# ============================================================
# Public capture factory
# ============================================================


def open_timestamped_capture(
    source: Any,
    source_type: str,
    *,
    opencv_factory: Callable[
        [Any, str],
        Any,
    ],
    timeout_seconds: float,
):
    """
    Open direct or connector-provided media without changing
    the existing INTEL-I caller flow.

    Supported sources include:

        rtsp
        sentinel
        hls
        http
        https
        upload
        recorded_video
        onvif
        nvr
        vms
        vendor_api
        vendor_sdk
        generic_vms
    """

    resolved_source, resolved_type = (
        _resolve_source(
            source,
            source_type,
        )
    )

    # ========================================================
    # Connector may directly return an OpenCV-like capture
    # object instead of a URL.
    # ========================================================

    if not isinstance(
        resolved_source,
        (
            str,
            bytes,
        ),
    ):

        return OpenCVTimestampedCapture(
            resolved_source,
            resolved_type,
        )

    # ========================================================
    # Select decoder
    # ========================================================

    decoder = os.getenv(
        "MEDIA_DECODER",
        "pyav",
    ).strip().lower()

    pts_capable_type = (
        resolved_type
        in {
            "rtsp",
            "http",
            "https",
            "hls",
            "live",
            "upload",
            "recorded_video",
        }
    )

    # ========================================================
    # PyAV
    # ========================================================

    if (
        decoder == "pyav"
        and pts_capable_type
    ):

        try:
            return PyAVTimestampedCapture(
                resolved_source,
                resolved_type,
                timeout_seconds,
            )

        except CameraSourceError:
            raise

        except Exception:

            strict = os.getenv(
                "STRICT_PTS_REQUIRED",
                "true",
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

            if strict:
                raise

            logger.warning(
                (
                    "PyAV decoder unavailable; "
                    "falling back to OpenCV "
                    "without guaranteed source PTS"
                )
            )

    # ========================================================
    # OpenCV fallback
    # ========================================================

    capture = opencv_factory(
        resolved_source,
        resolved_type,
    )

    if capture is None:
        raise CameraSourceError(
            "DECODER_OPEN_FAILED",
            (
                "OpenCV could not open "
                "the camera stream"
            ),
        )

    return OpenCVTimestampedCapture(
        capture,
        resolved_type,
    )