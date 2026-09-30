import threading
from typing import Any, Dict, Optional


MAX_CAMERA_ID_LENGTH = 256
MAX_MODE_LENGTH = 64

DEFAULT_MODE = "not_restricted"


class CameraConfigCache:
    """
    Thread-safe runtime cache for per-camera configuration.

    Existing callers are preserved:
        set_mode(cam_id, mode)
        get_mode(cam_id)
        remove(cam_id)
        clear()

    Additional stream properties are supported so the camera worker
    can use one consistent per-camera configuration object.

    The cache never stores secrets such as:
        - passwords
        - JWT tokens
        - private keys
        - API keys

    RTSP credentials should remain in the protected source/config
    layer and must never be copied into this cache.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Existing behavior.
        self._camera_modes: Dict[str, str] = {}

        # Runtime stream properties.
        self._camera_properties: Dict[str, Dict[str, Any]] = {}

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _normalize_camera_id(
        cam_id: Any,
    ) -> Optional[str]:
        if cam_id is None:
            return None

        try:
            value = str(cam_id).strip()
        except Exception:
            return None

        if not value:
            return None

        if len(value) > MAX_CAMERA_ID_LENGTH:
            return None

        return value

    @staticmethod
    def _normalize_mode(
        mode: Any,
    ) -> str:
        if mode is None:
            return DEFAULT_MODE

        try:
            value = str(mode).strip()
        except Exception:
            return DEFAULT_MODE

        if not value:
            return DEFAULT_MODE

        if len(value) > MAX_MODE_LENGTH:
            return DEFAULT_MODE

        return value

    @staticmethod
    def _safe_fps(
        fps: Any,
        default: float = 15.0,
    ) -> float:
        try:
            value = float(fps)
        except (TypeError, ValueError):
            return default

        if value <= 0:
            return default

        if value != value or value in (
            float("inf"),
            float("-inf"),
        ):
            return default

        # Keep configuration bounded.
        return max(
            1.0,
            min(
                120.0,
                value,
            ),
        )

    @staticmethod
    def _safe_dimension(
        value: Any,
        default: int,
    ) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return default

        return max(
            1,
            min(
                16384,
                result,
            ),
        )

    # ========================================================
    # EXISTING MODE API
    # ========================================================

    def set_mode(
        self,
        cam_id,
        mode,
    ):
        """
        Preserve the original camera-mode API.
        """

        safe_cam_id = self._normalize_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return

        safe_mode = self._normalize_mode(
            mode
        )

        with self._lock:
            self._camera_modes[
                safe_cam_id
            ] = safe_mode

            properties = self._camera_properties.setdefault(
                safe_cam_id,
                {},
            )

            properties[
                "mode"
            ] = safe_mode

    def get_mode(
        self,
        cam_id,
    ):
        safe_cam_id = self._normalize_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return DEFAULT_MODE

        with self._lock:
            return self._camera_modes.get(
                safe_cam_id,
                DEFAULT_MODE,
            )

    # ========================================================
    # STREAM CONFIGURATION
    # ========================================================

    def set_stream_config(
        self,
        cam_id,
        *,
        source_type=None,
        fps=None,
        width=None,
        height=None,
        transport=None,
        codec=None,
    ):
        """
        Store non-secret stream properties for a camera.

        Important:
            The actual source URL/password is intentionally NOT
            accepted here. Keep credentials in the protected camera
            configuration/database layer.

        transport defaults to TCP for RTSP because INTEL-I's
        submission requirement is RTSP-over-TCP.
        """

        safe_cam_id = self._normalize_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return

        with self._lock:
            properties = self._camera_properties.setdefault(
                safe_cam_id,
                {},
            )

            if source_type is not None:
                value = str(
                    source_type
                ).strip().lower()

                if len(value) <= 32:
                    properties[
                        "source_type"
                    ] = value

            if fps is not None:
                properties[
                    "fps"
                ] = self._safe_fps(
                    fps
                )

            if width is not None:
                properties[
                    "width"
                ] = self._safe_dimension(
                    width,
                    960,
                )

            if height is not None:
                properties[
                    "height"
                ] = self._safe_dimension(
                    height,
                    540,
                )

            if transport is not None:
                value = str(
                    transport
                ).strip().lower()

                # RTSP transport is intentionally restricted to TCP.
                if value in {
                    "tcp",
                    "rtsp-tcp",
                }:
                    properties[
                        "transport"
                    ] = "tcp"

                elif value in {
                    "",
                    "auto",
                    "default",
                }:
                    # For RTSP, default to TCP.
                    properties[
                        "transport"
                    ] = "tcp"

            if codec is not None:
                value = str(
                    codec
                ).strip().lower()

                if len(value) <= 32:
                    properties[
                        "codec"
                    ] = value

            # Always enforce TCP for RTSP-compatible stream config.
            if properties.get(
                "source_type"
            ) == "rtsp":
                properties[
                    "transport"
                ] = "tcp"

    def get_stream_config(
        self,
        cam_id,
    ) -> Dict[str, Any]:
        """
        Return a defensive copy of the camera's non-secret
        stream configuration.

        No RTSP URL or credentials are returned from this cache.
        """

        safe_cam_id = self._normalize_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return {}

        with self._lock:
            config = dict(
                self._camera_properties.get(
                    safe_cam_id,
                    {},
                )
            )

        if config.get(
            "source_type"
        ) == "rtsp":
            config[
                "transport"
            ] = "tcp"

        return config

    # ========================================================
    # GENERIC SAFE PROPERTY ACCESS
    # ========================================================

    def set_property(
        self,
        cam_id,
        key,
        value,
    ):
        """
        Store a small non-sensitive camera property.

        Secret-bearing keys are explicitly rejected.
        """

        safe_cam_id = self._normalize_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return False

        try:
            safe_key = str(
                key
            ).strip().lower()
        except Exception:
            return False

        if not safe_key or len(safe_key) > 64:
            return False

        forbidden_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "private_key",
            "api_key",
            "authorization",
            "credential",
            "credentials",
            "rtsp_password",
        }

        if safe_key in forbidden_keys:
            return False

        with self._lock:
            self._camera_properties.setdefault(
                safe_cam_id,
                {},
            )[
                safe_key
            ] = value

        return True

    def get_property(
        self,
        cam_id,
        key,
        default=None,
    ):
        safe_cam_id = self._normalize_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return default

        try:
            safe_key = str(
                key
            ).strip().lower()
        except Exception:
            return default

        if not safe_key:
            return default

        forbidden_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "private_key",
            "api_key",
            "authorization",
            "credential",
            "credentials",
            "rtsp_password",
        }

        if safe_key in forbidden_keys:
            return default

        with self._lock:
            return self._camera_properties.get(
                safe_cam_id,
                {},
            ).get(
                safe_key,
                default,
            )

    # ========================================================
    # SNAPSHOT / DEBUG-SAFE VIEW
    # ========================================================

    def get_safe_snapshot(
        self,
        cam_id,
    ) -> Dict[str, Any]:
        """
        Return only configuration metadata safe for diagnostics.

        Never returns source URLs, credentials, or authentication
        material.
        """

        config = self.get_stream_config(
            cam_id
        )

        return {
            "mode": self.get_mode(
                cam_id
            ),
            "source_type": config.get(
                "source_type"
            ),
            "fps": config.get(
                "fps",
                15.0,
            ),
            "width": config.get(
                "width",
                960,
            ),
            "height": config.get(
                "height",
                540,
            ),
            "transport": (
                "tcp"
                if config.get(
                    "source_type"
                ) == "rtsp"
                else config.get(
                    "transport"
                )
            ),
            "codec": config.get(
                "codec",
                "auto",
            ),
        }

    # ========================================================
    # CAMERA LIFECYCLE
    # ========================================================

    def remove(
        self,
        cam_id,
    ):
        safe_cam_id = self._normalize_camera_id(
            cam_id
        )

        if safe_cam_id is None:
            return

        with self._lock:
            self._camera_modes.pop(
                safe_cam_id,
                None,
            )

            self._camera_properties.pop(
                safe_cam_id,
                None,
            )

    def clear(self):
        with self._lock:
            self._camera_modes.clear()
            self._camera_properties.clear()


camera_config = CameraConfigCache()

# ========================================================
# GIS CAMERA METADATA
# ========================================================

MAX_LATITUDE = 90.0
MAX_LONGITUDE = 180.0
MAX_CAMERA_NAME_LENGTH = 128
MAX_ZONE_NAME_LENGTH = 128


def _safe_coordinate(value: Any, minimum: float, maximum: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if number != number or number in (float("inf"), float("-inf")):
        return None

    if number < minimum or number > maximum:
        return None

    return number


def _safe_optional_text(value: Any, maximum_length: int) -> Optional[str]:
    if value is None:
        return None

    try:
        text = str(value).strip()
    except Exception:
        return None

    if not text or len(text) > maximum_length:
        return None

    return text


class CameraGISConfig:
    """
    Thread-safe camera GIS metadata cache.

    Stores only camera identity/coordinate metadata. It intentionally
    does not accept or store stream URLs, usernames, passwords, tokens,
    API keys, embeddings, or vehicle identities.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cameras: Dict[str, Dict[str, Any]] = {}

    def set_camera(
        self,
        cam_id,
        latitude,
        longitude,
        camera_name=None,
        zone=None,
    ) -> bool:
        safe_cam_id = CameraConfigCache._normalize_camera_id(cam_id)
        if safe_cam_id is None:
            return False

        safe_latitude, safe_longitude = self._validate_coordinates(
            latitude,
            longitude,
        )

        if safe_latitude is None or safe_longitude is None:
            return False

        safe_name = _safe_optional_text(
            camera_name,
            MAX_CAMERA_NAME_LENGTH,
        )
        safe_zone = _safe_optional_text(
            zone,
            MAX_ZONE_NAME_LENGTH,
        )

        with self._lock:
            self._cameras[safe_cam_id] = {
                "camera_id": safe_cam_id,
                "camera_name": safe_name,
                "latitude": safe_latitude,
                "longitude": safe_longitude,
                "zone": safe_zone,
            }

        return True

    def update_camera(
        self,
        cam_id,
        *,
        latitude=None,
        longitude=None,
        camera_name=None,
        zone=None,
    ) -> bool:
        safe_cam_id = CameraConfigCache._normalize_camera_id(cam_id)
        if safe_cam_id is None:
            return False

        with self._lock:
            existing = dict(
                self._cameras.get(
                    safe_cam_id,
                    {},
                )
            )

        current_latitude = (
            existing.get("latitude")
            if latitude is None
            else latitude
        )
        current_longitude = (
            existing.get("longitude")
            if longitude is None
            else longitude
        )

        safe_latitude, safe_longitude = self._validate_coordinates(
            current_latitude,
            current_longitude,
        )

        if safe_latitude is None or safe_longitude is None:
            return False

        safe_name = (
            existing.get("camera_name")
            if camera_name is None
            else _safe_optional_text(
                camera_name,
                MAX_CAMERA_NAME_LENGTH,
            )
        )

        safe_zone = (
            existing.get("zone")
            if zone is None
            else _safe_optional_text(
                zone,
                MAX_ZONE_NAME_LENGTH,
            )
        )

        with self._lock:
            self._cameras[safe_cam_id] = {
                "camera_id": safe_cam_id,
                "camera_name": safe_name,
                "latitude": safe_latitude,
                "longitude": safe_longitude,
                "zone": safe_zone,
            }

        return True

    @staticmethod
    def _validate_coordinates(latitude, longitude):
        return (
            _safe_coordinate(
                latitude,
                -MAX_LATITUDE,
                MAX_LATITUDE,
            ),
            _safe_coordinate(
                longitude,
                -MAX_LONGITUDE,
                MAX_LONGITUDE,
            ),
        )

    def get_camera(self, cam_id):
        safe_cam_id = CameraConfigCache._normalize_camera_id(cam_id)
        if safe_cam_id is None:
            return None

        with self._lock:
            camera = self._cameras.get(safe_cam_id)
            return None if camera is None else dict(camera)

    def get_coordinates(self, cam_id):
        camera = self.get_camera(cam_id)
        if camera is None:
            return None

        return {
            "camera_id": camera["camera_id"],
            "latitude": camera["latitude"],
            "longitude": camera["longitude"],
        }

    def get_all(self):
        with self._lock:
            return [
                dict(camera)
                for camera in self._cameras.values()
            ]

    def remove(self, cam_id):
        safe_cam_id = CameraConfigCache._normalize_camera_id(cam_id)
        if safe_cam_id is None:
            return

        with self._lock:
            self._cameras.pop(safe_cam_id, None)

    def clear(self):
        with self._lock:
            self._cameras.clear()


camera_gis_config = CameraGISConfig()


def set_camera_gis(
    cam_id,
    latitude,
    longitude,
    camera_name=None,
    zone=None,
) -> bool:
    """Register validated GIS metadata for a camera."""
    return camera_gis_config.set_camera(
        cam_id,
        latitude,
        longitude,
        camera_name,
        zone,
    )


def get_camera_gis(cam_id):
    """Return GIS-safe metadata without stream credentials."""
    return camera_gis_config.get_camera(cam_id)


def get_camera_coordinates(cam_id):
    """Return only validated camera coordinates."""
    return camera_gis_config.get_coordinates(cam_id)


def get_all_camera_gis():
    """Return defensive copies of all GIS-safe camera metadata."""
    return camera_gis_config.get_all()


def remove_camera_gis(cam_id):
    camera_gis_config.remove(cam_id)
