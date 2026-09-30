"""
Production object storage abstraction for INTEL-I uploaded video evidence.

Primary production backend:
    MinIO / S3-compatible object storage

Development fallback:
    Local filesystem, only when explicitly selected or explicitly allowed.

Security and integrity properties:
- Private bucket/object access only; no public URLs are created.
- Object keys are validated and never used directly as arbitrary filesystem paths.
- SHA-256 is calculated for uploaded files and persisted as object metadata.
- Downloads can be SHA-256 verified before OpenCV/FFmpeg receives the file.
- Cache writes are atomic.
- MinIO failures do not silently downgrade to local storage in production.
- Credentials and private object data are never logged.
- MinIO imports are lazy so local development remains usable before the SDK
  is installed, as long as MinIO is not selected.

Expected environment variables:

    OBJECT_STORAGE_BACKEND=minio
    OBJECT_STORAGE_ALLOW_LOCAL_FALLBACK=false

    MINIO_ENDPOINT=minio:9000
    MINIO_ACCESS_KEY=<application-access-key>
    MINIO_SECRET_KEY=<application-secret-key>
    MINIO_VIDEO_BUCKET=intel-i-uploaded-videos
    MINIO_SECURE=false
    MINIO_REGION=
    MINIO_CERT_CHECK=true

    OBJECT_STORAGE_LOCAL_ROOT=uploads
    OBJECT_STORAGE_CACHE_DIR=/tmp/intel-i-object-cache
    OBJECT_STORAGE_CONNECT_TIMEOUT_SECONDS=5
    OBJECT_STORAGE_READ_TIMEOUT_SECONDS=120
    OBJECT_STORAGE_RETRIES=3

For a TLS-enabled external MinIO endpoint:
    MINIO_ENDPOINT=minio.example.gov:443
    MINIO_SECURE=true
    MINIO_CERT_CHECK=true

Do not put http:// or https:// in MINIO_ENDPOINT. The scheme is controlled by
MINIO_SECURE.

The Python package `minio` is required when OBJECT_STORAGE_BACKEND=minio.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional
from urllib.parse import urlsplit

logger = logging.getLogger("object-storage")


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

_OBJECT_KEY_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+=-]{0,254}$")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")

_DEFAULT_CHUNK_SIZE = 1024 * 1024
_MAX_METADATA_VALUE_LENGTH = 2048


class ObjectStorageError(RuntimeError):
    """Base class for storage failures safe to handle at service boundaries."""


class ObjectStorageConfigurationError(ObjectStorageError):
    """Storage configuration is invalid or incomplete."""


class ObjectStorageUnavailableError(ObjectStorageError):
    """Configured object storage is unavailable."""


class ObjectStorageIntegrityError(ObjectStorageError):
    """Stored/downloaded bytes failed integrity validation."""


class ObjectNotFoundError(ObjectStorageError):
    """Requested private object does not exist."""


@dataclass(frozen=True)
class StoredObject:
    bucket: str
    object_key: str
    size_bytes: int
    sha256: str
    content_type: str
    etag: Optional[str] = None
    version_id: Optional[str] = None
    backend: str = "minio"


@dataclass(frozen=True)
class ObjectStat:
    bucket: str
    object_key: str
    size_bytes: int
    content_type: Optional[str]
    sha256: Optional[str]
    etag: Optional[str]
    version_id: Optional[str]
    backend: str


@dataclass(frozen=True)
class StorageHealth:
    configured_backend: str
    active_backend: str
    healthy: bool
    bucket: str
    detail: str
    checked_at_epoch: float


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False

    raise ObjectStorageConfigurationError(
        f"{name} must be one of: true/false, 1/0, yes/no, on/off"
    )


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ObjectStorageConfigurationError(f"{name} must be an integer") from exc

    if value < minimum or value > maximum:
        raise ObjectStorageConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _is_production() -> bool:
    env = (
        os.getenv("ENV")
        or os.getenv("APP_ENV")
        or os.getenv("ENVIRONMENT")
        or "dev"
    )
    return env.strip().lower() in {"prod", "production"}


def _normalise_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    if not endpoint:
        raise ObjectStorageConfigurationError("MINIO_ENDPOINT is required")

    if "://" in endpoint:
        raise ObjectStorageConfigurationError(
            "MINIO_ENDPOINT must not include http:// or https://; use MINIO_SECURE"
        )

    parsed = urlsplit(f"//{endpoint}")
    if not parsed.hostname:
        raise ObjectStorageConfigurationError("MINIO_ENDPOINT is invalid")

    if parsed.username or parsed.password:
        raise ObjectStorageConfigurationError(
            "MINIO_ENDPOINT must not contain credentials"
        )

    return endpoint


def _validate_bucket_name(bucket: str) -> str:
    value = bucket.strip()
    if not _BUCKET_RE.fullmatch(value):
        raise ObjectStorageConfigurationError(
            "MINIO_VIDEO_BUCKET must be a valid lowercase S3 bucket name "
            "(3-63 characters)"
        )

    if ".." in value or ".-" in value or "-." in value:
        raise ObjectStorageConfigurationError("MINIO_VIDEO_BUCKET is invalid")

    # S3-compatible bucket names that look exactly like IPv4 addresses are
    # intentionally rejected.
    parts = value.split(".")
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        raise ObjectStorageConfigurationError(
            "MINIO_VIDEO_BUCKET must not be formatted as an IPv4 address"
        )

    return value


def validate_object_key(object_key: str) -> str:
    """
    Validate and normalize a private storage key.

    Slashes are allowed to provide namespaces, but empty, '.', '..', control
    characters, backslashes, absolute paths and unsafe filesystem-like path
    segments are rejected.
    """
    if not isinstance(object_key, str):
        raise ValueError("object_key must be a string")

    key = object_key.strip()
    if not key:
        raise ValueError("object_key is required")
    if len(key.encode("utf-8")) > 1024:
        raise ValueError("object_key exceeds 1024 UTF-8 bytes")
    if key.startswith("/") or key.endswith("/"):
        raise ValueError("object_key must be relative and must identify an object")
    if "\\" in key:
        raise ValueError("object_key must use '/' separators only")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in key):
        raise ValueError("object_key contains control characters")

    segments = key.split("/")
    for segment in segments:
        if segment in {"", ".", ".."}:
            raise ValueError("object_key contains an unsafe path segment")
        if not _OBJECT_KEY_SEGMENT_RE.fullmatch(segment):
            raise ValueError(
                "object_key contains unsupported characters; use letters, numbers, "
                "'.', '_', '@', '+', '=', '-' and '/'"
            )

    return "/".join(segments)


def sha256_file(path: str | Path, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def _clean_content_type(content_type: Optional[str], path: Path) -> str:
    if content_type:
        value = content_type.strip().lower()
        if (
            value
            and len(value) <= 255
            and "\r" not in value
            and "\n" not in value
        ):
            return value

    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _safe_metadata(metadata: Optional[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    if not metadata:
        return result

    for raw_key, raw_value in metadata.items():
        key = str(raw_key).strip().lower().replace("_", "-")
        if not key:
            continue
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", key):
            raise ValueError(f"Invalid object metadata key: {raw_key!r}")

        value = str(raw_value)
        if "\r" in value or "\n" in value:
            raise ValueError(f"Invalid object metadata value for {key}")
        if len(value) > _MAX_METADATA_VALUE_LENGTH:
            raise ValueError(f"Object metadata value too long for {key}")

        # Let the MinIO SDK add the x-amz-meta prefix.
        result[key] = value

    return result


def _extract_sha256_from_metadata(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not metadata:
        return None

    wanted = {
        "sha256",
        "x-amz-meta-sha256",
        "x-amz-meta-intel-i-sha256",
        "intel-i-sha256",
    }
    for key, value in metadata.items():
        if str(key).strip().lower() in wanted:
            candidate = str(value).strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", candidate):
                return candidate
    return None


class ObjectStorage:
    """
    INTEL-I production object-storage service.

    Create one process-level instance with get_object_storage(). Methods are
    thread-safe for client initialization and per-cache-object materialization.
    The MinIO SDK itself maintains its own HTTP connection pool.
    """

    def __init__(self) -> None:
        configured_backend = os.getenv("OBJECT_STORAGE_BACKEND", "").strip().lower()
        if not configured_backend:
            configured_backend = "minio" if _is_production() else "filesystem"

        if configured_backend not in {"minio", "filesystem"}:
            raise ObjectStorageConfigurationError(
                "OBJECT_STORAGE_BACKEND must be 'minio' or 'filesystem'"
            )

        self.configured_backend = configured_backend
        self.allow_local_fallback = _env_bool(
            "OBJECT_STORAGE_ALLOW_LOCAL_FALLBACK",
            False,
        )

        if _is_production() and configured_backend == "filesystem":
            if not self.allow_local_fallback:
                raise ObjectStorageConfigurationError(
                    "Production object storage cannot use the filesystem unless "
                    "OBJECT_STORAGE_ALLOW_LOCAL_FALLBACK=true is explicitly set"
                )
            logger.warning(
                "INTEL-I is using filesystem object storage in production because "
                "OBJECT_STORAGE_ALLOW_LOCAL_FALLBACK=true. This is not recommended "
                "for multi-node production."
            )

        self.bucket = _validate_bucket_name(
            os.getenv("MINIO_VIDEO_BUCKET", "intel-i-uploaded-videos")
        )

        local_root_raw = os.getenv("OBJECT_STORAGE_LOCAL_ROOT", "uploads").strip()
        if not local_root_raw:
            raise ObjectStorageConfigurationError(
                "OBJECT_STORAGE_LOCAL_ROOT must not be empty"
            )
        self.local_root = Path(local_root_raw).expanduser().resolve()

        cache_raw = os.getenv(
            "OBJECT_STORAGE_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "intel-i-object-cache"),
        ).strip()
        if not cache_raw:
            raise ObjectStorageConfigurationError(
                "OBJECT_STORAGE_CACHE_DIR must not be empty"
            )
        self.cache_dir = Path(cache_raw).expanduser().resolve()

        self.connect_timeout = _env_int(
            "OBJECT_STORAGE_CONNECT_TIMEOUT_SECONDS", 5, 1, 120
        )
        self.read_timeout = _env_int(
            "OBJECT_STORAGE_READ_TIMEOUT_SECONDS", 120, 5, 3600
        )
        self.retries = _env_int("OBJECT_STORAGE_RETRIES", 3, 0, 10)

        self._client: Any = None
        self._client_lock = threading.RLock()
        self._cache_locks: dict[str, threading.Lock] = {}
        self._cache_locks_guard = threading.Lock()

        self._endpoint: Optional[str] = None
        self._access_key: Optional[str] = None
        self._secret_key: Optional[str] = None
        self._secure = False
        self._cert_check = True
        self._region: Optional[str] = None

        if configured_backend == "minio":
            self._load_minio_config()

        # Cache is ephemeral and may be shared by multiple processing paths.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if configured_backend == "filesystem":
            self.local_root.mkdir(parents=True, exist_ok=True)

    @property
    def active_backend(self) -> str:
        return self.configured_backend

    def _load_minio_config(self) -> None:
        self._endpoint = _normalise_endpoint(os.getenv("MINIO_ENDPOINT", ""))

        access_key = (
            os.getenv("MINIO_ACCESS_KEY")
            or os.getenv("MINIO_ROOT_USER")
            or ""
        ).strip()
        secret_key = (
            os.getenv("MINIO_SECRET_KEY")
            or os.getenv("MINIO_ROOT_PASSWORD")
            or ""
        ).strip()

        if not access_key:
            raise ObjectStorageConfigurationError(
                "MINIO_ACCESS_KEY is required when OBJECT_STORAGE_BACKEND=minio"
            )
        if not secret_key:
            raise ObjectStorageConfigurationError(
                "MINIO_SECRET_KEY is required when OBJECT_STORAGE_BACKEND=minio"
            )
        if len(secret_key) < 8:
            raise ObjectStorageConfigurationError(
                "MINIO_SECRET_KEY is too short"
            )

        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = _env_bool("MINIO_SECURE", False)
        self._cert_check = _env_bool("MINIO_CERT_CHECK", True)

        region = os.getenv("MINIO_REGION", "").strip()
        self._region = region or None

        if _is_production() and not self._secure:
            # Internal Docker networks can legitimately use HTTP while ingress
            # traffic remains HTTPS. Make the condition visible without
            # preventing that common topology.
            logger.info(
                "MinIO transport is HTTP inside the application network. "
                "Use MINIO_SECURE=true when MinIO is reached across an untrusted "
                "network or via a TLS-enabled endpoint."
            )

        if self._secure and not self._cert_check:
            logger.warning(
                "MINIO_CERT_CHECK=false disables TLS certificate verification. "
                "Do not use this setting in production."
            )
            if _is_production():
                raise ObjectStorageConfigurationError(
                    "MINIO_CERT_CHECK must be true for TLS-enabled production MinIO"
                )

    def _get_minio_client(self) -> Any:
        if self.configured_backend != "minio":
            raise ObjectStorageConfigurationError("MinIO backend is not active")

        with self._client_lock:
            if self._client is not None:
                return self._client

            try:
                from minio import Minio
                import urllib3
                from urllib3.util.retry import Retry
            except ImportError as exc:
                raise ObjectStorageConfigurationError(
                    "MinIO backend is enabled but the Python package 'minio' is "
                    "not installed. Add a pinned minio dependency to requirements.txt."
                ) from exc

            retry = Retry(
                total=self.retries,
                connect=self.retries,
                read=self.retries,
                status=self.retries,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(
                    {"DELETE", "GET", "HEAD", "OPTIONS", "PUT"}
                ),
                raise_on_status=False,
            )

            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(
                    connect=float(self.connect_timeout),
                    read=float(self.read_timeout),
                ),
                retries=retry,
                cert_reqs="CERT_REQUIRED" if self._cert_check else "CERT_NONE",
            )

            self._client = Minio(
                endpoint=self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
                region=self._region,
                http_client=http_client,
            )

            return self._client

    def ensure_ready(self, create_bucket: bool = True) -> None:
        """
        Verify backend readiness.

        MinIO buckets are private by default. This method deliberately does not
        configure a public bucket policy.
        """
        if self.configured_backend == "filesystem":
            self.local_root.mkdir(parents=True, exist_ok=True)
            probe = self.local_root / ".intel-i-storage-probe"
            try:
                probe.write_bytes(b"ok")
                probe.unlink(missing_ok=True)
            except OSError as exc:
                raise ObjectStorageUnavailableError(
                    "Filesystem object storage is not writable"
                ) from exc
            return

        client = self._get_minio_client()
        try:
            if client.bucket_exists(self.bucket):
                return
            if not create_bucket:
                raise ObjectStorageUnavailableError(
                    "Configured MinIO video bucket does not exist"
                )
            client.make_bucket(self.bucket, location=self._region)
            logger.info("INTEL-I private MinIO video bucket is ready")
        except ObjectStorageError:
            raise
        except Exception as exc:
            raise ObjectStorageUnavailableError(
                "MinIO is unavailable or the video bucket cannot be initialized"
            ) from exc

    def health(self) -> StorageHealth:
        now = time.time()
        try:
            self.ensure_ready(create_bucket=False)
            return StorageHealth(
                configured_backend=self.configured_backend,
                active_backend=self.active_backend,
                healthy=True,
                bucket=self.bucket,
                detail="ready",
                checked_at_epoch=now,
            )
        except ObjectStorageError as exc:
            # Keep health details free of endpoints, credentials and raw SDK
            # errors that may expose infrastructure information.
            return StorageHealth(
                configured_backend=self.configured_backend,
                active_backend=self.active_backend,
                healthy=False,
                bucket=self.bucket,
                detail=exc.__class__.__name__,
                checked_at_epoch=now,
            )

    def _local_object_path(self, object_key: str) -> Path:
        key = validate_object_key(object_key)
        candidate = (self.local_root / Path(*key.split("/"))).resolve()

        try:
            candidate.relative_to(self.local_root)
        except ValueError as exc:
            raise ValueError("object_key escapes local object storage root") from exc

        return candidate

    def _cache_path(self, object_key: str) -> Path:
        key = validate_object_key(object_key)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()

        suffix = Path(key).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix or ""):
            suffix = ""

        # Two-level sharding avoids very large flat directories.
        return self.cache_dir / digest[:2] / f"{digest}{suffix}"

    def _cache_lock(self, object_key: str) -> threading.Lock:
        cache_key = hashlib.sha256(
            validate_object_key(object_key).encode("utf-8")
        ).hexdigest()
        with self._cache_locks_guard:
            lock = self._cache_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._cache_locks[cache_key] = lock
            return lock

    def stat_object(self, object_key: str) -> ObjectStat:
        key = validate_object_key(object_key)

        if self.configured_backend == "filesystem":
            path = self._local_object_path(key)
            if not path.is_file():
                raise ObjectNotFoundError("Object not found")

            sidecar = path.with_name(path.name + ".sha256")
            expected_hash: Optional[str] = None
            try:
                candidate = sidecar.read_text(encoding="ascii").strip().lower()
                if re.fullmatch(r"[0-9a-f]{64}", candidate):
                    expected_hash = candidate
            except FileNotFoundError:
                pass

            return ObjectStat(
                bucket=self.bucket,
                object_key=key,
                size_bytes=path.stat().st_size,
                content_type=_clean_content_type(None, path),
                sha256=expected_hash,
                etag=None,
                version_id=None,
                backend="filesystem",
            )

        client = self._get_minio_client()
        try:
            result = client.stat_object(self.bucket, key)
        except Exception as exc:
            if _is_minio_not_found(exc):
                raise ObjectNotFoundError("Object not found") from exc
            raise ObjectStorageUnavailableError(
                "Unable to read object metadata from MinIO"
            ) from exc

        metadata = getattr(result, "metadata", None) or {}
        return ObjectStat(
            bucket=self.bucket,
            object_key=key,
            size_bytes=int(getattr(result, "size", 0) or 0),
            content_type=getattr(result, "content_type", None),
            sha256=_extract_sha256_from_metadata(metadata),
            etag=_strip_etag(getattr(result, "etag", None)),
            version_id=getattr(result, "version_id", None),
            backend="minio",
        )

    def object_exists(self, object_key: str) -> bool:
        try:
            self.stat_object(object_key)
            return True
        except ObjectNotFoundError:
            return False

    def put_file(
        self,
        local_path: str | Path,
        object_key: str,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expected_sha256: Optional[str] = None,
        verify_remote: bool = True,
    ) -> StoredObject:
        """
        Store an already validated local file.

        This is the preferred path for INTEL-I video uploads: FastAPI writes an
        upload to a temporary file, FFmpeg/OpenCV validates it, then this method
        persists that file to private object storage.
        """
        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Upload file not found: {path}")

        key = validate_object_key(object_key)
        size = path.stat().st_size
        digest = sha256_file(path)

        if expected_sha256:
            expected = expected_sha256.strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("expected_sha256 must be a 64-character hex digest")
            if digest != expected:
                raise ObjectStorageIntegrityError(
                    "Local upload SHA-256 does not match expected digest"
                )

        mime = _clean_content_type(content_type, path)
        safe_metadata = _safe_metadata(metadata)
        safe_metadata["intel-i-sha256"] = digest
        safe_metadata["sha256"] = digest

        if self.configured_backend == "filesystem":
            destination = self._local_object_path(key)
            destination.parent.mkdir(parents=True, exist_ok=True)

            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            sidecar = destination.with_name(destination.name + ".sha256")

            try:
                shutil.copyfile(path, temporary)
                copied_hash = sha256_file(temporary)
                if copied_hash != digest:
                    raise ObjectStorageIntegrityError(
                        "Filesystem copy SHA-256 verification failed"
                    )
                os.replace(temporary, destination)
                sidecar.write_text(digest + "\n", encoding="ascii")
            finally:
                temporary.unlink(missing_ok=True)

            return StoredObject(
                bucket=self.bucket,
                object_key=key,
                size_bytes=size,
                sha256=digest,
                content_type=mime,
                backend="filesystem",
            )

        client = self._get_minio_client()

        try:
            result = client.fput_object(
                self.bucket,
                key,
                str(path),
                content_type=mime,
                metadata=safe_metadata,
            )
        except Exception as exc:
            raise ObjectStorageUnavailableError(
                "Unable to persist uploaded video to MinIO"
            ) from exc

        stored = StoredObject(
            bucket=self.bucket,
            object_key=key,
            size_bytes=size,
            sha256=digest,
            content_type=mime,
            etag=_strip_etag(getattr(result, "etag", None)),
            version_id=getattr(result, "version_id", None),
            backend="minio",
        )

        if verify_remote:
            remote = self.stat_object(key)
            if remote.size_bytes != size:
                # Best-effort cleanup of a bad upload.
                try:
                    self.delete_object(key)
                except ObjectStorageError:
                    logger.exception(
                        "Failed to remove object after size verification failure"
                    )
                raise ObjectStorageIntegrityError(
                    "Remote object size does not match uploaded file"
                )

            if remote.sha256 and remote.sha256 != digest:
                try:
                    self.delete_object(key)
                except ObjectStorageError:
                    logger.exception(
                        "Failed to remove object after SHA-256 verification failure"
                    )
                raise ObjectStorageIntegrityError(
                    "Remote object SHA-256 metadata does not match uploaded file"
                )

        return stored

    def put_stream(
        self,
        stream: BinaryIO,
        object_key: str,
        *,
        length: int,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StoredObject:
        """
        Stream known-length bytes directly to MinIO.

        For untrusted uploaded video, put_file() is preferred because INTEL-I
        should validate the complete temporary video before durable storage.
        """
        if self.configured_backend != "minio":
            raise ObjectStorageConfigurationError(
                "put_stream is only available with the MinIO backend"
            )
        if length < 0:
            raise ValueError("length must be >= 0")

        key = validate_object_key(object_key)
        safe_metadata = _safe_metadata(metadata)
        client = self._get_minio_client()

        try:
            result = client.put_object(
                self.bucket,
                key,
                stream,
                length=length,
                content_type=content_type,
                metadata=safe_metadata,
            )
        except Exception as exc:
            raise ObjectStorageUnavailableError(
                "Unable to stream object to MinIO"
            ) from exc

        remote = self.stat_object(key)
        if remote.size_bytes != length:
            raise ObjectStorageIntegrityError(
                "Remote object size does not match streamed length"
            )

        return StoredObject(
            bucket=self.bucket,
            object_key=key,
            size_bytes=remote.size_bytes,
            sha256=remote.sha256 or "",
            content_type=remote.content_type or content_type,
            etag=_strip_etag(getattr(result, "etag", None)),
            version_id=getattr(result, "version_id", None),
            backend="minio",
        )

    def materialize_to_cache(
        self,
        object_key: str,
        *,
        expected_sha256: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Path:
        """
        Return a verified local path suitable for OpenCV/FFmpeg.

        The object is downloaded into an ephemeral cache using an atomic rename.
        A cached file is reused only after optional SHA-256 verification.
        """
        key = validate_object_key(object_key)
        cache_path = self._cache_path(key)
        lock = self._cache_lock(key)

        expected: Optional[str] = None
        if expected_sha256:
            expected = expected_sha256.strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("expected_sha256 must be a 64-character hex digest")

        with lock:
            if cache_path.is_file() and not force_refresh:
                if expected is None or sha256_file(cache_path) == expected:
                    return cache_path
                cache_path.unlink(missing_ok=True)

            remote_stat = self.stat_object(key)
            remote_sha = expected or remote_stat.sha256

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.part"
            )
            temporary.unlink(missing_ok=True)

            try:
                if self.configured_backend == "filesystem":
                    source = self._local_object_path(key)
                    if not source.is_file():
                        raise ObjectNotFoundError("Object not found")
                    shutil.copyfile(source, temporary)
                else:
                    client = self._get_minio_client()
                    try:
                        client.fget_object(self.bucket, key, str(temporary))
                    except Exception as exc:
                        if _is_minio_not_found(exc):
                            raise ObjectNotFoundError("Object not found") from exc
                        raise ObjectStorageUnavailableError(
                            "Unable to download object from MinIO"
                        ) from exc

                actual_size = temporary.stat().st_size
                if remote_stat.size_bytes >= 0 and actual_size != remote_stat.size_bytes:
                    raise ObjectStorageIntegrityError(
                        "Downloaded object size does not match object metadata"
                    )

                if remote_sha:
                    actual_sha = sha256_file(temporary)
                    if actual_sha != remote_sha:
                        raise ObjectStorageIntegrityError(
                            "Downloaded object failed SHA-256 verification"
                        )

                os.replace(temporary, cache_path)
                return cache_path
            finally:
                temporary.unlink(missing_ok=True)

    def download_to_path(
        self,
        object_key: str,
        destination: str | Path,
        *,
        expected_sha256: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Export a private object to a caller-selected path.

        Uses materialize_to_cache() first so integrity checks are identical to
        the OpenCV/FFmpeg processing path.
        """
        destination_path = Path(destination).expanduser().resolve()

        if destination_path.exists() and not overwrite:
            raise FileExistsError(str(destination_path))

        cached = self.materialize_to_cache(
            object_key,
            expected_sha256=expected_sha256,
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        temporary = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}.tmp"
        )
        try:
            shutil.copyfile(cached, temporary)
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)

        return destination_path

    def delete_object(
        self,
        object_key: str,
        *,
        delete_cache: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        key = validate_object_key(object_key)

        if self.configured_backend == "filesystem":
            path = self._local_object_path(key)
            sidecar = path.with_name(path.name + ".sha256")

            if not path.exists() and not ignore_missing:
                raise ObjectNotFoundError("Object not found")

            path.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)

            # Remove only empty directories below the configured root.
            parent = path.parent
            while parent != self.local_root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        else:
            client = self._get_minio_client()
            if not ignore_missing and not self.object_exists(key):
                raise ObjectNotFoundError("Object not found")

            try:
                client.remove_object(self.bucket, key)
            except Exception as exc:
                if ignore_missing and _is_minio_not_found(exc):
                    pass
                else:
                    raise ObjectStorageUnavailableError(
                        "Unable to delete object from MinIO"
                    ) from exc

        if delete_cache:
            self.invalidate_cache(key)

    def invalidate_cache(self, object_key: str) -> None:
        key = validate_object_key(object_key)
        cache_path = self._cache_path(key)
        with self._cache_lock(key):
            cache_path.unlink(missing_ok=True)

    def clear_cache(self, *, older_than_seconds: Optional[int] = None) -> int:
        """
        Delete cached object copies.

        When older_than_seconds is provided, only regular cache files with an
        mtime older than that threshold are removed.
        """
        if older_than_seconds is not None and older_than_seconds < 0:
            raise ValueError("older_than_seconds must be >= 0")

        removed = 0
        cutoff = (
            time.time() - older_than_seconds
            if older_than_seconds is not None
            else None
        )

        if not self.cache_dir.exists():
            return 0

        for path in self.cache_dir.rglob("*"):
            if not path.is_file():
                continue

            try:
                if cutoff is not None and path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning(
                    "Unable to remove object-cache file",
                    exc_info=True,
                )

        # Best-effort cleanup of now-empty cache directories.
        for directory in sorted(
            (p for p in self.cache_dir.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

        return removed


def _strip_etag(value: Any) -> Optional[str]:
    if value is None:
        return None
    result = str(value).strip().strip('"')
    return result or None


def _is_minio_not_found(exc: Exception) -> bool:
    """
    Recognize S3/MinIO not-found errors without importing MinIO at module load.
    """
    code = str(getattr(exc, "code", "") or "").strip()
    status = getattr(exc, "status", None)

    if code in {
        "NoSuchKey",
        "NoSuchObject",
        "NoSuchBucket",
        "NotFound",
        "XMinioInvalidObjectName",
    }:
        return code not in {"NoSuchBucket", "XMinioInvalidObjectName"}

    return status == 404


_storage_singleton: Optional[ObjectStorage] = None
_storage_singleton_lock = threading.Lock()


def get_object_storage(*, reset: bool = False) -> ObjectStorage:
    """
    Get the process-level INTEL-I object-storage service.

    reset=True is intended for tests/configuration reload before application
    startup, not normal request handling.
    """
    global _storage_singleton

    with _storage_singleton_lock:
        if reset:
            _storage_singleton = None

        if _storage_singleton is None:
            _storage_singleton = ObjectStorage()

        return _storage_singleton


def storage_readiness() -> dict[str, Any]:
    """
    Safe readiness payload suitable for authenticated system diagnostics.
    """
    storage = get_object_storage()
    health = storage.health()
    return {
        "configured_backend": health.configured_backend,
        "active_backend": health.active_backend,
        "healthy": health.healthy,
        "bucket": health.bucket,
        "detail": health.detail,
        "checked_at_epoch": health.checked_at_epoch,
    }


__all__ = [
    "ObjectNotFoundError",
    "ObjectStat",
    "ObjectStorage",
    "ObjectStorageConfigurationError",
    "ObjectStorageError",
    "ObjectStorageIntegrityError",
    "ObjectStorageUnavailableError",
    "StorageHealth",
    "StoredObject",
    "get_object_storage",
    "sha256_file",
    "storage_readiness",
    "validate_object_key",
]
