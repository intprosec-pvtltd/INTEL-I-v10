"""
INTEL-I production authentication security service.

Provides:
- TOTP MFA enrollment
- TOTP verification
- TOTP replay protection
- MFA brute-force lockout
- encrypted TOTP secret persistence
- one-time hashed MFA recovery codes
- recovery-code regeneration
- MFA disable flow
- self-service password change
- secure forgot-password token generation
- single-use password reset
- Super Admin password reset support
- password-history/reuse prevention
- access/refresh session revocation
- security audit logging

SECURITY GUARANTEES
-------------------
The service NEVER persists:
- plaintext passwords
- plaintext TOTP secrets
- plaintext OTP values
- plaintext recovery codes
- plaintext password-reset tokens

TOTP secrets are encrypted with Fernet before persistence.

Recovery codes are stored only as keyed HMAC-SHA256 digests.

Password-reset tokens are cryptographically random and stored only as
keyed HMAC-SHA256 digests.

All password mutations increment User.token_version. Existing INTEL-I
access tokens therefore fail immediately because auth.auth.get_current_user()
already validates the JWT "ver" claim against User.token_version.

All RefreshToken records are revoked on password changes/resets.

Rate limiting is enforced by the FastAPI/SlowAPI router layer. This service
adds persistent MFA failed-attempt lockout as defense in depth.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import desc
from sqlalchemy.orm import Session

from auth.auth import (
    hash_password,
    validate_password_strength,
    verify_password,
)
from db.auth_security_model import (
    MFARecoveryCode,
    PasswordHistory,
    PasswordResetToken,
    UserMFA,
    indian_time,
)
from db.model import (
    AuditLog,
    RefreshToken,
    User,
)


logger = logging.getLogger("auth-security")


# =========================================================================
# CONSTANTS / CONFIGURATION
# =========================================================================


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name, str(default)).strip()

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be an integer"
        ) from exc

    if value < minimum or value > maximum:
        raise RuntimeError(
            f"{name} must be between {minimum} and {maximum}"
        )

    return value


MFA_ISSUER = (
    os.getenv(
        "MFA_ISSUER",
        "INTEL-I",
    ).strip()
    or "INTEL-I"
)

MFA_TOTP_DIGITS = _env_int(
    "MFA_TOTP_DIGITS",
    6,
    minimum=6,
    maximum=8,
)

MFA_TOTP_PERIOD_SECONDS = _env_int(
    "MFA_TOTP_PERIOD_SECONDS",
    30,
    minimum=15,
    maximum=60,
)

MFA_TOTP_VALID_WINDOW = _env_int(
    "MFA_TOTP_VALID_WINDOW",
    1,
    minimum=0,
    maximum=2,
)

MFA_MAX_FAILED_ATTEMPTS = _env_int(
    "MFA_MAX_FAILED_ATTEMPTS",
    5,
    minimum=3,
    maximum=20,
)

MFA_LOCKOUT_MINUTES = _env_int(
    "MFA_LOCKOUT_MINUTES",
    15,
    minimum=1,
    maximum=1440,
)

MFA_RECOVERY_CODE_COUNT = _env_int(
    "MFA_RECOVERY_CODE_COUNT",
    10,
    minimum=5,
    maximum=20,
)

PASSWORD_RESET_EXPIRY_MINUTES = _env_int(
    "PASSWORD_RESET_EXPIRY_MINUTES",
    20,
    minimum=5,
    maximum=120,
)

PASSWORD_HISTORY_COUNT = _env_int(
    "PASSWORD_HISTORY_COUNT",
    5,
    minimum=1,
    maximum=24,
)

MAX_PASSWORD_UTF8_BYTES = 72


# =========================================================================
# RESULT TYPES
# =========================================================================


@dataclass(frozen=True)
class MFAEnrollmentResult:
    """
    Returned exactly once when MFA enrollment begins.

    secret and provisioning_uri are sensitive transient values.
    They must never be logged or persisted as plaintext.
    """

    secret: str
    provisioning_uri: str


@dataclass(frozen=True)
class MFAActivationResult:
    """
    Recovery codes are returned only at generation time.
    """

    recovery_codes: tuple[str, ...]


@dataclass(frozen=True)
class PasswordResetGrant:
    """
    Internal password-reset delivery payload.

    IMPORTANT:
    The token must be delivered through the approved password-reset delivery
    mechanism. A public production API must NOT simply echo this token back
    to an unauthenticated caller.
    """

    user_id: int
    email: str
    token: str
    expires_at: datetime


@dataclass(frozen=True)
class PasswordMutationResult:
    user_id: int
    refresh_tokens_revoked: int


# =========================================================================
# SECURITY CONFIGURATION
# =========================================================================


def _required_secret(
    env_name: str,
    *,
    minimum_length: int = 32,
) -> str:
    value = os.getenv(
        env_name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(
            f"{env_name} is required"
        )

    if len(value.encode("utf-8")) < minimum_length:
        raise RuntimeError(
            f"{env_name} must contain at least "
            f"{minimum_length} bytes"
        )

    return value


def _mfa_fernet() -> Fernet:
    """
    Load and validate the dedicated MFA encryption key.

    Never silently fall back to another application encryption key. MFA
    secrets should have an independent key so the credential domain can be
    rotated independently.
    """

    raw = os.getenv(
        "MFA_TOTP_ENCRYPTION_KEY",
        "",
    ).strip()

    if not raw:
        raise RuntimeError(
            "MFA_TOTP_ENCRYPTION_KEY is required"
        )

    try:
        return Fernet(
            raw.encode("ascii")
        )
    except (
        ValueError,
        TypeError,
        UnicodeEncodeError,
    ) as exc:
        raise RuntimeError(
            "MFA_TOTP_ENCRYPTION_KEY must be a valid Fernet key"
        ) from exc


def validate_security_configuration() -> None:
    """
    Fail explicitly when the security keys required by the MFA/password-reset
    subsystem are invalid.

    Call this during production startup after environment variables have been
    loaded.
    """

    _mfa_fernet()

    _required_secret(
        "MFA_RECOVERY_CODE_PEPPER",
    )

    _required_secret(
        "PASSWORD_RESET_TOKEN_PEPPER",
    )

    _required_secret(
        "AUTH_FINGERPRINT_PEPPER",
    )


# =========================================================================
# BASIC HELPERS
# =========================================================================


def _normalize_email(
    email: str,
) -> str:
    return str(
        email or ""
    ).strip().lower()


def _normalize_source_ip(
    source_ip: str | None,
) -> str | None:
    value = str(
        source_ip or ""
    ).strip()

    if not value:
        return None

    return value[:64]


def _safe_user_id(
    user: User,
) -> int:
    if user is None or user.id is None:
        raise ValueError(
            "A persisted user is required"
        )

    return int(user.id)


def _constant_equals(
    left: str,
    right: str,
) -> bool:
    try:
        return hmac.compare_digest(
            str(left),
            str(right),
        )
    except Exception:
        return False


# =========================================================================
# SECURITY AUDIT
# =========================================================================


_FORBIDDEN_AUDIT_KEYS = {
    "password",
    "new_password",
    "current_password",
    "secret",
    "totp_secret",
    "otp",
    "totp",
    "code",
    "recovery_code",
    "recovery_codes",
    "token",
    "reset_token",
    "access_token",
    "refresh_token",
}


def _sanitize_audit_details(
    value: Any,
) -> Any:
    """
    Defense in depth against accidentally inserting credentials into
    audit_logs.details.
    """

    if isinstance(
        value,
        dict,
    ):
        result: dict[str, Any] = {}

        for key, item in value.items():
            normalized_key = str(
                key
            ).strip().lower()

            if normalized_key in _FORBIDDEN_AUDIT_KEYS:
                continue

            result[str(key)] = _sanitize_audit_details(
                item
            )

        return result

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _sanitize_audit_details(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

    return str(value)


def _audit(
    db: Session,
    *,
    action: str,
    resource_user_id: int | None,
    actor_user_id: int | None = None,
    details: dict[str, Any] | None = None,
    source_ip: str | None = None,
) -> None:
    """
    Add a security event to the existing INTEL-I AuditLog.

    Caller controls the surrounding database transaction.
    """

    db.add(
        AuditLog(
            user_id=actor_user_id,
            action=str(action)[:100],
            resource_type="user",
            resource_id=(
                str(resource_user_id)
                if resource_user_id is not None
                else None
            ),
            details=_sanitize_audit_details(
                details or {}
            ),
            source_ip=_normalize_source_ip(
                source_ip
            ),
        )
    )


# =========================================================================
# REQUEST FINGERPRINT
# =========================================================================


def build_request_fingerprint(
    *,
    source_ip: str | None,
    user_agent: str | None,
) -> str:
    """
    Produce a privacy-preserving HMAC fingerprint for a password-reset
    request.

    PasswordResetToken.request_fingerprint therefore does not need to contain
    the raw IP/User-Agent combination.
    """

    pepper = _required_secret(
        "AUTH_FINGERPRINT_PEPPER",
    ).encode(
        "utf-8"
    )

    material = (
        f"{str(source_ip or '').strip()}|"
        f"{str(user_agent or '').strip()[:512]}"
    ).encode(
        "utf-8"
    )

    return hmac.new(
        pepper,
        material,
        hashlib.sha256,
    ).hexdigest()


# =========================================================================
# TOTP SECRET ENCRYPTION
# =========================================================================


def _encrypt_totp_secret(
    secret: str,
) -> str:
    value = str(
        secret or ""
    ).strip()

    if not value:
        raise ValueError(
            "TOTP secret cannot be empty"
        )

    token = _mfa_fernet().encrypt(
        value.encode(
            "ascii"
        )
    )

    return token.decode(
        "ascii"
    )


def _decrypt_totp_secret(
    encrypted_secret: str,
) -> str:
    value = str(
        encrypted_secret or ""
    ).strip()

    if not value:
        raise RuntimeError(
            "Encrypted MFA secret is missing"
        )

    try:
        decrypted = _mfa_fernet().decrypt(
            value.encode(
                "ascii"
            )
        )
    except (
        InvalidToken,
        ValueError,
        TypeError,
    ) as exc:
        logger.error(
            "Unable to decrypt MFA secret"
        )

        raise RuntimeError(
            "Unable to decrypt MFA secret"
        ) from exc

    try:
        return decrypted.decode(
            "ascii"
        )
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Decrypted MFA secret is invalid"
        ) from exc


# =========================================================================
# RFC 6238 TOTP
# =========================================================================


def _generate_totp_secret() -> str:
    """
    Generate a 160-bit RFC-compatible Base32 secret.
    """

    raw = secrets.token_bytes(
        20
    )

    return (
        base64.b32encode(
            raw
        )
        .decode("ascii")
        .rstrip("=")
    )


def _decode_base32_secret(
    secret: str,
) -> bytes:
    normalized = (
        str(secret or "")
        .strip()
        .replace(" ", "")
        .upper()
    )

    if not normalized:
        raise ValueError(
            "TOTP secret is empty"
        )

    padding = (
        8 - len(normalized) % 8
    ) % 8

    normalized += "=" * padding

    try:
        return base64.b32decode(
            normalized,
            casefold=True,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid TOTP secret"
        ) from exc


def _totp_digest(
    algorithm: str,
):
    normalized = str(
        algorithm or "SHA1"
    ).strip().upper()

    if normalized == "SHA1":
        return hashlib.sha1

    if normalized == "SHA256":
        return hashlib.sha256

    if normalized == "SHA512":
        return hashlib.sha512

    raise ValueError(
        "Unsupported TOTP algorithm"
    )


def _totp_code_for_step(
    secret: str,
    *,
    step: int,
    digits: int,
    algorithm: str,
) -> str:
    if step < 0:
        raise ValueError(
            "TOTP step cannot be negative"
        )

    key = _decode_base32_secret(
        secret
    )

    counter = struct.pack(
        ">Q",
        int(step),
    )

    digest = hmac.new(
        key,
        counter,
        _totp_digest(
            algorithm
        ),
    ).digest()

    offset = (
        digest[-1] & 0x0F
    )

    binary = (
        (
            digest[offset]
            & 0x7F
        )
        << 24
        | digest[offset + 1] << 16
        | digest[offset + 2] << 8
        | digest[offset + 3]
    )

    modulo = 10 ** int(
        digits
    )

    return str(
        binary % modulo
    ).zfill(
        digits
    )


def _normalize_totp_code(
    code: str,
    *,
    digits: int,
) -> str | None:
    normalized = (
        str(code or "")
        .replace(" ", "")
        .replace("-", "")
        .strip()
    )

    if (
        len(normalized) != int(digits)
        or not normalized.isdigit()
    ):
        return None

    return normalized


def _find_matching_totp_step(
    secret: str,
    submitted_code: str,
    *,
    digits: int,
    period_seconds: int,
    algorithm: str,
    valid_window: int = MFA_TOTP_VALID_WINDOW,
    unix_time: int | None = None,
) -> int | None:
    normalized_code = _normalize_totp_code(
        submitted_code,
        digits=digits,
    )

    if normalized_code is None:
        return None

    timestamp = (
        int(time.time())
        if unix_time is None
        else int(unix_time)
    )

    current_step = (
        timestamp
        // int(period_seconds)
    )

    # Check current step first, then neighbouring clock-skew windows.
    offsets = [0]

    for distance in range(
        1,
        int(valid_window) + 1,
    ):
        offsets.extend(
            [
                -distance,
                distance,
            ]
        )

    for offset in offsets:
        candidate_step = (
            current_step + offset
        )

        if candidate_step < 0:
            continue

        expected = _totp_code_for_step(
            secret,
            step=candidate_step,
            digits=digits,
            algorithm=algorithm,
        )

        if _constant_equals(
            normalized_code,
            expected,
        ):
            return candidate_step

    return None


def _provisioning_uri(
    *,
    email: str,
    secret: str,
) -> str:
    """
    Build the standard otpauth URI accepted by:
    - Google Authenticator
    - Microsoft Authenticator
    - Authy
    - 1Password
    - compatible TOTP applications
    """

    label = quote(
        f"{MFA_ISSUER}:{email}",
        safe="",
    )

    issuer = quote(
        MFA_ISSUER,
        safe="",
    )

    return (
        f"otpauth://totp/{label}"
        f"?secret={quote(secret, safe='')}"
        f"&issuer={issuer}"
        f"&algorithm=SHA1"
        f"&digits={MFA_TOTP_DIGITS}"
        f"&period={MFA_TOTP_PERIOD_SECONDS}"
    )


# =========================================================================
# MFA RECORD
# =========================================================================


def get_mfa_status(
    db: Session,
    user_id: int,
) -> dict[str, Any]:
    """
    Safe MFA status response.

    No secret, ciphertext or recovery-code hash is exposed.
    """

    record = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == int(user_id)
        )
        .first()
    )

    if record is None:
        return {
            "enabled": False,
            "enrollment_pending": False,
            "locked": False,
            "locked_until": None,
            "enabled_at": None,
            "last_verified_at": None,
            "recovery_codes_remaining": 0,
        }

    now = indian_time()

    remaining = (
        db.query(MFARecoveryCode)
        .filter(
            MFARecoveryCode.user_id
            == int(user_id),
            MFARecoveryCode.used_at.is_(
                None
            ),
        )
        .count()
    )

    return {
        "enabled": bool(
            record.enabled
        ),
        "enrollment_pending": bool(
            record.pending_secret_encrypted
        ),
        "locked": bool(
            record.locked_until
            and record.locked_until > now
        ),
        "locked_until": (
            record.locked_until
        ),
        "enabled_at": (
            record.enabled_at
        ),
        "last_verified_at": (
            record.last_verified_at
        ),
        "recovery_codes_remaining": int(
            remaining
        ),
    }


# =========================================================================
# MFA LOCKOUT
# =========================================================================


def _ensure_mfa_not_locked(
    mfa: UserMFA,
) -> None:
    now = indian_time()

    if (
        mfa.locked_until is not None
        and mfa.locked_until > now
    ):
        raise PermissionError(
            "MFA verification temporarily locked"
        )

    # Expired lock can be cleared automatically.
    if (
        mfa.locked_until is not None
        and mfa.locked_until <= now
    ):
        mfa.locked_until = None
        mfa.failed_attempts = 0


def _record_mfa_failure(
    mfa: UserMFA,
) -> None:
    now = indian_time()

    mfa.failed_attempts = int(
        mfa.failed_attempts or 0
    ) + 1

    mfa.last_failed_at = now

    if (
        mfa.failed_attempts
        >= MFA_MAX_FAILED_ATTEMPTS
    ):
        mfa.locked_until = (
            now
            + timedelta(
                minutes=MFA_LOCKOUT_MINUTES
            )
        )

        # Reset the counter for the next lockout window.
        mfa.failed_attempts = 0


def _record_mfa_success(
    mfa: UserMFA,
) -> None:
    mfa.failed_attempts = 0
    mfa.locked_until = None
    mfa.last_verified_at = indian_time()


# =========================================================================
# MFA ENROLLMENT
# =========================================================================


def begin_totp_enrollment(
    db: Session,
    *,
    user: User,
    source_ip: str | None = None,
) -> MFAEnrollmentResult:
    """
    Start/restart MFA enrollment.

    The raw secret is returned only to the authenticated caller. Only its
    encrypted representation is persisted.

    This function commits its own security transaction.
    """

    user_id = _safe_user_id(
        user
    )

    mfa = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == user_id
        )
        .with_for_update()
        .first()
    )

    if (
        mfa is not None
        and mfa.enabled
    ):
        raise ValueError(
            "MFA is already enabled"
        )

    secret = _generate_totp_secret()

    encrypted_secret = (
        _encrypt_totp_secret(
            secret
        )
    )

    now = indian_time()

    if mfa is None:
        mfa = UserMFA(
            user_id=user_id,
            enabled=False,
            pending_secret_encrypted=(
                encrypted_secret
            ),
            encryption_key_version=1,
            totp_algorithm="SHA1",
            totp_digits=MFA_TOTP_DIGITS,
            totp_period_seconds=(
                MFA_TOTP_PERIOD_SECONDS
            ),
            enrollment_started_at=now,
            failed_attempts=0,
        )

        db.add(
            mfa
        )

    else:
        mfa.enabled = False
        mfa.pending_secret_encrypted = (
            encrypted_secret
        )
        mfa.totp_secret_encrypted = None
        mfa.totp_algorithm = "SHA1"
        mfa.totp_digits = MFA_TOTP_DIGITS
        mfa.totp_period_seconds = (
            MFA_TOTP_PERIOD_SECONDS
        )
        mfa.enrollment_started_at = now
        mfa.last_accepted_totp_step = None
        mfa.failed_attempts = 0
        mfa.locked_until = None

    _audit(
        db,
        action="auth.mfa.enrollment.started",
        resource_user_id=user_id,
        actor_user_id=user_id,
        details={
            "method": "totp",
        },
        source_ip=source_ip,
    )

    try:
        db.commit()

    except Exception:
        db.rollback()
        raise

    return MFAEnrollmentResult(
        secret=secret,
        provisioning_uri=_provisioning_uri(
            email=_normalize_email(
                user.email
            ),
            secret=secret,
        ),
    )


def cancel_totp_enrollment(
    db: Session,
    *,
    user: User,
    source_ip: str | None = None,
) -> None:
    """
    Cancel a pending enrollment without changing an already-enabled MFA
    configuration.
    """

    user_id = _safe_user_id(
        user
    )

    mfa = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == user_id
        )
        .with_for_update()
        .first()
    )

    if mfa is None:
        return

    if mfa.enabled:
        raise ValueError(
            "Enabled MFA cannot be cancelled through enrollment cancellation"
        )

    mfa.pending_secret_encrypted = None
    mfa.enrollment_started_at = None
    mfa.failed_attempts = 0
    mfa.locked_until = None

    _audit(
        db,
        action="auth.mfa.enrollment.cancelled",
        resource_user_id=user_id,
        actor_user_id=user_id,
        source_ip=source_ip,
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


# =========================================================================
# RECOVERY CODES
# =========================================================================


def _normalize_recovery_code(
    code: str,
) -> str:
    return (
        str(code or "")
        .upper()
        .replace("-", "")
        .replace(" ", "")
        .strip()
    )


def _hash_recovery_code(
    code: str,
) -> str:
    normalized = _normalize_recovery_code(
        code
    )

    if not normalized:
        raise ValueError(
            "Recovery code cannot be empty"
        )

    pepper = _required_secret(
        "MFA_RECOVERY_CODE_PEPPER",
    ).encode(
        "utf-8"
    )

    return hmac.new(
        pepper,
        normalized.encode(
            "ascii"
        ),
        hashlib.sha256,
    ).hexdigest()


def _generate_recovery_code() -> str:
    """
    80 bits of random recovery-code entropy, formatted for humans.

    Example shape:
        A12BC-34DEF-56GHJ-78KLM

    Ambiguous formatting has no security meaning because the normalized
    value is hashed before persistence.
    """

    # Crockford-like alphabet with ambiguous characters removed.
    alphabet = (
        "23456789"
        "ABCDEFGHJKLMNPQRSTUVWXYZ"
    )

    raw = "".join(
        secrets.choice(
            alphabet
        )
        for _ in range(20)
    )

    return "-".join(
        raw[index:index + 5]
        for index in range(
            0,
            len(raw),
            5,
        )
    )


def _replace_recovery_codes(
    db: Session,
    *,
    user_id: int,
) -> tuple[str, ...]:
    db.query(
        MFARecoveryCode
    ).filter(
        MFARecoveryCode.user_id
        == int(user_id)
    ).delete(
        synchronize_session=False
    )

    plaintext_codes: list[str] = []

    hashes: set[str] = set()

    while (
        len(plaintext_codes)
        < MFA_RECOVERY_CODE_COUNT
    ):
        code = _generate_recovery_code()

        code_hash = (
            _hash_recovery_code(
                code
            )
        )

        if code_hash in hashes:
            continue

        hashes.add(
            code_hash
        )

        plaintext_codes.append(
            code
        )

        db.add(
            MFARecoveryCode(
                user_id=int(user_id),
                code_hash=code_hash,
            )
        )

    return tuple(
        plaintext_codes
    )


# =========================================================================
# CONFIRM MFA ENROLLMENT
# =========================================================================


def confirm_totp_enrollment(
    db: Session,
    *,
    user: User,
    totp_code: str,
    source_ip: str | None = None,
) -> MFAActivationResult:
    """
    Validate the first TOTP and permanently activate MFA.

    Recovery codes are generated here and returned exactly once.
    """

    user_id = _safe_user_id(
        user
    )

    mfa = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == user_id
        )
        .with_for_update()
        .first()
    )

    if (
        mfa is None
        or not mfa.pending_secret_encrypted
    ):
        raise ValueError(
            "No MFA enrollment is pending"
        )

    if mfa.enabled:
        raise ValueError(
            "MFA is already enabled"
        )

    _ensure_mfa_not_locked(
        mfa
    )

    secret = _decrypt_totp_secret(
        mfa.pending_secret_encrypted
    )

    matching_step = (
        _find_matching_totp_step(
            secret,
            totp_code,
            digits=int(
                mfa.totp_digits
                or MFA_TOTP_DIGITS
            ),
            period_seconds=int(
                mfa.totp_period_seconds
                or MFA_TOTP_PERIOD_SECONDS
            ),
            algorithm=(
                mfa.totp_algorithm
                or "SHA1"
            ),
        )
    )

    if matching_step is None:
        _record_mfa_failure(
            mfa
        )

        _audit(
            db,
            action="auth.mfa.enrollment.verify_failed",
            resource_user_id=user_id,
            actor_user_id=user_id,
            source_ip=source_ip,
        )

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise PermissionError(
            "Invalid MFA code"
        )

    now = indian_time()

    mfa.totp_secret_encrypted = (
        mfa.pending_secret_encrypted
    )

    mfa.pending_secret_encrypted = None
    mfa.enabled = True
    mfa.enabled_at = now
    mfa.disabled_at = None
    mfa.last_accepted_totp_step = (
        matching_step
    )

    _record_mfa_success(
        mfa
    )

    recovery_codes = (
        _replace_recovery_codes(
            db,
            user_id=user_id,
        )
    )

    mfa.recovery_codes_generated_at = now

    _audit(
        db,
        action="auth.mfa.enabled",
        resource_user_id=user_id,
        actor_user_id=user_id,
        details={
            "method": "totp",
            "recovery_code_count": len(
                recovery_codes
            ),
        },
        source_ip=source_ip,
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    return MFAActivationResult(
        recovery_codes=recovery_codes
    )


# =========================================================================
# TOTP VERIFICATION
# =========================================================================


def _verify_totp_locked(
    *,
    mfa: UserMFA,
    submitted_code: str,
) -> bool:
    """
    Verify a TOTP while the caller holds a FOR UPDATE lock on UserMFA.

    This makes last_accepted_totp_step a true replay-protection barrier across
    concurrent requests.
    """

    if (
        not mfa.enabled
        or not mfa.totp_secret_encrypted
    ):
        return False

    _ensure_mfa_not_locked(
        mfa
    )

    secret = _decrypt_totp_secret(
        mfa.totp_secret_encrypted
    )

    matching_step = (
        _find_matching_totp_step(
            secret,
            submitted_code,
            digits=int(
                mfa.totp_digits
                or MFA_TOTP_DIGITS
            ),
            period_seconds=int(
                mfa.totp_period_seconds
                or MFA_TOTP_PERIOD_SECONDS
            ),
            algorithm=(
                mfa.totp_algorithm
                or "SHA1"
            ),
        )
    )

    if matching_step is None:
        return False

    previous_step = (
        mfa.last_accepted_totp_step
    )

    if (
        previous_step is not None
        and int(matching_step)
        <= int(previous_step)
    ):
        # Reusing a previously consumed TOTP is not allowed.
        return False

    mfa.last_accepted_totp_step = (
        int(matching_step)
    )

    return True


# =========================================================================
# RECOVERY-CODE VERIFICATION
# =========================================================================


def _verify_recovery_code_locked(
    db: Session,
    *,
    user_id: int,
    submitted_code: str,
) -> bool:
    """
    Consume a recovery code exactly once.

    Rows are locked to prevent two concurrent requests consuming the same
    recovery credential.
    """

    normalized = _normalize_recovery_code(
        submitted_code
    )

    if not normalized:
        return False

    submitted_hash = (
        _hash_recovery_code(
            normalized
        )
    )

    rows = (
        db.query(MFARecoveryCode)
        .filter(
            MFARecoveryCode.user_id
            == int(user_id),
            MFARecoveryCode.used_at.is_(
                None
            ),
        )
        .with_for_update()
        .all()
    )

    matched: MFARecoveryCode | None = None

    # Deliberately compare against the available records rather than using
    # raw recovery-code material in a SQL expression.
    for row in rows:
        if hmac.compare_digest(
            row.code_hash,
            submitted_hash,
        ):
            matched = row

    if matched is None:
        return False

    matched.used_at = indian_time()

    return True


# =========================================================================
# GENERAL MFA VERIFICATION
# =========================================================================


def verify_mfa(
    db: Session,
    *,
    user_id: int,
    code: str,
    allow_recovery_code: bool = True,
    source_ip: str | None = None,
    audit_success: bool = True,
) -> str:
    """
    Verify an enabled user's second factor.

    Returns:
        "totp"
        "recovery"

    Raises:
        ValueError       MFA not configured
        PermissionError  invalid/locked MFA

    This function commits verification/replay/lockout state.
    """

    user_id = int(
        user_id
    )

    mfa = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == user_id
        )
        .with_for_update()
        .first()
    )

    if (
        mfa is None
        or not mfa.enabled
    ):
        raise ValueError(
            "MFA is not enabled"
        )

    try:
        _ensure_mfa_not_locked(
            mfa
        )
    except PermissionError:
        _audit(
            db,
            action="auth.mfa.locked_attempt",
            resource_user_id=user_id,
            actor_user_id=user_id,
            source_ip=source_ip,
        )

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise

    method: str | None = None

    if _verify_totp_locked(
        mfa=mfa,
        submitted_code=code,
    ):
        method = "totp"

    elif (
        allow_recovery_code
        and _verify_recovery_code_locked(
            db,
            user_id=user_id,
            submitted_code=code,
        )
    ):
        method = "recovery"

    if method is None:
        _record_mfa_failure(
            mfa
        )

        _audit(
            db,
            action="auth.mfa.verify_failed",
            resource_user_id=user_id,
            actor_user_id=user_id,
            source_ip=source_ip,
        )

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise PermissionError(
            "Invalid MFA code"
        )

    _record_mfa_success(
        mfa
    )

    if audit_success:
        _audit(
            db,
            action="auth.mfa.verified",
            resource_user_id=user_id,
            actor_user_id=user_id,
            details={
                "method": method,
            },
            source_ip=source_ip,
        )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    return method


# =========================================================================
# REGENERATE MFA RECOVERY CODES
# =========================================================================


def regenerate_recovery_codes(
    db: Session,
    *,
    user: User,
    current_password: str,
    mfa_code: str,
    source_ip: str | None = None,
) -> MFAActivationResult:
    """
    High-risk operation requiring both password and an existing MFA factor.

    Old recovery codes become invalid immediately.
    """

    user_id = _safe_user_id(
        user
    )

    if not verify_password(
        current_password,
        user.password,
    ):
        _audit(
            db,
            action="auth.mfa.recovery_regeneration.password_failed",
            resource_user_id=user_id,
            actor_user_id=user_id,
            source_ip=source_ip,
        )

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise PermissionError(
            "Current password is incorrect"
        )

    # This commits verification state before regeneration.
    verify_mfa(
        db,
        user_id=user_id,
        code=mfa_code,
        allow_recovery_code=False,
        source_ip=source_ip,
        audit_success=False,
    )

    mfa = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == user_id
        )
        .with_for_update()
        .first()
    )

    if (
        mfa is None
        or not mfa.enabled
    ):
        raise ValueError(
            "MFA is not enabled"
        )

    codes = _replace_recovery_codes(
        db,
        user_id=user_id,
    )

    mfa.recovery_codes_generated_at = (
        indian_time()
    )

    _audit(
        db,
        action="auth.mfa.recovery_codes.regenerated",
        resource_user_id=user_id,
        actor_user_id=user_id,
        details={
            "recovery_code_count": len(
                codes
            ),
        },
        source_ip=source_ip,
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    return MFAActivationResult(
        recovery_codes=codes
    )


# =========================================================================
# MFA DISABLE
# =========================================================================


def disable_mfa(
    db: Session,
    *,
    user: User,
    current_password: str,
    mfa_code: str,
    source_ip: str | None = None,
) -> None:
    """
    Disable MFA only after password + second-factor verification.

    A TOTP or unused recovery code may be used as the second factor.
    """

    user_id = _safe_user_id(
        user
    )

    if not verify_password(
        current_password,
        user.password,
    ):
        _audit(
            db,
            action="auth.mfa.disable.password_failed",
            resource_user_id=user_id,
            actor_user_id=user_id,
            source_ip=source_ip,
        )

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise PermissionError(
            "Current password is incorrect"
        )

    verify_mfa(
        db,
        user_id=user_id,
        code=mfa_code,
        allow_recovery_code=True,
        source_ip=source_ip,
        audit_success=False,
    )

    mfa = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == user_id
        )
        .with_for_update()
        .first()
    )

    if (
        mfa is None
        or not mfa.enabled
    ):
        raise ValueError(
            "MFA is not enabled"
        )

    now = indian_time()

    # Delete all MFA credentials; do not keep a dormant reusable secret.
    mfa.enabled = False
    mfa.totp_secret_encrypted = None
    mfa.pending_secret_encrypted = None
    mfa.last_accepted_totp_step = None
    mfa.failed_attempts = 0
    mfa.locked_until = None
    mfa.disabled_at = now

    db.query(
        MFARecoveryCode
    ).filter(
        MFARecoveryCode.user_id
        == user_id
    ).delete(
        synchronize_session=False
    )

    _audit(
        db,
        action="auth.mfa.disabled",
        resource_user_id=user_id,
        actor_user_id=user_id,
        source_ip=source_ip,
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


# =========================================================================
# PASSWORD POLICY / HISTORY
# =========================================================================


def _validate_password_for_user(
    *,
    user: User,
    password: str,
) -> None:
    """
    Reuse existing INTEL-I password policy and add protections needed for the
    application's existing bcrypt implementation.
    """

    if not isinstance(
        password,
        str,
    ):
        raise ValueError(
            "Password is required"
        )

    if len(password) > 128:
        raise ValueError(
            "Password must not exceed 128 characters"
        )

    # bcrypt uses at most the first 72 bytes. Reject longer values rather than
    # silently treating different passwords as equivalent.
    if (
        len(
            password.encode(
                "utf-8"
            )
        )
        > MAX_PASSWORD_UTF8_BYTES
    ):
        raise ValueError(
            "Password must not exceed 72 UTF-8 bytes"
        )

    validate_password_strength(
        password
    )

    lowered = password.lower()

    email = _normalize_email(
        user.email
    )

    local_part = (
        email.split(
            "@",
            1,
        )[0]
        if "@" in email
        else email
    )

    if (
        local_part
        and len(local_part) >= 4
        and local_part.lower() in lowered
    ):
        raise ValueError(
            "Password must not contain your email username"
        )

    full_name = str(
        user.full_name or ""
    ).strip()

    for part in full_name.split():
        normalized_part = (
            part.strip().lower()
        )

        if (
            len(normalized_part) >= 4
            and normalized_part in lowered
        ):
            raise ValueError(
                "Password must not contain your name"
            )


def _assert_password_not_reused(
    db: Session,
    *,
    user: User,
    new_password: str,
) -> None:
    """
    Reject the user's current password and recent historical passwords.
    """

    if verify_password(
        new_password,
        user.password,
    ):
        raise ValueError(
            "New password must be different from the current password"
        )

    history = (
        db.query(PasswordHistory)
        .filter(
            PasswordHistory.user_id
            == int(user.id)
        )
        .order_by(
            desc(
                PasswordHistory.created_at
            ),
            desc(
                PasswordHistory.id
            ),
        )
        .limit(
            PASSWORD_HISTORY_COUNT
        )
        .all()
    )

    for item in history:
        if verify_password(
            new_password,
            item.password_hash,
        ):
            raise ValueError(
                f"The last {PASSWORD_HISTORY_COUNT} passwords cannot be reused"
            )


def _save_current_password_to_history(
    db: Session,
    *,
    user: User,
    reason: str,
) -> None:
    if not user.password:
        return

    db.add(
        PasswordHistory(
            user_id=int(
                user.id
            ),
            password_hash=user.password,
            reason=str(
                reason
            )[:32],
        )
    )


def _prune_password_history(
    db: Session,
    *,
    user_id: int,
) -> None:
    """
    Keep a small bounded history rather than indefinitely retaining old
    credential hashes.
    """

    rows = (
        db.query(
            PasswordHistory.id
        )
        .filter(
            PasswordHistory.user_id
            == int(user_id)
        )
        .order_by(
            desc(
                PasswordHistory.created_at
            ),
            desc(
                PasswordHistory.id
            ),
        )
        .offset(
            PASSWORD_HISTORY_COUNT
        )
        .all()
    )

    stale_ids = [
        int(row[0])
        for row in rows
    ]

    if stale_ids:
        (
            db.query(
                PasswordHistory
            )
            .filter(
                PasswordHistory.id.in_(
                    stale_ids
                )
            )
            .delete(
                synchronize_session=False
            )
        )


# =========================================================================
# SESSION REVOCATION
# =========================================================================


def revoke_user_sessions(
    db: Session,
    *,
    user: User,
) -> int:
    """
    Revoke every existing user session.

    ACCESS TOKENS:
        invalidated through token_version.

    REFRESH TOKENS:
        explicitly marked revoked in PostgreSQL.
    """

    user.token_version = int(
        user.token_version or 0
    ) + 1

    revoked = (
        db.query(RefreshToken)
        .filter(
            RefreshToken.user_id
            == int(user.id),
            RefreshToken.revoked.is_(
                False
            ),
        )
        .update(
            {
                "revoked": True,
            },
            synchronize_session=False,
        )
    )

    return int(
        revoked or 0
    )


# =========================================================================
# SELF-SERVICE PASSWORD CHANGE
# =========================================================================


def change_password(
    db: Session,
    *,
    user: User,
    current_password: str,
    new_password: str,
    mfa_code: str | None = None,
    source_ip: str | None = None,
) -> PasswordMutationResult:
    """
    Change an authenticated user's password.

    If MFA is enabled, MFA verification is mandatory.

    Every existing session is revoked after the password changes.
    """

    user_id = _safe_user_id(
        user
    )

    if not verify_password(
        current_password,
        user.password,
    ):
        _audit(
            db,
            action="auth.password.change_failed",
            resource_user_id=user_id,
            actor_user_id=user_id,
            details={
                "reason": "invalid_current_password",
            },
            source_ip=source_ip,
        )

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise PermissionError(
            "Current password is incorrect"
        )

    mfa = (
        db.query(UserMFA)
        .filter(
            UserMFA.user_id
            == user_id,
            UserMFA.enabled.is_(
                True
            ),
        )
        .first()
    )

    if mfa is not None:
        if not str(
            mfa_code or ""
        ).strip():
            raise PermissionError(
                "MFA code is required"
            )

        verify_mfa(
            db,
            user_id=user_id,
            code=str(mfa_code),
            allow_recovery_code=True,
            source_ip=source_ip,
            audit_success=False,
        )

        # Refresh ORM state because verify_mfa commits its transaction.
        db.refresh(
            user
        )

    _validate_password_for_user(
        user=user,
        password=new_password,
    )

    _assert_password_not_reused(
        db,
        user=user,
        new_password=new_password,
    )

    _save_current_password_to_history(
        db,
        user=user,
        reason="change",
    )

    user.password = hash_password(
        new_password
    )

    revoked = revoke_user_sessions(
        db,
        user=user,
    )

    _audit(
        db,
        action="auth.password.changed",
        resource_user_id=user_id,
        actor_user_id=user_id,
        details={
            "sessions_revoked": revoked,
        },
        source_ip=source_ip,
    )

    try:
        db.flush()

        _prune_password_history(
            db,
            user_id=user_id,
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    return PasswordMutationResult(
        user_id=user_id,
        refresh_tokens_revoked=revoked,
    )


# =========================================================================
# PASSWORD RESET TOKEN
# =========================================================================


def _generate_reset_token() -> str:
    """
    Generate >=256 bits of password-reset entropy.
    """

    return secrets.token_urlsafe(
        32
    )


def _hash_reset_token(
    token: str,
) -> str:
    normalized = str(
        token or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "Reset token is required"
        )

    pepper = _required_secret(
        "PASSWORD_RESET_TOKEN_PEPPER",
    ).encode(
        "utf-8"
    )

    return hmac.new(
        pepper,
        normalized.encode(
            "utf-8"
        ),
        hashlib.sha256,
    ).hexdigest()


def _revoke_password_reset_tokens(
    db: Session,
    *,
    user_id: int,
    exclude_id: int | None = None,
) -> int:
    query = (
        db.query(
            PasswordResetToken
        )
        .filter(
            PasswordResetToken.user_id
            == int(user_id),
            PasswordResetToken.revoked.is_(
                False
            ),
            PasswordResetToken.consumed_at.is_(
                None
            ),
        )
    )

    if exclude_id is not None:
        query = query.filter(
            PasswordResetToken.id
            != int(exclude_id)
        )

    return int(
        query.update(
            {
                "revoked": True,
            },
            synchronize_session=False,
        )
        or 0
    )


# =========================================================================
# FORGOT PASSWORD
# =========================================================================


def create_password_reset_request(
    db: Session,
    *,
    email: str,
    request_fingerprint: str | None = None,
    source_ip: str | None = None,
) -> PasswordResetGrant | None:
    """
    Create a password-reset credential.

    IMPORTANT ENUMERATION RULE
    --------------------------
    If the email does not exist, this function returns None. The API router
    must still return the SAME generic HTTP response used for an existing
    account.

    The PasswordResetGrant is for an internal email/delivery layer only.
    Never return grant.token directly from the production forgot-password
    endpoint.
    """

    normalized_email = (
        _normalize_email(
            email
        )
    )

    if not normalized_email:
        return None

    user = (
        db.query(User)
        .filter(
            User.email
            == normalized_email
        )
        .first()
    )

    # Prevent account enumeration through externally visible behaviour.
    if (
        user is None
        or not user.is_active
    ):
        _audit(
            db,
            action="auth.password.reset_requested_unknown",
            resource_user_id=None,
            actor_user_id=None,
            details={
                "account_match": False,
            },
            source_ip=source_ip,
        )

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        return None

    user_id = int(
        user.id
    )

    # A newly issued token invalidates every older unconsumed token.
    _revoke_password_reset_tokens(
        db,
        user_id=user_id,
    )

    raw_token = (
        _generate_reset_token()
    )

    token_hash = (
        _hash_reset_token(
            raw_token
        )
    )

    expires_at = (
        indian_time()
        + timedelta(
            minutes=(
                PASSWORD_RESET_EXPIRY_MINUTES
            )
        )
    )

    fingerprint = str(
        request_fingerprint or ""
    ).strip()

    if fingerprint:
        # Fingerprints are SHA256/HMAC hex values.
        if (
            len(fingerprint) != 64
            or any(
                character
                not in "0123456789abcdefABCDEF"
                for character
                in fingerprint
            )
        ):
            raise ValueError(
                "Invalid request fingerprint"
            )

        fingerprint = (
            fingerprint.lower()
        )
    else:
        fingerprint = None

    record = PasswordResetToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        request_fingerprint=(
            fingerprint
        ),
        revoked=False,
    )

    db.add(
        record
    )

    _audit(
        db,
        action="auth.password.reset_requested",
        resource_user_id=user_id,
        actor_user_id=None,
        details={
            "expires_in_minutes": (
                PASSWORD_RESET_EXPIRY_MINUTES
            ),
        },
        source_ip=source_ip,
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    return PasswordResetGrant(
        user_id=user_id,
        email=normalized_email,
        token=raw_token,
        expires_at=expires_at,
    )


# =========================================================================
# COMPLETE FORGOT-PASSWORD RESET
# =========================================================================


def reset_password_with_token(
    db: Session,
    *,
    reset_token: str,
    new_password: str,
    source_ip: str | None = None,
) -> PasswordMutationResult:
    """
    Consume a single-use password-reset token and replace the password.

    Security behaviour:
    - token is compared only through HMAC digest
    - row locked during consumption
    - expiry enforced
    - one-time consumption enforced
    - password history enforced
    - all old reset tokens revoked
    - all access tokens invalidated
    - all refresh tokens revoked
    """

    try:
        token_hash = (
            _hash_reset_token(
                reset_token
            )
        )
    except ValueError as exc:
        raise PermissionError(
            "Invalid or expired reset token"
        ) from exc

    reset_record = (
        db.query(
            PasswordResetToken
        )
        .filter(
            PasswordResetToken.token_hash
            == token_hash
        )
        .with_for_update()
        .first()
    )

    now = indian_time()

    if (
        reset_record is None
        or bool(
            reset_record.revoked
        )
        or reset_record.consumed_at
        is not None
        or reset_record.expires_at
        <= now
    ):
        if reset_record is not None:
            _audit(
                db,
                action="auth.password.reset_failed",
                resource_user_id=(
                    int(
                        reset_record.user_id
                    )
                ),
                actor_user_id=None,
                details={
                    "reason": (
                        "invalid_or_expired_token"
                    ),
                },
                source_ip=source_ip,
            )

            try:
                db.commit()
            except Exception:
                db.rollback()
                raise

        raise PermissionError(
            "Invalid or expired reset token"
        )

    user = (
        db.query(User)
        .filter(
            User.id
            == int(
                reset_record.user_id
            )
        )
        .with_for_update()
        .first()
    )

    if (
        user is None
        or not user.is_active
    ):
        reset_record.revoked = True

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        raise PermissionError(
            "Invalid or expired reset token"
        )

    _validate_password_for_user(
        user=user,
        password=new_password,
    )

    _assert_password_not_reused(
        db,
        user=user,
        new_password=new_password,
    )

    _save_current_password_to_history(
        db,
        user=user,
        reason="forgot_reset",
    )

    user.password = hash_password(
        new_password
    )

    # Mark the supplied token consumed before revoking the rest.
    reset_record.consumed_at = now
    reset_record.revoked = True

    _revoke_password_reset_tokens(
        db,
        user_id=int(
            user.id
        ),
        exclude_id=int(
            reset_record.id
        ),
    )

    revoked = revoke_user_sessions(
        db,
        user=user,
    )

    _audit(
        db,
        action="auth.password.reset_completed",
        resource_user_id=int(
            user.id
        ),
        actor_user_id=int(
            user.id
        ),
        details={
            "sessions_revoked": revoked,
            "method": "password_reset_token",
        },
        source_ip=source_ip,
    )

    try:
        db.flush()

        _prune_password_history(
            db,
            user_id=int(
                user.id
            ),
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    return PasswordMutationResult(
        user_id=int(
            user.id
        ),
        refresh_tokens_revoked=revoked,
    )


# =========================================================================
# SUPER ADMIN PASSWORD RESET
# =========================================================================


def admin_reset_password(
    db: Session,
    *,
    actor: User,
    target: User,
    new_password: str,
    source_ip: str | None = None,
) -> PasswordMutationResult:
    """
    Production password mutation primitive for the existing RBAC Super Admin
    reset-password endpoint.

    Authorization remains the responsibility of security.rbac /
    routers.rbac. This service deliberately does not duplicate RBAC policy.

    The target's existing MFA configuration is retained. Changing a password
    must not silently weaken an already-enrolled account.
    """

    actor_id = _safe_user_id(
        actor
    )

    target_id = _safe_user_id(
        target
    )

    if actor_id == target_id:
        raise ValueError(
            "Use the self-service password change flow for your own account"
        )

    _validate_password_for_user(
        user=target,
        password=new_password,
    )

    _assert_password_not_reused(
        db,
        user=target,
        new_password=new_password,
    )

    _save_current_password_to_history(
        db,
        user=target,
        reason="admin_reset",
    )

    target.password = hash_password(
        new_password
    )

    # Any emailed reset links also become invalid after an administrator
    # changes the account password.
    _revoke_password_reset_tokens(
        db,
        user_id=target_id,
    )

    revoked = revoke_user_sessions(
        db,
        user=target,
    )

    _audit(
        db,
        action="rbac.password.reset",
        resource_user_id=target_id,
        actor_user_id=actor_id,
        details={
            "sessions_revoked": revoked,
            "reset_method": "super_admin",
        },
        source_ip=source_ip,
    )

    try:
        db.flush()

        _prune_password_history(
            db,
            user_id=target_id,
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    return PasswordMutationResult(
        user_id=target_id,
        refresh_tokens_revoked=revoked,
    )


# =========================================================================
# ACCOUNT-WIDE SECURITY REVOCATION
# =========================================================================


def revoke_account_security(
    db: Session,
    *,
    actor: User,
    target: User,
    reason: str,
    source_ip: str | None = None,
) -> PasswordMutationResult:
    """
    Administrative emergency revocation helper.

    Useful when:
    - account compromise is suspected
    - user is disabled
    - credentials are believed stolen

    It does not change the password or disable MFA. It invalidates sessions
    and outstanding forgot-password tokens.
    """

    actor_id = _safe_user_id(
        actor
    )

    target_id = _safe_user_id(
        target
    )

    revoked = revoke_user_sessions(
        db,
        user=target,
    )

    _revoke_password_reset_tokens(
        db,
        user_id=target_id,
    )

    _audit(
        db,
        action="auth.account.sessions_revoked",
        resource_user_id=target_id,
        actor_user_id=actor_id,
        details={
            "reason": str(
                reason or "administrative"
            )[:100],
            "sessions_revoked": revoked,
        },
        source_ip=source_ip,
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    return PasswordMutationResult(
        user_id=target_id,
        refresh_tokens_revoked=revoked,
    )


# =========================================================================
# PASSWORD RESET TOKEN CLEANUP
# =========================================================================


def cleanup_expired_password_reset_tokens(
    db: Session,
    *,
    retention_days: int = 7,
) -> int:
    """
    Remove old consumed/revoked/expired password-reset records.

    Intended for a scheduled maintenance worker, not a request hot path.
    """

    retention_days = max(
        1,
        min(
            int(retention_days),
            90,
        ),
    )

    cutoff = (
        indian_time()
        - timedelta(
            days=retention_days
        )
    )

    rows = (
        db.query(
            PasswordResetToken
        )
        .filter(
            PasswordResetToken.expires_at
            < cutoff
        )
        .delete(
            synchronize_session=False
        )
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    return int(
        rows or 0
    )