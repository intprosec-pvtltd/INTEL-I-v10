"""
Strict Pydantic schemas for INTEL-I authentication security APIs.

Covers:
- MFA status
- TOTP enrollment
- TOTP enrollment confirmation
- MFA login verification
- recovery-code verification
- recovery-code regeneration
- MFA disable
- self-service password change
- forgot password
- password reset
- Super Admin managed-user password reset
- generic security operation responses

SECURITY DESIGN
---------------
1. All request schemas use extra="forbid".
   Unexpected fields are rejected instead of silently ignored.

2. OTP and recovery-code values are validated syntactically but are never
   persisted by these schemas.

3. Passwords are accepted only in request models. Response models never
   contain password fields.

4. Forgot-password responses are intentionally generic so callers cannot use
   the endpoint to discover whether an email address exists.

5. Password-reset tokens are accepted only in the reset request and are
   never included in public response models.

6. MFA TOTP secrets and recovery codes are exposed only in the one-time
   enrollment/creation responses where the authenticated user must receive
   them.

7. Validation here handles shape/length/normalization. The authoritative
   password-strength/history checks remain in authSecurityService.py.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


# =========================================================================
# CONSTANTS
# =========================================================================

TOTP_CODE_PATTERN = re.compile(
    r"^\d{6,8}$"
)

RECOVERY_CODE_NORMALIZED_PATTERN = re.compile(
    r"^[23456789ABCDEFGHJKLMNPQRSTUVWXYZ]{20}$"
)

RESET_TOKEN_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{32,256}$"
)

MAX_PASSWORD_CHARS = 128

MAX_PASSWORD_UTF8_BYTES = 72


# =========================================================================
# BASE SCHEMAS
# =========================================================================


class StrictRequest(BaseModel):
    """
    Base class for security-sensitive API request bodies.

    Unknown JSON properties are rejected to reduce ambiguity and accidental
    acceptance of attacker-controlled fields.
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class StrictResponse(BaseModel):
    """
    Base class for security API responses.

    from_attributes=True allows safe construction from ORM/service objects
    where appropriate.
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )


# =========================================================================
# SHARED VALIDATION HELPERS
# =========================================================================


def _validate_password_shape(
    password: str,
) -> str:
    """
    Validate transport-level password constraints.

    The service performs the full INTEL-I password policy validation,
    including:
    - password strength
    - email/name checks
    - current-password comparison
    - password-history checks

    This validator deliberately does not duplicate those rules.
    """

    if not isinstance(
        password,
        str,
    ):
        raise ValueError(
            "Password is required"
        )

    if not password:
        raise ValueError(
            "Password is required"
        )

    if len(password) < 12:
        raise ValueError(
            "Password must contain at least 12 characters"
        )

    if len(password) > MAX_PASSWORD_CHARS:
        raise ValueError(
            "Password must not exceed 128 characters"
        )

    try:
        utf8_length = len(
            password.encode(
                "utf-8"
            )
        )
    except UnicodeEncodeError as exc:
        raise ValueError(
            "Password contains invalid characters"
        ) from exc

    # Existing INTEL-I password hashing uses bcrypt.
    # bcrypt only safely considers up to 72 bytes.
    if utf8_length > MAX_PASSWORD_UTF8_BYTES:
        raise ValueError(
            "Password must not exceed 72 UTF-8 bytes"
        )

    return password


def _normalize_totp_code(
    value: str,
) -> str:
    """
    Accept common visual formatting but return only digits.

    Examples accepted:
        123456
        123 456
        123-456
    """

    normalized = (
        str(value or "")
        .strip()
        .replace(" ", "")
        .replace("-", "")
    )

    if not TOTP_CODE_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "MFA code must contain 6 to 8 digits"
        )

    return normalized


def _normalize_recovery_code(
    value: str,
) -> str:
    """
    Normalize a recovery code to the canonical form expected by the service.

    User-facing codes may look like:

        ABCDE-FGHJK-MNPQR-STUVW
    """

    normalized = (
        str(value or "")
        .strip()
        .upper()
        .replace("-", "")
        .replace(" ", "")
    )

    if not RECOVERY_CODE_NORMALIZED_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "Invalid recovery code format"
        )

    return normalized


def _normalize_mfa_factor(
    value: str,
) -> str:
    """
    A login MFA field may contain either:
    - TOTP
    - recovery code

    Keep the value normalized without exposing which check will succeed.
    """

    raw = str(
        value or ""
    ).strip()

    totp_candidate = (
        raw
        .replace(" ", "")
        .replace("-", "")
    )

    if TOTP_CODE_PATTERN.fullmatch(
        totp_candidate
    ):
        return totp_candidate

    recovery_candidate = (
        raw
        .upper()
        .replace("-", "")
        .replace(" ", "")
    )

    if RECOVERY_CODE_NORMALIZED_PATTERN.fullmatch(
        recovery_candidate
    ):
        return recovery_candidate

    raise ValueError(
        "Invalid MFA verification code format"
    )


def _validate_reset_token(
    value: str,
) -> str:
    normalized = str(
        value or ""
    ).strip()

    if not RESET_TOKEN_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "Invalid password reset token format"
        )

    return normalized


# =========================================================================
# MFA STATUS
# =========================================================================


class MFAStatusResponse(StrictResponse):
    """
    Safe account MFA status.

    No encrypted secret, raw secret, code hash, or recovery-code hash is
    exposed.
    """

    enabled: bool

    enrollment_pending: bool

    locked: bool

    locked_until: datetime | None = None

    enabled_at: datetime | None = None

    last_verified_at: datetime | None = None

    recovery_codes_remaining: int = Field(
        default=0,
        ge=0,
        le=100,
    )


# =========================================================================
# BEGIN MFA ENROLLMENT
# =========================================================================


class MFASetupRequest(StrictRequest):
    """
    Deliberately empty.

    Enrollment operates only on the currently authenticated account. The
    caller cannot submit another user_id or email address.
    """

    pass


class MFASetupResponse(StrictResponse):
    """
    One-time enrollment data returned only to the authenticated account.

    secret and provisioning_uri are sensitive and MUST NOT be logged.
    """

    secret: str = Field(
        ...,
        min_length=16,
        max_length=128,
    )

    provisioning_uri: str = Field(
        ...,
        min_length=20,
        max_length=1024,
    )

    issuer: str = Field(
        default="INTEL-I",
        min_length=1,
        max_length=100,
    )

    account_name: EmailStr

    algorithm: Literal[
        "SHA1",
        "SHA256",
        "SHA512",
    ] = "SHA1"

    digits: int = Field(
        default=6,
        ge=6,
        le=8,
    )

    period_seconds: int = Field(
        default=30,
        ge=15,
        le=60,
    )


# =========================================================================
# CONFIRM MFA ENROLLMENT
# =========================================================================


class MFASetupConfirmRequest(StrictRequest):
    totp_code: str = Field(
        ...,
        min_length=6,
        max_length=16,
    )

    @field_validator(
        "totp_code"
    )
    @classmethod
    def validate_totp_code(
        cls,
        value: str,
    ) -> str:
        return _normalize_totp_code(
            value
        )


class MFASetupConfirmResponse(StrictResponse):
    enabled: Literal[
        True
    ] = True

    recovery_codes: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
    )

    warning: str = (
        "Store these recovery codes securely. "
        "They will not be shown again."
    )

    @field_validator(
        "recovery_codes"
    )
    @classmethod
    def validate_recovery_codes(
        cls,
        value: list[str],
    ) -> list[str]:
        """
        Keep formatted server-generated recovery codes in the response while
        validating that each normalizes correctly.
        """

        if len(
            set(value)
        ) != len(value):
            raise ValueError(
                "Recovery codes must be unique"
            )

        for code in value:
            _normalize_recovery_code(
                code
            )

        return value


# =========================================================================
# CANCEL MFA ENROLLMENT
# =========================================================================


class MFACancelEnrollmentRequest(StrictRequest):
    """
    No identity parameter is accepted.

    The authenticated user can only cancel their own pending enrollment.
    """

    pass


# =========================================================================
# MFA LOGIN / STEP-UP VERIFICATION
# =========================================================================


class MFAVerifyRequest(StrictRequest):
    """
    Used for a second-factor verification challenge.

    code may be:
    - TOTP
    - one-time recovery code
    """

    code: str = Field(
        ...,
        min_length=6,
        max_length=64,
    )

    @field_validator(
        "code"
    )
    @classmethod
    def validate_code(
        cls,
        value: str,
    ) -> str:
        return _normalize_mfa_factor(
            value
        )


class MFAVerifyResponse(StrictResponse):
    verified: Literal[
        True
    ] = True

    method: Literal[
        "totp",
        "recovery",
    ]


# =========================================================================
# MFA LOGIN CHALLENGE
# =========================================================================


class MFAChallengeResponse(StrictResponse):
    """
    Returned by login when the password is correct but MFA must still be
    completed.

    mfa_token is a short-lived signed challenge token, NOT a normal access
    token and NOT a refresh token.

    The router/auth service added next will enforce its dedicated token type,
    expiry and single-user purpose.
    """

    mfa_required: Literal[
        True
    ] = True

    mfa_token: str = Field(
        ...,
        min_length=20,
        max_length=4096,
    )

    expires_in: int = Field(
        ...,
        ge=30,
        le=600,
    )


class MFALoginVerifyRequest(StrictRequest):
    """
    Complete a login that is waiting for MFA.

    The client supplies:
    - short-lived MFA challenge token
    - TOTP or recovery code
    """

    mfa_token: str = Field(
        ...,
        min_length=20,
        max_length=4096,
    )

    code: str = Field(
        ...,
        min_length=6,
        max_length=64,
    )

    @field_validator(
        "code"
    )
    @classmethod
    def validate_code(
        cls,
        value: str,
    ) -> str:
        return _normalize_mfa_factor(
            value
        )


# =========================================================================
# REGENERATE RECOVERY CODES
# =========================================================================


class MFARecoveryRegenerateRequest(StrictRequest):
    """
    High-risk operation.

    Requires:
    - current password
    - valid current TOTP

    Recovery-code regeneration intentionally does not accept an existing
    recovery code as its MFA proof.
    """

    current_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    totp_code: str = Field(
        ...,
        min_length=6,
        max_length=16,
    )

    @field_validator(
        "totp_code"
    )
    @classmethod
    def validate_totp_code(
        cls,
        value: str,
    ) -> str:
        return _normalize_totp_code(
            value
        )


class MFARecoveryRegenerateResponse(
    StrictResponse
):
    recovery_codes: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
    )

    warning: str = (
        "Previous recovery codes are now invalid. "
        "Store these new codes securely because they will not be shown again."
    )

    @field_validator(
        "recovery_codes"
    )
    @classmethod
    def validate_recovery_codes(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(
            set(value)
        ) != len(value):
            raise ValueError(
                "Recovery codes must be unique"
            )

        for code in value:
            _normalize_recovery_code(
                code
            )

        return value


# =========================================================================
# DISABLE MFA
# =========================================================================


class MFADisableRequest(StrictRequest):
    """
    MFA disable is intentionally protected by two proofs:
    - current account password
    - active TOTP/recovery factor
    """

    current_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    mfa_code: str = Field(
        ...,
        min_length=6,
        max_length=64,
    )

    @field_validator(
        "mfa_code"
    )
    @classmethod
    def validate_mfa_code(
        cls,
        value: str,
    ) -> str:
        return _normalize_mfa_factor(
            value
        )


class MFADisableResponse(StrictResponse):
    enabled: Literal[
        False
    ] = False

    message: str = (
        "Multi-factor authentication has been disabled."
    )


# =========================================================================
# SELF-SERVICE PASSWORD CHANGE
# =========================================================================


class ChangePasswordRequest(StrictRequest):
    """
    For an already authenticated user.

    When MFA is enabled on the account, mfa_code becomes mandatory in the
    service/router layer.
    """

    current_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    new_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )

    confirm_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )

    mfa_code: str | None = Field(
        default=None,
        min_length=6,
        max_length=64,
    )

    @field_validator(
        "new_password",
        "confirm_password",
    )
    @classmethod
    def validate_new_password(
        cls,
        value: str,
    ) -> str:
        return _validate_password_shape(
            value
        )

    @field_validator(
        "mfa_code"
    )
    @classmethod
    def validate_optional_mfa_code(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return _normalize_mfa_factor(
            value
        )

    @model_validator(
        mode="after"
    )
    def passwords_match(
        self,
    ) -> "ChangePasswordRequest":
        if (
            self.new_password
            != self.confirm_password
        ):
            raise ValueError(
                "New password and confirmation do not match"
            )

        if (
            self.current_password
            == self.new_password
        ):
            raise ValueError(
                "New password must be different from the current password"
            )

        return self


class ChangePasswordResponse(
    StrictResponse
):
    success: Literal[
        True
    ] = True

    sessions_revoked: int = Field(
        default=0,
        ge=0,
    )

    reauthentication_required: Literal[
        True
    ] = True

    message: str = (
        "Password changed successfully. "
        "Please sign in again."
    )


# =========================================================================
# FORGOT PASSWORD
# =========================================================================


class ForgotPasswordRequest(
    StrictRequest
):
    email: EmailStr

    @field_validator(
        "email"
    )
    @classmethod
    def normalize_email(
        cls,
        value: EmailStr,
    ) -> str:
        return (
            str(value)
            .strip()
            .lower()
        )


class ForgotPasswordResponse(
    StrictResponse
):
    """
    ALWAYS use the same response for:
    - existing account
    - nonexistent account
    - disabled account

    This prevents email/account enumeration.
    """

    accepted: Literal[
        True
    ] = True

    message: str = (
        "If an eligible account exists for that email address, "
        "password reset instructions will be sent."
    )


# =========================================================================
# RESET PASSWORD
# =========================================================================


class ResetPasswordRequest(
    StrictRequest
):
    token: str = Field(
        ...,
        min_length=32,
        max_length=256,
    )

    new_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )

    confirm_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )

    @field_validator(
        "token"
    )
    @classmethod
    def validate_token(
        cls,
        value: str,
    ) -> str:
        return _validate_reset_token(
            value
        )

    @field_validator(
        "new_password",
        "confirm_password",
    )
    @classmethod
    def validate_password(
        cls,
        value: str,
    ) -> str:
        return _validate_password_shape(
            value
        )

    @model_validator(
        mode="after"
    )
    def passwords_match(
        self,
    ) -> "ResetPasswordRequest":
        if (
            self.new_password
            != self.confirm_password
        ):
            raise ValueError(
                "New password and confirmation do not match"
            )

        return self


class ResetPasswordResponse(
    StrictResponse
):
    success: Literal[
        True
    ] = True

    sessions_revoked: int = Field(
        default=0,
        ge=0,
    )

    message: str = (
        "Password reset successfully. "
        "Please sign in using the new password."
    )


# =========================================================================
# SUPER ADMIN — MANAGED USER PASSWORD RESET
# =========================================================================


class AdminPasswordResetRequest(
    StrictRequest
):
    """
    Replacement for the simpler existing rbac.PasswordReset body.

    Authorization is still enforced by the RBAC router. A client cannot
    specify actor identity here.
    """

    new_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )

    confirm_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )

    @field_validator(
        "new_password",
        "confirm_password",
    )
    @classmethod
    def validate_password(
        cls,
        value: str,
    ) -> str:
        return _validate_password_shape(
            value
        )

    @model_validator(
        mode="after"
    )
    def passwords_match(
        self,
    ) -> "AdminPasswordResetRequest":
        if (
            self.new_password
            != self.confirm_password
        ):
            raise ValueError(
                "New password and confirmation do not match"
            )

        return self


class AdminPasswordResetResponse(
    StrictResponse
):
    success: Literal[
        True
    ] = True

    user_id: int = Field(
        ...,
        ge=1,
    )

    sessions_revoked: int = Field(
        default=0,
        ge=0,
    )

    message: str = (
        "User password reset successfully. "
        "Existing sessions have been revoked."
    )


# =========================================================================
# MFA RECOVERY LOGIN
# =========================================================================


class MFARecoveryVerifyRequest(
    StrictRequest
):
    """
    Explicit recovery-code-only verification schema.

    This is useful for the account recovery UI where TOTP and recovery code
    inputs are shown separately.
    """

    recovery_code: str = Field(
        ...,
        min_length=20,
        max_length=64,
    )

    @field_validator(
        "recovery_code"
    )
    @classmethod
    def validate_recovery_code(
        cls,
        value: str,
    ) -> str:
        return _normalize_recovery_code(
            value
        )


class MFARecoveryVerifyResponse(
    StrictResponse
):
    verified: Literal[
        True
    ] = True

    method: Literal[
        "recovery"
    ] = "recovery"

    remaining_recovery_codes: int = Field(
        ...,
        ge=0,
        le=100,
    )


# =========================================================================
# GENERIC SECURITY RESPONSE
# =========================================================================


class SecurityOperationResponse(
    StrictResponse
):
    success: bool

    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )


# =========================================================================
# SAFE ACCOUNT SECURITY SUMMARY
# =========================================================================


class AccountSecurityResponse(
    StrictResponse
):
    """
    Intended for the future frontend Account Security page.

    Contains no credential material.
    """

    user_id: int = Field(
        ...,
        ge=1,
    )

    email: EmailStr

    mfa_enabled: bool

    mfa_enrollment_pending: bool

    mfa_locked: bool

    mfa_locked_until: datetime | None = None

    recovery_codes_remaining: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    last_mfa_verified_at: datetime | None = None