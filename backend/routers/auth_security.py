"""
INTEL-I MFA and password-security API routes.

Endpoints
---------
GET  /auth/mfa/status
POST /auth/mfa/setup
POST /auth/mfa/setup/confirm
POST /auth/mfa/verify
POST /auth/mfa/recovery/regenerate
POST /auth/mfa/disable

POST /auth/password/change
POST /auth/password/forgot
POST /auth/password/reset

Security properties
-------------------
- TOTP MFA works for every authenticated INTEL-I role.
- MFA secrets are never logged.
- OTP values are never logged.
- Recovery codes are never logged.
- Passwords are never logged.
- Password-reset tokens are never logged.
- Forgot-password responses do not reveal whether an account exists.
- Password-reset tokens are delivered only through the configured delivery
  channel and never returned by the public forgot-password endpoint.
- Sensitive endpoints use SlowAPI rate limiting.
- Source-IP extraction does not blindly trust X-Forwarded-For.
- Password mutations revoke existing access/refresh sessions.
- Existing INTEL-I AuditLog is used by the service layer.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any
from urllib.parse import quote, urlparse

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from db.database import getDB
from db.model import User
from schemas.auth_security import (
    AccountSecurityResponse,
    ChangePasswordRequest,
    ChangePasswordResponse,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    MFADisableRequest,
    MFADisableResponse,
    MFARecoveryRegenerateRequest,
    MFARecoveryRegenerateResponse,
    MFASetupConfirmRequest,
    MFASetupConfirmResponse,
    MFASetupResponse,
    MFAStatusResponse,
    MFAVerifyRequest,
    MFAVerifyResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    SecurityOperationResponse,
)
from security.rateLimit import limiter
from services.authSecurityService import (
    MFA_ISSUER,
    MFA_TOTP_DIGITS,
    MFA_TOTP_PERIOD_SECONDS,
    begin_totp_enrollment,
    build_request_fingerprint,
    change_password,
    confirm_totp_enrollment,
    create_password_reset_request,
    disable_mfa,
    get_mfa_status,
    regenerate_recovery_codes,
    reset_password_with_token,
)


logger = logging.getLogger("auth-security-router")


router = APIRouter(
    prefix="/auth",
    tags=["Authentication Security"],
)


# =========================================================================
# ENVIRONMENT HELPERS
# =========================================================================


def _env_bool(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.getenv(
        name,
        "true" if default else "false",
    )

    return str(raw).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be an integer"
        ) from exc

    if value < minimum or value > maximum:
        raise RuntimeError(
            f"{name} must be between "
            f"{minimum} and {maximum}"
        )

    return value


ENV = os.getenv(
    "ENV",
    "dev",
).strip().lower()

IS_PROD = ENV in {
    "prod",
    "production",
}


# =========================================================================
# SOURCE IP
# =========================================================================


def _configured_trusted_proxies() -> set[str]:
    """
    Return proxy IP addresses explicitly trusted by the application.

    Never trust X-Forwarded-For merely because it exists.
    """

    raw = os.getenv(
        "AUTH_TRUSTED_PROXY_IPS",
        "",
    )

    return {
        item.strip()
        for item in raw.split(",")
        if item.strip()
    }


def _source_ip(
    request: Request,
) -> str | None:
    """
    Resolve the client IP safely.

    Default:
        request.client.host

    X-Forwarded-For is honored only when:
    - AUTH_TRUST_PROXY_HEADERS=true
    - the immediate peer is explicitly present in AUTH_TRUSTED_PROXY_IPS

    In a production reverse-proxy deployment, the proxy must also remove any
    client-supplied X-Forwarded-For header before inserting its own.
    """

    immediate_peer = None

    if request.client is not None:
        immediate_peer = str(
            request.client.host or ""
        ).strip()

    if (
        _env_bool(
            "AUTH_TRUST_PROXY_HEADERS",
            False,
        )
        and immediate_peer
        and immediate_peer
        in _configured_trusted_proxies()
    ):
        forwarded = str(
            request.headers.get(
                "x-forwarded-for",
                "",
            )
        ).strip()

        if forwarded:
            candidate = (
                forwarded.split(
                    ",",
                    1,
                )[0]
                .strip()
            )

            if candidate:
                return candidate[:64]

        real_ip = str(
            request.headers.get(
                "x-real-ip",
                "",
            )
        ).strip()

        if real_ip:
            return real_ip[:64]

    if immediate_peer:
        return immediate_peer[:64]

    return None


# =========================================================================
# COOKIE INVALIDATION
# =========================================================================


def _delete_cookie(
    response: Response,
    *,
    key: str,
    path: str,
) -> None:
    """
    Match the cookie configuration already used by INTEL-I main.py.

    Password change/reset increments token_version and revokes database
    refresh-token records. Deleting the browser cookies immediately avoids
    leaving known-invalid credentials in the browser.
    """

    cookie_domain = (
        os.getenv(
            "COOKIE_DOMAIN",
            "",
        ).strip()
        or None
    )

    cookie_secure = _env_bool(
        "COOKIE_SECURE",
        IS_PROD,
    )

    cookie_samesite = (
        os.getenv(
            "COOKIE_SAMESITE",
            "none" if IS_PROD else "lax",
        )
        .strip()
        .lower()
    )

    if cookie_samesite not in {
        "lax",
        "strict",
        "none",
    }:
        cookie_samesite = (
            "none"
            if IS_PROD
            else "lax"
        )

    response.delete_cookie(
        key=key,
        path=path,
        domain=cookie_domain,
        secure=cookie_secure,
        samesite=cookie_samesite,
    )


def _clear_auth_cookies(
    response: Response,
) -> None:
    _delete_cookie(
        response,
        key="access_token",
        path="/",
    )

    _delete_cookie(
        response,
        key="refresh_token",
        path="/auth",
    )

    _delete_cookie(
        response,
        key="csrf_token",
        path="/",
    )


# =========================================================================
# SERVICE ERROR TRANSLATION
# =========================================================================


def _detail_from_exception(
    exc: Exception,
    fallback: str,
) -> str:
    """
    Extract only ordinary validation messages.

    Do not use repr(exc), because it can unexpectedly include internal state.
    """

    text = str(exc).strip()

    if not text:
        return fallback

    return text[:500]


def _raise_validation_error(
    exc: Exception,
    *,
    fallback: str,
) -> None:
    if isinstance(
        exc,
        HTTPException,
    ):
        raise exc

    if isinstance(
        exc,
        PermissionError,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_detail_from_exception(
                exc,
                fallback,
            ),
        )

    if isinstance(
        exc,
        ValueError,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_detail_from_exception(
                exc,
                fallback,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=fallback,
    )


# =========================================================================
# PASSWORD RESET DELIVERY
# =========================================================================


def _password_reset_frontend_url() -> str:
    """
    Resolve the frontend password-reset page.

    Preferred:
        PASSWORD_RESET_FRONTEND_URL=https://intel-i.example/reset-password

    Fallback:
        FRONTEND_URL + /reset-password

    Production requires HTTPS.
    """

    explicit = os.getenv(
        "PASSWORD_RESET_FRONTEND_URL",
        "",
    ).strip()

    if explicit:
        url = explicit
    else:
        frontend = os.getenv(
            "FRONTEND_URL",
            "",
        ).strip().rstrip("/")

        if not frontend:
            raise RuntimeError(
                "PASSWORD_RESET_FRONTEND_URL or FRONTEND_URL "
                "must be configured"
            )

        url = (
            f"{frontend}/reset-password"
        )

    parsed = urlparse(
        url
    )

    if parsed.scheme not in {
        "http",
        "https",
    }:
        raise RuntimeError(
            "Password reset frontend URL must use HTTP or HTTPS"
        )

    if not parsed.netloc:
        raise RuntimeError(
            "Password reset frontend URL is invalid"
        )

    if IS_PROD and parsed.scheme != "https":
        raise RuntimeError(
            "Production password reset frontend URL must use HTTPS"
        )

    return url


def _build_reset_url(
    token: str,
) -> str:
    """
    Build reset URL without ever logging it.

    The reset token is placed in the URL only for delivery to the intended
    user's mailbox.
    """

    base_url = (
        _password_reset_frontend_url()
    )

    separator = (
        "&"
        if "?" in base_url
        else "?"
    )

    return (
        f"{base_url}"
        f"{separator}"
        f"token={quote(token, safe='')}"
    )


def _smtp_configuration() -> dict[str, Any]:
    """
    Load SMTP settings only when password-reset delivery is actually needed.

    Required in production:
        SMTP_HOST
        SMTP_FROM_EMAIL

    Authentication is optional so an internal authenticated relay or managed
    mail relay can also be used.
    """

    host = os.getenv(
        "SMTP_HOST",
        "",
    ).strip()

    from_email = os.getenv(
        "SMTP_FROM_EMAIL",
        "",
    ).strip()

    if not host:
        raise RuntimeError(
            "SMTP_HOST is required for password reset delivery"
        )

    if not from_email:
        raise RuntimeError(
            "SMTP_FROM_EMAIL is required for password reset delivery"
        )

    port = _env_int(
        "SMTP_PORT",
        587,
        minimum=1,
        maximum=65535,
    )

    timeout = _env_int(
        "SMTP_TIMEOUT_SECONDS",
        15,
        minimum=3,
        maximum=60,
    )

    username = os.getenv(
        "SMTP_USERNAME",
        "",
    ).strip()

    password = os.getenv(
        "SMTP_PASSWORD",
        "",
    )

    use_ssl = _env_bool(
        "SMTP_SSL",
        False,
    )

    use_starttls = _env_bool(
        "SMTP_STARTTLS",
        True,
    )

    if use_ssl and use_starttls:
        raise RuntimeError(
            "SMTP_SSL and SMTP_STARTTLS cannot both be enabled"
        )

    if IS_PROD and not (
        use_ssl
        or use_starttls
    ):
        raise RuntimeError(
            "Production SMTP must use TLS"
        )

    if bool(username) != bool(password):
        raise RuntimeError(
            "SMTP_USERNAME and SMTP_PASSWORD must be configured together"
        )

    return {
        "host": host,
        "port": port,
        "timeout": timeout,
        "username": username,
        "password": password,
        "from_email": from_email,
        "use_ssl": use_ssl,
        "use_starttls": use_starttls,
    }


def _deliver_password_reset_email(
    *,
    recipient: str,
    token: str,
) -> None:
    """
    Deliver a reset credential.

    The caller MUST NOT log:
    - token
    - reset URL
    - message body

    This function also deliberately does not return the reset URL.
    """

    config = (
        _smtp_configuration()
    )

    reset_url = _build_reset_url(
        token
    )

    subject = os.getenv(
        "PASSWORD_RESET_EMAIL_SUBJECT",
        "INTEL-I password reset",
    ).strip()

    if not subject:
        subject = (
            "INTEL-I password reset"
        )

    message = EmailMessage()

    message["Subject"] = subject
    message["From"] = config[
        "from_email"
    ]
    message["To"] = recipient

    reset_expiry = os.getenv(
        "PASSWORD_RESET_EXPIRY_MINUTES",
        "20",
    ).strip()

    message.set_content(
        "\n".join(
            [
                "A password reset was requested for your INTEL-I account.",
                "",
                "Use the secure link below to choose a new password:",
                "",
                reset_url,
                "",
                (
                    "This link is single-use and expires in "
                    f"{reset_expiry} minutes."
                ),
                "",
                (
                    "If you did not request this password reset, "
                    "you can ignore this message."
                ),
                "",
                (
                    "INTEL-I security will never ask you to send your "
                    "password or MFA code by email."
                ),
            ]
        )
    )

    tls_context = (
        ssl.create_default_context()
    )

    if config["use_ssl"]:
        with smtplib.SMTP_SSL(
            host=config["host"],
            port=config["port"],
            timeout=config[
                "timeout"
            ],
            context=tls_context,
        ) as smtp:
            if config["username"]:
                smtp.login(
                    config[
                        "username"
                    ],
                    config[
                        "password"
                    ],
                )

            smtp.send_message(
                message
            )

        return

    with smtplib.SMTP(
        host=config["host"],
        port=config["port"],
        timeout=config[
            "timeout"
        ],
    ) as smtp:
        smtp.ehlo()

        if config[
            "use_starttls"
        ]:
            smtp.starttls(
                context=tls_context
            )

            smtp.ehlo()

        if config[
            "username"
        ]:
            smtp.login(
                config[
                    "username"
                ],
                config[
                    "password"
                ],
            )

        smtp.send_message(
            message
        )


# =========================================================================
# MFA STATUS
# =========================================================================


@router.get(
    "/mfa/status",
    response_model=MFAStatusResponse,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_MFA_STATUS",
        "60/minute",
    )
)
def mfa_status(
    request: Request,
    response: Response,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Return safe MFA state for the currently authenticated user.

    No secret/ciphertext/hash is exposed.
    """

    try:
        return MFAStatusResponse(
            **get_mfa_status(
                db,
                int(
                    current_user.id
                ),
            )
        )

    except Exception:
        logger.exception(
            "Unable to read MFA status | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to read MFA status",
        )


# =========================================================================
# ACCOUNT SECURITY SUMMARY
# =========================================================================


@router.get(
    "/security",
    response_model=AccountSecurityResponse,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_ACCOUNT_SECURITY",
        "60/minute",
    )
)
def account_security(
    request: Request,
    response: Response,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Safe account-security view for the frontend Account Security page.
    """

    try:
        mfa = get_mfa_status(
            db,
            int(
                current_user.id
            ),
        )

        return AccountSecurityResponse(
            user_id=int(
                current_user.id
            ),
            email=current_user.email,
            mfa_enabled=bool(
                mfa["enabled"]
            ),
            mfa_enrollment_pending=bool(
                mfa[
                    "enrollment_pending"
                ]
            ),
            mfa_locked=bool(
                mfa["locked"]
            ),
            mfa_locked_until=mfa[
                "locked_until"
            ],
            recovery_codes_remaining=int(
                mfa[
                    "recovery_codes_remaining"
                ]
            ),
            last_mfa_verified_at=mfa[
                "last_verified_at"
            ],
        )

    except Exception:
        logger.exception(
            "Unable to read account security | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to read account security information",
        )


# =========================================================================
# BEGIN MFA SETUP
# =========================================================================


@router.post(
    "/mfa/setup",
    response_model=MFASetupResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_MFA_SETUP",
        "5/hour",
    )
)
def mfa_setup(
    request: Request,
    response: Response,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Start TOTP enrollment for the current authenticated account.

    The plaintext TOTP secret is returned exactly here so the user can add it
    to an authenticator application. The service stores only ciphertext.
    """

    try:
        result = (
            begin_totp_enrollment(
                db,
                user=current_user,
                source_ip=_source_ip(
                    request
                ),
            )
        )

        return MFASetupResponse(
            secret=result.secret,
            provisioning_uri=(
                result.provisioning_uri
            ),
            issuer=MFA_ISSUER,
            account_name=(
                current_user.email
            ),
            algorithm="SHA1",
            digits=MFA_TOTP_DIGITS,
            period_seconds=(
                MFA_TOTP_PERIOD_SECONDS
            ),
        )

    except (
        ValueError,
        PermissionError,
        HTTPException,
    ) as exc:
        _raise_validation_error(
            exc,
            fallback=(
                "Unable to start MFA setup"
            ),
        )

    except Exception:
        logger.exception(
            "MFA enrollment start failed | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to start MFA setup",
        )


# =========================================================================
# CONFIRM MFA SETUP
# =========================================================================


@router.post(
    "/mfa/setup/confirm",
    response_model=MFASetupConfirmResponse,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_MFA_SETUP_CONFIRM",
        "10/minute",
    )
)
def mfa_setup_confirm(
    request: Request,
    response: Response,
    payload: MFASetupConfirmRequest,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Verify the first TOTP and activate MFA.

    Recovery codes are displayed once after successful enrollment.
    """

    try:
        result = (
            confirm_totp_enrollment(
                db,
                user=current_user,
                totp_code=(
                    payload.totp_code
                ),
                source_ip=_source_ip(
                    request
                ),
            )
        )

        return MFASetupConfirmResponse(
            enabled=True,
            recovery_codes=list(
                result.recovery_codes
            ),
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_detail_from_exception(
                exc,
                "Unable to confirm MFA setup",
            ),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "MFA enrollment confirmation failed | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to confirm MFA setup",
        )


# =========================================================================
# VERIFY MFA / STEP-UP
# =========================================================================


@router.post(
    "/mfa/verify",
    response_model=MFAVerifyResponse,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_MFA_VERIFY",
        "10/minute",
    )
)
def mfa_verify(
    request: Request,
    response: Response,
    payload: MFAVerifyRequest,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Authenticated MFA step-up verification.

    code may be:
    - authenticator TOTP
    - unused recovery code

    Login-time MFA is wired separately when /auth/login is updated because
    the login flow must not issue a normal access token before MFA succeeds.
    """

    from services.authSecurityService import (
        verify_mfa,
    )

    try:
        method = verify_mfa(
            db,
            user_id=int(
                current_user.id
            ),
            code=payload.code,
            allow_recovery_code=True,
            source_ip=_source_ip(
                request
            ),
        )

        return MFAVerifyResponse(
            verified=True,
            method=method,
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_detail_from_exception(
                exc,
                "Unable to verify MFA",
            ),
        )

    except Exception:
        logger.exception(
            "MFA verification failed unexpectedly | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to verify MFA",
        )


# =========================================================================
# REGENERATE RECOVERY CODES
# =========================================================================


@router.post(
    "/mfa/recovery/regenerate",
    response_model=(
        MFARecoveryRegenerateResponse
    ),
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_MFA_RECOVERY_REGENERATE",
        "3/hour",
    )
)
def mfa_recovery_regenerate(
    request: Request,
    response: Response,
    payload: MFARecoveryRegenerateRequest,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Replace all MFA recovery codes.

    Requires:
    - current password
    - current authenticator TOTP

    Existing recovery codes become unusable immediately.
    """

    try:
        result = (
            regenerate_recovery_codes(
                db,
                user=current_user,
                current_password=(
                    payload.current_password
                ),
                mfa_code=(
                    payload.totp_code
                ),
                source_ip=_source_ip(
                    request
                ),
            )
        )

        return (
            MFARecoveryRegenerateResponse(
                recovery_codes=list(
                    result.recovery_codes
                ),
            )
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Password or MFA verification failed"
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_detail_from_exception(
                exc,
                "Unable to regenerate recovery codes",
            ),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "MFA recovery-code regeneration failed | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to regenerate MFA recovery codes",
        )


# =========================================================================
# DISABLE MFA
# =========================================================================


@router.post(
    "/mfa/disable",
    response_model=MFADisableResponse,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_MFA_DISABLE",
        "5/hour",
    )
)
def mfa_disable(
    request: Request,
    response: Response,
    payload: MFADisableRequest,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Disable MFA only after password + second-factor verification.
    """

    try:
        disable_mfa(
            db,
            user=current_user,
            current_password=(
                payload.current_password
            ),
            mfa_code=(
                payload.mfa_code
            ),
            source_ip=_source_ip(
                request
            ),
        )

        return MFADisableResponse(
            enabled=False,
            message=(
                "Multi-factor authentication has been disabled."
            ),
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Password or MFA verification failed"
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_detail_from_exception(
                exc,
                "Unable to disable MFA",
            ),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "MFA disable failed | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to disable MFA",
        )


# =========================================================================
# SELF-SERVICE CHANGE PASSWORD
# =========================================================================


@router.post(
    "/password/change",
    response_model=(
        ChangePasswordResponse
    ),
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_PASSWORD_CHANGE",
        "5/hour",
    )
)
def password_change(
    request: Request,
    response: Response,
    payload: ChangePasswordRequest,
    db: Session = Depends(
        getDB
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Change the current user's password.

    If MFA is enabled, an MFA code/recovery code is mandatory.

    Successful change:
    - stores current password hash in bounded history
    - writes new bcrypt hash
    - increments User.token_version
    - revokes all RefreshToken records
    - records an audit event
    - removes authentication cookies from this browser
    """

    try:
        result = change_password(
            db,
            user=current_user,
            current_password=(
                payload.current_password
            ),
            new_password=(
                payload.new_password
            ),
            mfa_code=(
                payload.mfa_code
            ),
            source_ip=_source_ip(
                request
            ),
        )

        _clear_auth_cookies(
            response
        )

        return ChangePasswordResponse(
            success=True,
            sessions_revoked=(
                result.refresh_tokens_revoked
            ),
            reauthentication_required=True,
            message=(
                "Password changed successfully. "
                "Please sign in again."
            ),
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Password or MFA verification failed"
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_detail_from_exception(
                exc,
                "Unable to change password",
            ),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Password change failed | user_id=%s",
            current_user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to change password",
        )


# =========================================================================
# FORGOT PASSWORD
# =========================================================================


@router.post(
    "/password/forgot",
    response_model=(
        ForgotPasswordResponse
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_PASSWORD_FORGOT",
        "5/hour",
    )
)
def password_forgot(
    request: Request,
    response: Response,
    payload: ForgotPasswordRequest,
    db: Session = Depends(
        getDB
    ),
):
    """
    Request password-reset instructions.

    CRITICAL:
    The HTTP response is identical for:
    - existing account
    - nonexistent account
    - disabled account
    - email delivery failure

    This prevents account enumeration.

    A successful reset token is NEVER returned from this endpoint.
    """

    generic_response = (
        ForgotPasswordResponse(
            accepted=True,
            message=(
                "If an eligible account exists for that email address, "
                "password reset instructions will be sent."
            ),
        )
    )

    source_ip = _source_ip(
        request
    )

    user_agent = str(
        request.headers.get(
            "user-agent",
            "",
        )
    )[:512]

    try:
        fingerprint = (
            build_request_fingerprint(
                source_ip=source_ip,
                user_agent=user_agent,
            )
        )

        grant = (
            create_password_reset_request(
                db,
                email=str(
                    payload.email
                ),
                request_fingerprint=(
                    fingerprint
                ),
                source_ip=source_ip,
            )
        )

    except Exception:
        # Do not include submitted email or any credential material.
        logger.exception(
            "Password-reset request processing failed"
        )

        # The public response stays generic.
        return generic_response

    if grant is None:
        return generic_response

    try:
        _deliver_password_reset_email(
            recipient=grant.email,
            token=grant.token,
        )

    except Exception:
        # Never log:
        # - grant.token
        # - reset URL
        # - message body
        # - SMTP password
        #
        # user_id is safe operational metadata.
        logger.exception(
            "Password-reset delivery failed | user_id=%s",
            grant.user_id,
        )

        # Still return the same generic response so SMTP state cannot be used
        # as an account-enumeration side channel.
        return generic_response

    logger.info(
        "Password-reset instructions dispatched | user_id=%s",
        grant.user_id,
    )

    return generic_response


# =========================================================================
# RESET PASSWORD
# =========================================================================


@router.post(
    "/password/reset",
    response_model=(
        ResetPasswordResponse
    ),
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_PASSWORD_RESET",
        "10/hour",
    )
)
def password_reset(
    request: Request,
    response: Response,
    payload: ResetPasswordRequest,
    db: Session = Depends(
        getDB
    ),
):
    """
    Consume a single-use reset token.

    Token is:
    - never logged
    - never returned
    - persisted only as HMAC-SHA256 digest
    - expiry checked
    - one-time consumed
    """

    try:
        result = (
            reset_password_with_token(
                db,
                reset_token=(
                    payload.token
                ),
                new_password=(
                    payload.new_password
                ),
                source_ip=_source_ip(
                    request
                ),
            )
        )

        # Also remove any stale auth cookie present in the browser performing
        # the reset.
        _clear_auth_cookies(
            response
        )

        return ResetPasswordResponse(
            success=True,
            sessions_revoked=(
                result.refresh_tokens_revoked
            ),
            message=(
                "Password reset successfully. "
                "Please sign in using the new password."
            ),
        )

    except PermissionError:
        # Do not distinguish:
        # - missing token
        # - expired token
        # - consumed token
        # - revoked token
        # - unknown token
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid or expired password reset link"
            ),
        )

    except ValueError as exc:
        # Password-policy/history failures may be returned because possession
        # of a valid reset credential has already been established inside
        # the service before the password mutation succeeds.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_detail_from_exception(
                exc,
                "Unable to reset password",
            ),
        )

    except HTTPException:
        raise

    except Exception:
        # Never log payload/token.
        logger.exception(
            "Password reset failed unexpectedly"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to reset password",
        )


# =========================================================================
# OPTIONAL HEALTH / READINESS HELPER
# =========================================================================


@router.get(
    "/security/readiness",
    response_model=SecurityOperationResponse,
    include_in_schema=False,
)
@limiter.limit(
    os.getenv(
        "RATE_LIMIT_SECURITY_READINESS",
        "30/minute",
    )
)
def security_readiness(
    request: Request,
    response: Response,
    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Authenticated lightweight configuration check.

    Does not reveal secret values.

    Kept out of OpenAPI because it is an operational diagnostic, not a
    normal application feature.
    """

    missing: list[str] = []

    required = (
        "MFA_TOTP_ENCRYPTION_KEY",
        "MFA_RECOVERY_CODE_PEPPER",
        "PASSWORD_RESET_TOKEN_PEPPER",
        "AUTH_FINGERPRINT_PEPPER",
    )

    for name in required:
        if not os.getenv(
            name,
            "",
        ).strip():
            missing.append(
                name
            )

    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Authentication security configuration is incomplete"
            ),
        )

    return SecurityOperationResponse(
        success=True,
        message=(
            "Authentication security configuration is available."
        ),
    )