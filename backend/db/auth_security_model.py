from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from db.database import Base


def indian_time() -> datetime:
    """Backward-compatible name; database timestamps are canonical UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class UserMFA(Base):
    """
    Persistent MFA state for one user.

    Security invariants:
    - Exactly one MFA row may exist for a user.
    - TOTP secrets are stored only in encrypted form.
    - The last accepted TOTP time-step is persisted to prevent replay.
    - Brute-force state is persisted so it survives process restarts.
    """

    __tablename__ = "user_mfa"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )

    user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    # ------------------------------------------------------------
    # MFA STATE
    # ------------------------------------------------------------

    enabled = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ------------------------------------------------------------
    # ENCRYPTED TOTP SECRETS
    # ------------------------------------------------------------

    totp_secret_encrypted = Column(
        Text,
        nullable=True,
    )

    pending_secret_encrypted = Column(
        Text,
        nullable=True,
    )

    # Version allows future encryption-key rotation/migration.
    encryption_key_version = Column(
        Integer,
        nullable=False,
        default=1,
    )

    # ------------------------------------------------------------
    # TOTP CONFIGURATION
    # ------------------------------------------------------------

    totp_algorithm = Column(
        String(16),
        nullable=False,
        default="SHA1",
    )

    totp_digits = Column(
        Integer,
        nullable=False,
        default=6,
    )

    totp_period_seconds = Column(
        Integer,
        nullable=False,
        default=30,
    )

    # ------------------------------------------------------------
    # REPLAY / BRUTE-FORCE PROTECTION
    # ------------------------------------------------------------

    last_accepted_totp_step = Column(
        BigInteger,
        nullable=True,
    )

    failed_attempts = Column(
        Integer,
        nullable=False,
        default=0,
    )

    locked_until = Column(
        DateTime,
        nullable=True,
        index=True,
    )

    last_failed_at = Column(
        DateTime,
        nullable=True,
    )

    last_verified_at = Column(
        DateTime,
        nullable=True,
    )

    # ------------------------------------------------------------
    # MFA LIFECYCLE
    # ------------------------------------------------------------

    enrollment_started_at = Column(
        DateTime,
        nullable=True,
    )

    enabled_at = Column(
        DateTime,
        nullable=True,
    )

    disabled_at = Column(
        DateTime,
        nullable=True,
    )

    recovery_codes_generated_at = Column(
        DateTime,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        onupdate=indian_time,
    )

    __table_args__ = (
        # `enabled` already has index=True above, which creates
        # ix_user_mfa_enabled. Do not create a second redundant enabled
        # index here; keeping only the composite lock index prevents
        # Alembic metadata drift and duplicate PostgreSQL indexes.
        Index(
            "idx_user_mfa_lock",
            "user_id",
            "locked_until",
        ),
    )


# ========================================================================
# MFA RECOVERY CODES
# ========================================================================


class MFARecoveryCode(Base):
    """
    Hashed one-time MFA recovery code.

    IMPORTANT:
        `code_hash` contains only a keyed digest of a recovery code.

    The plaintext recovery code exists only when originally generated and
    returned to the authenticated user. It must never be written to:
    - PostgreSQL
    - Redis
    - application logs
    - audit details
    - exception traces

    After successful use, used_at is permanently populated so the code
    cannot be reused.
    """

    __tablename__ = "mfa_recovery_codes"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )

    user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    code_hash = Column(
        String(64),
        nullable=False,
    )

    used_at = Column(
        DateTime,
        nullable=True,
        index=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "code_hash",
            name="uq_mfa_recovery_user_hash",
        ),
        Index(
            "idx_mfa_recovery_available",
            "user_id",
            "used_at",
        ),
    )


# ========================================================================
# PASSWORD RESET TOKENS
# ========================================================================


class PasswordResetToken(Base):
    """
    Secure password-reset request.

    Only a digest of the externally delivered reset token is stored.

    The actual token is generated with cryptographically secure randomness
    and is shown/sent only to the intended user. Database compromise alone
    therefore does not expose active reset links.

    A token becomes unusable when any of the following are true:
    - consumed_at is populated
    - revoked is True
    - expires_at has passed

    Creating a newer password-reset request will also revoke previous active
    reset tokens for that user in the service layer.
    """

    __tablename__ = "password_reset_tokens"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )

    user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    token_hash = Column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
    )

    expires_at = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    consumed_at = Column(
        DateTime,
        nullable=True,
        index=True,
    )

    revoked = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # Hash/fingerprint only. Never persist a raw reset token here.
    request_fingerprint = Column(
        String(64),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
    )

    __table_args__ = (
        Index(
            "idx_password_reset_user_active",
            "user_id",
            "revoked",
            "consumed_at",
            "expires_at",
        ),
    )


# ========================================================================
# PASSWORD HISTORY
# ========================================================================


class PasswordHistory(Base):
    """
    Previous password hashes used for password-reuse prevention.

    This table stores the SAME TYPE of slow password hash used for normal
    authentication, never plaintext passwords.

    The authentication service can check the last N password hashes before
    allowing a password change/reset.

    Historical hashes are intentionally associated only with user_id.
    """

    __tablename__ = "password_history"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )

    user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    password_hash = Column(
        String(255),
        nullable=False,
    )

    reason = Column(
        String(32),
        nullable=False,
        default="change",
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=indian_time,
        index=True,
    )

    __table_args__ = (
        Index(
            "idx_password_history_user_time",
            "user_id",
            "created_at",
        ),
    )


# ========================================================================
# SECURITY HELPERS
# ========================================================================


def is_reset_token_active(
    reset_token: PasswordResetToken,
    *,
    now: datetime | None = None,
) -> bool:
    """
    Return True only while a password-reset token is still usable.

    All persisted security timestamps in this module are naive IST values,
    matching the existing INTEL-I database timestamp convention.
    """

    current_time = now or indian_time()

    if reset_token.revoked:
        return False

    if reset_token.consumed_at is not None:
        return False

    if reset_token.expires_at <= current_time:
        return False

    return True


def is_mfa_locked(
    mfa: UserMFA,
    *,
    now: datetime | None = None,
) -> bool:
    """
    Return True while the user's persisted MFA lockout is still active.
    """

    if mfa.locked_until is None:
        return False

    current_time = now or indian_time()

    return mfa.locked_until > current_time
