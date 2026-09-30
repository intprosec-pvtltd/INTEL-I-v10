"""Add production MFA and password-security persistence.

Revision ID: b2e7c4f9a1d6
Revises: a9c4e7f1b2d3

Adds:
- TOTP MFA security state
- encrypted/pending MFA secret storage
- TOTP replay protection
- MFA brute-force lock state
- hashed MFA recovery codes
- hashed password-reset tokens
- password history
- deterministic indexes required by authentication/security workflows

Security:
Sensitive credential material is never stored in plaintext. The application
encrypts TOTP secrets and stores only keyed/hash representations of recovery
codes and password-reset tokens.

Migration design:
Application-level defaults intentionally remain application-level defaults.
This migration therefore does not introduce server defaults that would drift
from SQLAlchemy metadata when Alembic compare_server_default=True.
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "b2e7c4f9a1d6"
down_revision: str | Sequence[str] | None = "a9c4e7f1b2d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---------------------------------------------------------------------
    # user_mfa
    # ---------------------------------------------------------------------
    op.create_table(
        "user_mfa",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "totp_secret_encrypted",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "pending_secret_encrypted",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "encryption_key_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "totp_algorithm",
            sa.String(length=16),
            nullable=False,
        ),
        sa.Column(
            "totp_digits",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "totp_period_seconds",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "last_accepted_totp_step",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "failed_attempts",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "locked_until",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "last_failed_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "last_verified_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "enrollment_started_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "enabled_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "disabled_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "recovery_codes_generated_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_mfa_user_id_users",
            ondelete="CASCADE",
        ),
    )

    # A single unique index is sufficient for both lookup and the
    # one-MFA-record-per-user invariant. Do not duplicate it with a second
    # UNIQUE constraint on the same column.
    op.create_index(
        "ix_user_mfa_user_id",
        "user_mfa",
        ["user_id"],
        unique=True,
    )
    op.create_index(
        "ix_user_mfa_enabled",
        "user_mfa",
        ["enabled"],
        unique=False,
    )
    op.create_index(
        "ix_user_mfa_locked_until",
        "user_mfa",
        ["locked_until"],
        unique=False,
    )
    op.create_index(
        "idx_user_mfa_lock",
        "user_mfa",
        ["user_id", "locked_until"],
        unique=False,
    )

    # ---------------------------------------------------------------------
    # mfa_recovery_codes
    # ---------------------------------------------------------------------
    op.create_table(
        "mfa_recovery_codes",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "code_hash",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "used_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_mfa_recovery_codes_user_id_users",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "user_id",
            "code_hash",
            name="uq_mfa_recovery_user_hash",
        ),
    )

    op.create_index(
        "ix_mfa_recovery_codes_user_id",
        "mfa_recovery_codes",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_mfa_recovery_codes_used_at",
        "mfa_recovery_codes",
        ["used_at"],
        unique=False,
    )
    op.create_index(
        "idx_mfa_recovery_available",
        "mfa_recovery_codes",
        ["user_id", "used_at"],
        unique=False,
    )

    # ---------------------------------------------------------------------
    # password_reset_tokens
    # ---------------------------------------------------------------------
    op.create_table(
        "password_reset_tokens",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "token_hash",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "consumed_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "revoked",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "request_fingerprint",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_password_reset_tokens_user_id_users",
            ondelete="CASCADE",
        ),
    )

    # token_hash is both the lookup key and uniqueness boundary; one unique
    # index avoids a redundant UNIQUE constraint + unique-index pair.
    op.create_index(
        "ix_password_reset_tokens_token_hash",
        "password_reset_tokens",
        ["token_hash"],
        unique=True,
    )
    op.create_index(
        "ix_password_reset_tokens_user_id",
        "password_reset_tokens",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_password_reset_tokens_expires_at",
        "password_reset_tokens",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_password_reset_tokens_consumed_at",
        "password_reset_tokens",
        ["consumed_at"],
        unique=False,
    )
    op.create_index(
        "ix_password_reset_tokens_revoked",
        "password_reset_tokens",
        ["revoked"],
        unique=False,
    )
    op.create_index(
        "idx_password_reset_user_active",
        "password_reset_tokens",
        ["user_id", "revoked", "consumed_at", "expires_at"],
        unique=False,
    )

    # ---------------------------------------------------------------------
    # password_history
    # ---------------------------------------------------------------------
    op.create_table(
        "password_history",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "password_hash",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_password_history_user_id_users",
            ondelete="CASCADE",
        ),
    )

    op.create_index(
        "ix_password_history_user_id",
        "password_history",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_password_history_created_at",
        "password_history",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "idx_password_history_user_time",
        "password_history",
        ["user_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    # Drop dependent/security tables before user_mfa. All user foreign keys
    # reference users.id, so users itself is intentionally untouched.

    op.drop_index(
        "idx_password_history_user_time",
        table_name="password_history",
    )
    op.drop_index(
        "ix_password_history_created_at",
        table_name="password_history",
    )
    op.drop_index(
        "ix_password_history_user_id",
        table_name="password_history",
    )
    op.drop_table("password_history")

    op.drop_index(
        "idx_password_reset_user_active",
        table_name="password_reset_tokens",
    )
    op.drop_index(
        "ix_password_reset_tokens_revoked",
        table_name="password_reset_tokens",
    )
    op.drop_index(
        "ix_password_reset_tokens_consumed_at",
        table_name="password_reset_tokens",
    )
    op.drop_index(
        "ix_password_reset_tokens_expires_at",
        table_name="password_reset_tokens",
    )
    op.drop_index(
        "ix_password_reset_tokens_user_id",
        table_name="password_reset_tokens",
    )
    op.drop_index(
        "ix_password_reset_tokens_token_hash",
        table_name="password_reset_tokens",
    )
    op.drop_table("password_reset_tokens")

    op.drop_index(
        "idx_mfa_recovery_available",
        table_name="mfa_recovery_codes",
    )
    op.drop_index(
        "ix_mfa_recovery_codes_used_at",
        table_name="mfa_recovery_codes",
    )
    op.drop_index(
        "ix_mfa_recovery_codes_user_id",
        table_name="mfa_recovery_codes",
    )
    op.drop_table("mfa_recovery_codes")

    op.drop_index(
        "idx_user_mfa_lock",
        table_name="user_mfa",
    )
    op.drop_index(
        "ix_user_mfa_locked_until",
        table_name="user_mfa",
    )
    op.drop_index(
        "ix_user_mfa_enabled",
        table_name="user_mfa",
    )
    op.drop_index(
        "ix_user_mfa_user_id",
        table_name="user_mfa",
    )
    op.drop_table("user_mfa")
