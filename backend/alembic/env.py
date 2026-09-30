from __future__ import annotations

from logging.config import fileConfig
import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool


# =========================================================================
# ENVIRONMENT
# =========================================================================

load_dotenv()


# =========================================================================
# ALEMBIC CONFIGURATION
# =========================================================================

config = context.config

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL missing in environment/.env. "
        "Alembic migrations cannot run without an explicit database."
    )

# Alembic's ConfigParser treats '%' as interpolation syntax. Escaping the
# percent signs here preserves URL-encoded credentials such as %40.
config.set_main_option(
    "sqlalchemy.url",
    DATABASE_URL.replace("%", "%%"),
)

# Configure Python logging from alembic.ini when available.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# =========================================================================
# SQLALCHEMY METADATA
# =========================================================================

from db.database import Base  # noqa: E402


# -------------------------------------------------------------------------
# EXISTING INTEL-I MODELS
#
# These imports are required so every production table is registered in
# Base.metadata before Alembic performs metadata comparison/autogeneration.
# -------------------------------------------------------------------------

from db import model  # noqa: E402,F401
from db import intelligence_model  # noqa: E402,F401
from db import advanced_intelligence_model  # noqa: E402,F401
from db import watchlist_model  # noqa: E402,F401


# -------------------------------------------------------------------------
# MFA + PASSWORD SECURITY MODELS
#
# Registers:
#   user_mfa
#   mfa_recovery_codes
#   password_reset_tokens
#   password_history
# -------------------------------------------------------------------------

from db import auth_security_model  # noqa: E402,F401


target_metadata = Base.metadata


# =========================================================================
# MIGRATION CONFIGURATION
# =========================================================================


def _configure_context_common() -> dict:
    """
    Return the schema-comparison options shared by online/offline migrations.

    INTEL-I currently stores application tables in the default PostgreSQL
    schema. We intentionally keep include_schemas=False so Alembic
    autogeneration does not inspect PostgreSQL/PostGIS/system schemas and
    accidentally propose destructive migrations for extension-owned objects.
    """

    return {
        "target_metadata": target_metadata,

        # Detect meaningful SQLAlchemy type changes.
        "compare_type": True,

        # Detect server-side default changes. The MFA/password-security
        # migration is intentionally aligned with ORM metadata so this does
        # not immediately generate default drift.
        "compare_server_default": True,

        # INTEL-I models currently use the default PostgreSQL schema.
        # PostGIS does not require this to be True merely because the
        # extension is installed.
        "include_schemas": False,

        # Each Alembic revision is applied transactionally.
        "transaction_per_migration": True,
    }


# =========================================================================
# OFFLINE MIGRATIONS
# =========================================================================


def run_migrations_offline() -> None:
    """
    Emit migration SQL without opening a database connection.

    Example:
        alembic upgrade head --sql
    """

    url = config.get_main_option("sqlalchemy.url")

    if not url:
        raise RuntimeError(
            "Alembic SQLAlchemy URL is not configured."
        )

    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        **_configure_context_common(),
    )

    with context.begin_transaction():
        context.run_migrations()


# =========================================================================
# ONLINE MIGRATIONS
# =========================================================================


def run_migrations_online() -> None:
    """
    Apply migrations using a live database connection.

    NullPool is intentional because Alembic is a short-lived administrative
    process and should not retain application-style pooled connections.
    """

    configuration = config.get_section(
        config.config_ini_section,
        {},
    )

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            **_configure_context_common(),
        )

        with context.begin_transaction():
            context.run_migrations()


# =========================================================================
# ENTRYPOINT
# =========================================================================


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
