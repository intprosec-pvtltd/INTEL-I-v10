"""add watchlist fields to alerts

Revision ID: b8e41f7c2a91
Revises: a7c92f6e3b11
Create Date: 2026-08-25

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# ============================================================================
# ALEMBIC
# ============================================================================

revision: str = "b8e41f7c2a91"

down_revision: Union[str, Sequence[str], None] = "a7c92f6e3b11"

branch_labels: Union[str, Sequence[str], None] = None

depends_on: Union[str, Sequence[str], None] = None


# ============================================================================
# UPGRADE
# ============================================================================

def upgrade() -> None:
    """
    Add Watchlist-related fields to the existing alerts table.

    Existing alert records are preserved.
    All new fields are nullable because historical alerts do not contain
    Watchlist information.
    """

    # ------------------------------------------------------------------------
    # Watchlist entry reference
    # ------------------------------------------------------------------------

    op.add_column(
        "alerts",
        sa.Column(
            "watchlist_entry_id",
            sa.BigInteger(),
            nullable=True,
        ),
    )

    # ------------------------------------------------------------------------
    # Detected plate
    # ------------------------------------------------------------------------

    op.add_column(
        "alerts",
        sa.Column(
            "plate",
            sa.String(length=50),
            nullable=True,
        ),
    )

    # ------------------------------------------------------------------------
    # Watchlist classification
    # ------------------------------------------------------------------------

    op.add_column(
        "alerts",
        sa.Column(
            "watchlist_category",
            sa.String(length=50),
            nullable=True,
        ),
    )

    op.add_column(
        "alerts",
        sa.Column(
            "watchlist_status",
            sa.String(length=30),
            nullable=True,
        ),
    )

    # ------------------------------------------------------------------------
    # Match information
    # ------------------------------------------------------------------------

    op.add_column(
        "alerts",
        sa.Column(
            "watchlist_match_type",
            sa.String(length=30),
            nullable=True,
        ),
    )

    op.add_column(
        "alerts",
        sa.Column(
            "watchlist_match_confidence",
            sa.Float(),
            nullable=True,
        ),
    )

    # ------------------------------------------------------------------------
    # Watchlist version
    # ------------------------------------------------------------------------

    op.add_column(
        "alerts",
        sa.Column(
            "watchlist_version",
            sa.Integer(),
            nullable=True,
        ),
    )

    # =========================================================================
    # INDEXES
    # =========================================================================

    op.create_index(
        "ix_alerts_watchlist_entry_id",
        "alerts",
        ["watchlist_entry_id"],
        unique=False,
    )

    op.create_index(
        "ix_alerts_plate",
        "alerts",
        ["plate"],
        unique=False,
    )

    op.create_index(
        "ix_alerts_watchlist_category",
        "alerts",
        ["watchlist_category"],
        unique=False,
    )

    op.create_index(
        "ix_alerts_watchlist_status",
        "alerts",
        ["watchlist_status"],
        unique=False,
    )

    op.create_index(
        "ix_alerts_watchlist_match_type",
        "alerts",
        ["watchlist_match_type"],
        unique=False,
    )

    op.create_index(
        "ix_alerts_watchlist_match_confidence",
        "alerts",
        ["watchlist_match_confidence"],
        unique=False,
    )

    op.create_index(
        "ix_alerts_watchlist_version",
        "alerts",
        ["watchlist_version"],
        unique=False,
    )

    # =========================================================================
    # COMPOSITE INDEX FOR WATCHLIST ALERT DEDUPLICATION
    # =========================================================================

    op.create_index(
        "idx_alerts_watchlist_lookup",
        "alerts",
        [
            "user_id",
            "cam_id",
            "rule",
            "track_id",
            "watchlist_entry_id",
            "created_at",
        ],
        unique=False,
    )


# ============================================================================
# DOWNGRADE
# ============================================================================

def downgrade() -> None:
    """
    Remove Watchlist-related alert fields.

    Existing non-Watchlist alert fields remain untouched.
    """

    # ------------------------------------------------------------------------
    # Composite index
    # ------------------------------------------------------------------------

    op.drop_index(
        "idx_alerts_watchlist_lookup",
        table_name="alerts",
    )

    # ------------------------------------------------------------------------
    # Individual indexes
    # ------------------------------------------------------------------------

    op.drop_index(
        "ix_alerts_watchlist_version",
        table_name="alerts",
    )

    op.drop_index(
        "ix_alerts_watchlist_match_confidence",
        table_name="alerts",
    )

    op.drop_index(
        "ix_alerts_watchlist_match_type",
        table_name="alerts",
    )

    op.drop_index(
        "ix_alerts_watchlist_status",
        table_name="alerts",
    )

    op.drop_index(
        "ix_alerts_watchlist_category",
        table_name="alerts",
    )

    op.drop_index(
        "ix_alerts_plate",
        table_name="alerts",
    )

    op.drop_index(
        "ix_alerts_watchlist_entry_id",
        table_name="alerts",
    )

    # ------------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------------

    op.drop_column(
        "alerts",
        "watchlist_version",
    )

    op.drop_column(
        "alerts",
        "watchlist_match_confidence",
    )

    op.drop_column(
        "alerts",
        "watchlist_match_type",
    )

    op.drop_column(
        "alerts",
        "watchlist_status",
    )

    op.drop_column(
        "alerts",
        "watchlist_category",
    )

    op.drop_column(
        "alerts",
        "plate",
    )

    op.drop_column(
        "alerts",
        "watchlist_entry_id",
    )