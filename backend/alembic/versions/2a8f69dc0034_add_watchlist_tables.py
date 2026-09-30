"""add watchlist tables

Revision ID: a7c92f6e3b11
Revises: 895522127a03
Create Date: 2026-08-25

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# ============================================================================
# ALEMBIC IDENTIFIERS
# ============================================================================

revision: str = "a7c92f6e3b11"

down_revision: Union[str, Sequence[str], None] = "895522127a03"

branch_labels: Union[str, Sequence[str], None] = None

depends_on: Union[str, Sequence[str], None] = None


# ============================================================================
# UPGRADE
# ============================================================================

def upgrade() -> None:
    """
    Create INTEL-I Watchlist infrastructure.

    Creates:

        watchlists
        watchlist_entries

    Existing INTEL-I tables are not modified by this migration.
    """


    # ========================================================================
    # WATCHLISTS
    # ========================================================================

    op.create_table(
        "watchlists",

        # --------------------------------------------------------------------
        # Primary key
        # --------------------------------------------------------------------

        sa.Column(
            "id",
            sa.BigInteger(),
            nullable=False,
        ),

        # --------------------------------------------------------------------
        # Owner
        # --------------------------------------------------------------------

        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),

        # --------------------------------------------------------------------
        # Watchlist information
        # --------------------------------------------------------------------

        sa.Column(
            "name",
            sa.String(length=150),
            nullable=False,
        ),

        sa.Column(
            "description",
            sa.Text(),
            nullable=True,
        ),

        # --------------------------------------------------------------------
        # Status
        # --------------------------------------------------------------------

        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),

        # --------------------------------------------------------------------
        # Timestamps
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # Foreign key
        # --------------------------------------------------------------------

        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_watchlists_user_id_users",
            ondelete="CASCADE",
        ),

        # --------------------------------------------------------------------
        # Primary key
        # --------------------------------------------------------------------

        sa.PrimaryKeyConstraint(
            "id",
            name="pk_watchlists",
        ),
    )


    # ========================================================================
    # WATCHLISTS INDEXES
    # ========================================================================

    op.create_index(
        "ix_watchlists_id",
        "watchlists",
        ["id"],
        unique=False,
    )

    op.create_index(
        "ix_watchlists_user_id",
        "watchlists",
        ["user_id"],
        unique=False,
    )

    op.create_index(
        "ix_watchlists_is_active",
        "watchlists",
        ["is_active"],
        unique=False,
    )

    op.create_index(
        "ix_watchlists_created_at",
        "watchlists",
        ["created_at"],
        unique=False,
    )

    op.create_index(
        "ix_watchlists_updated_at",
        "watchlists",
        ["updated_at"],
        unique=False,
    )

    op.create_index(
        "idx_watchlist_user_active",
        "watchlists",
        ["user_id", "is_active"],
        unique=False,
    )


    # ========================================================================
    # WATCHLIST ENTRIES
    # ========================================================================

    op.create_table(
        "watchlist_entries",

        # --------------------------------------------------------------------
        # Primary key
        # --------------------------------------------------------------------

        sa.Column(
            "id",
            sa.BigInteger(),
            nullable=False,
        ),

        # --------------------------------------------------------------------
        # Parent watchlist
        # --------------------------------------------------------------------

        sa.Column(
            "watchlist_id",
            sa.BigInteger(),
            nullable=False,
        ),

        # --------------------------------------------------------------------
        # Owner
        # --------------------------------------------------------------------

        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),

        # --------------------------------------------------------------------
        # Vehicle registration
        # --------------------------------------------------------------------

        sa.Column(
            "plate_normalized",
            sa.String(length=50),
            nullable=False,
        ),

        sa.Column(
            "plate_display",
            sa.String(length=50),
            nullable=False,
        ),

        # --------------------------------------------------------------------
        # Classification
        # --------------------------------------------------------------------

        sa.Column(
            "category",
            sa.String(length=50),
            nullable=False,
            server_default="SUSPICIOUS",
        ),

        sa.Column(
            "status",
            sa.String(length=30),
            nullable=False,
            server_default="ACTIVE",
        ),

        sa.Column(
            "priority",
            sa.String(length=20),
            nullable=False,
            server_default="HIGH",
        ),

        # --------------------------------------------------------------------
        # Additional information
        # --------------------------------------------------------------------

        sa.Column(
            "description",
            sa.Text(),
            nullable=True,
        ),

        sa.Column(
            "source",
            sa.String(length=150),
            nullable=True,
        ),

        # --------------------------------------------------------------------
        # Validity period
        # --------------------------------------------------------------------

        sa.Column(
            "effective_from",
            sa.DateTime(),
            nullable=True,
        ),

        sa.Column(
            "effective_until",
            sa.DateTime(),
            nullable=True,
        ),

        # --------------------------------------------------------------------
        # Versioning
        # --------------------------------------------------------------------

        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),

        # --------------------------------------------------------------------
        # Flexible metadata
        # --------------------------------------------------------------------

        sa.Column(
            "metadata",
            sa.JSON(),
            nullable=True,
        ),

        # --------------------------------------------------------------------
        # Timestamps
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # Watchlist foreign key
        # --------------------------------------------------------------------

        sa.ForeignKeyConstraint(
            ["watchlist_id"],
            ["watchlists.id"],
            name="fk_watchlist_entries_watchlist_id_watchlists",
            ondelete="CASCADE",
        ),

        # --------------------------------------------------------------------
        # User foreign key
        # --------------------------------------------------------------------

        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_watchlist_entries_user_id_users",
            ondelete="CASCADE",
        ),

        # --------------------------------------------------------------------
        # Primary key
        # --------------------------------------------------------------------

        sa.PrimaryKeyConstraint(
            "id",
            name="pk_watchlist_entries",
        ),
    )


    # ========================================================================
    # WATCHLIST ENTRY INDEXES
    # ========================================================================

    op.create_index(
        "ix_watchlist_entries_id",
        "watchlist_entries",
        ["id"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_watchlist_id",
        "watchlist_entries",
        ["watchlist_id"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_user_id",
        "watchlist_entries",
        ["user_id"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_plate_normalized",
        "watchlist_entries",
        ["plate_normalized"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_category",
        "watchlist_entries",
        ["category"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_status",
        "watchlist_entries",
        ["status"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_priority",
        "watchlist_entries",
        ["priority"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_effective_from",
        "watchlist_entries",
        ["effective_from"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_effective_until",
        "watchlist_entries",
        ["effective_until"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_created_at",
        "watchlist_entries",
        ["created_at"],
        unique=False,
    )

    op.create_index(
        "ix_watchlist_entries_updated_at",
        "watchlist_entries",
        ["updated_at"],
        unique=False,
    )

    # ------------------------------------------------------------------------
    # Composite indexes used by Watchlist matching
    # ------------------------------------------------------------------------

    op.create_index(
        "idx_watchlist_entry_user_plate",
        "watchlist_entries",
        ["user_id", "plate_normalized"],
        unique=False,
    )

    op.create_index(
        "idx_watchlist_entry_user_category",
        "watchlist_entries",
        ["user_id", "category"],
        unique=False,
    )

    op.create_index(
        "idx_watchlist_entry_active_effective",
        "watchlist_entries",
        [
            "user_id",
            "status",
            "effective_from",
            "effective_until",
        ],
        unique=False,
    )


# ============================================================================
# DOWNGRADE
# ============================================================================

def downgrade() -> None:
    """
    Remove INTEL-I Watchlist infrastructure.

    watchlist_entries is removed first because it references watchlists.
    """

    # ========================================================================
    # WATCHLIST ENTRY INDEXES
    # ========================================================================

    op.drop_index(
        "idx_watchlist_entry_active_effective",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "idx_watchlist_entry_user_category",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "idx_watchlist_entry_user_plate",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_updated_at",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_created_at",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_effective_until",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_effective_from",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_priority",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_status",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_category",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_plate_normalized",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_user_id",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_watchlist_id",
        table_name="watchlist_entries",
    )

    op.drop_index(
        "ix_watchlist_entries_id",
        table_name="watchlist_entries",
    )


    # ========================================================================
    # WATCHLIST ENTRIES TABLE
    # ========================================================================

    op.drop_table(
        "watchlist_entries"
    )


    # ========================================================================
    # WATCHLIST INDEXES
    # ========================================================================

    op.drop_index(
        "idx_watchlist_user_active",
        table_name="watchlists",
    )

    op.drop_index(
        "ix_watchlists_updated_at",
        table_name="watchlists",
    )

    op.drop_index(
        "ix_watchlists_created_at",
        table_name="watchlists",
    )

    op.drop_index(
        "ix_watchlists_is_active",
        table_name="watchlists",
    )

    op.drop_index(
        "ix_watchlists_user_id",
        table_name="watchlists",
    )

    op.drop_index(
        "ix_watchlists_id",
        table_name="watchlists",
    )


    # ========================================================================
    # WATCHLISTS TABLE
    # ========================================================================

    op.drop_table(
        "watchlists"
    )