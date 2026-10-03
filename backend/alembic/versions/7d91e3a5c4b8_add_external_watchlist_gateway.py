"""Add admin-managed external watchlist sources, records and sync history."""
from alembic import op
import sqlalchemy as sa

revision = "7d91e3a5c4b8"
down_revision = ("a9c4e7f1b2d3", "e5a7c9d2f410", "f9a1b2c3d4e5")
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("external_watchlist_sources",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("owner_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(150), nullable=False), sa.Column("description", sa.Text()),
        sa.Column("provider_type", sa.String(30), nullable=False, server_default="REST_JSON"),
        sa.Column("base_url", sa.String(2048), nullable=False), sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("read_only", sa.Boolean(), nullable=False, server_default=sa.true()), sa.Column("auth_type", sa.String(30), nullable=False, server_default="NONE"),
        sa.Column("credential_ciphertext", sa.LargeBinary()), sa.Column("api_key_header", sa.String(100)),
        sa.Column("entity_types", sa.JSON(), nullable=False), sa.Column("field_mapping", sa.JSON(), nullable=False),
        sa.Column("records_path", sa.String(250), nullable=False, server_default="records"), sa.Column("sync_mode", sa.String(20), nullable=False, server_default="INTERVAL"), sa.Column("sync_interval_seconds", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("pagination", sa.JSON(), nullable=False), sa.Column("tls_verify", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("status", sa.String(30), nullable=False, server_default="NEVER_SYNCED"), sa.Column("last_sync_at", sa.DateTime()),
        sa.Column("last_success_at", sa.DateTime()), sa.Column("last_error", sa.String(500)),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False))
    op.create_index("ix_external_watchlist_sources_owner_user_id", "external_watchlist_sources", ["owner_user_id"])
    op.create_index("ix_external_watchlist_due", "external_watchlist_sources", ["enabled", "last_sync_at"])
    op.create_table("external_watchlist_records",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.BigInteger(), sa.ForeignKey("external_watchlist_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_id", sa.String(200), nullable=False), sa.Column("entity_type", sa.String(20), nullable=False),
        sa.Column("category", sa.String(50), nullable=False), sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("priority", sa.String(20)), sa.Column("case_reference", sa.String(200)), sa.Column("name", sa.String(150)),
        sa.Column("image_available", sa.Boolean(), nullable=False, server_default=sa.false()), sa.Column("plate_display", sa.String(50)),
        sa.Column("plate_normalized", sa.String(50)), sa.Column("make", sa.String(100)), sa.Column("model", sa.String(100)), sa.Column("color", sa.String(60)),
        sa.Column("source_name", sa.String(150), nullable=False), sa.Column("raw_metadata", sa.JSON(), nullable=False),
        sa.Column("local_watchlist_entry_id", sa.BigInteger()), sa.Column("last_synced_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("source_id", "external_id", name="uq_external_watchlist_source_record"))
    op.create_index("ix_external_watchlist_records_plate_normalized", "external_watchlist_records", ["plate_normalized"])
    op.create_index("ix_external_watchlist_owner_type_active", "external_watchlist_records", ["owner_user_id", "entity_type", "active"])
    op.create_table("external_watchlist_sync_history",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.BigInteger(), sa.ForeignKey("external_watchlist_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")), sa.Column("action", sa.String(30), nullable=False),
        sa.Column("result", sa.String(30), nullable=False), sa.Column("records_received", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_created", sa.Integer(), nullable=False, server_default="0"), sa.Column("records_updated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_unchanged", sa.Integer(), nullable=False, server_default="0"), sa.Column("records_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"), sa.Column("message", sa.String(500)), sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_external_watchlist_sync_history_source_id", "external_watchlist_sync_history", ["source_id"])

def downgrade():
    op.drop_table("external_watchlist_sync_history")
    op.drop_table("external_watchlist_records")
    op.drop_table("external_watchlist_sources")
