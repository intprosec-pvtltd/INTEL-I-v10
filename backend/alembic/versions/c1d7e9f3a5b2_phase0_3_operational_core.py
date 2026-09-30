"""phase 0-3 operational core tables

Revision ID: c1d7e9f3a5b2
Revises: b7e4c9d1a2f3
"""
from alembic import op
import sqlalchemy as sa

revision = "c1d7e9f3a5b2"
down_revision = "b7e4c9d1a2f3"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("camera_health",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("camera_id", sa.String(length=100), nullable=False),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("frame_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("last_frame_at", sa.DateTime(), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_camera_health_camera_time", "camera_health", ["camera_id", "created_at"])
    op.create_index("idx_camera_health_state_time", "camera_health", ["state", "created_at"])
    op.create_table("evidence_hashes",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("evidence_key", sa.String(length=500), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("algorithm", sa.String(length=20), nullable=False, server_default="SHA-256"),
        sa.Column("is_original", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("evidence_key", name="uq_evidence_hash_key"),
    )
    op.create_index("idx_evidence_hash_sha256", "evidence_hashes", ["sha256"])
    op.create_table("audit_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("resource_type", sa.String(length=100), nullable=False),
        sa.Column("resource_id", sa.String(length=150), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("source_ip", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
    )
    op.create_index("idx_audit_user_time", "audit_logs", ["user_id", "created_at"])
    op.create_index("idx_audit_resource_time", "audit_logs", ["resource_type", "resource_id", "created_at"])
    op.create_table("system_events",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=30), nullable=False, server_default="INFO"),
        sa.Column("component", sa.String(length=100), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table("watchlist_versions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("watchlist_id", sa.BigInteger(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["watchlist_id"], ["watchlists.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("watchlist_id", "version", name="uq_watchlist_version"),
    )


def downgrade():
    op.drop_table("watchlist_versions")
    op.drop_table("system_events")
    op.drop_table("audit_logs")
    op.drop_table("evidence_hashes")
    op.drop_table("camera_health")
