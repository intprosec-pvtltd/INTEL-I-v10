"""phase 4-6 camera health, time sync and stream telemetry

Revision ID: d4f6a8c2e1b9
Revises: c1d7e9f3a5b2
"""
from alembic import op
import sqlalchemy as sa

revision = "d4f6a8c2e1b9"
down_revision = "c1d7e9f3a5b2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "camera_time_sync",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("camera_id", sa.String(100), nullable=False),
        sa.Column("camera_timestamp", sa.DateTime(), nullable=True),
        sa.Column("server_timestamp", sa.DateTime(), nullable=False),
        sa.Column("ingestion_timestamp", sa.DateTime(), nullable=False),
        sa.Column("normalized_timestamp", sa.DateTime(), nullable=False),
        sa.Column("clock_offset_ms", sa.Float(), nullable=True),
        sa.Column("jitter_ms", sa.Float(), nullable=True),
        sa.Column("sync_status", sa.String(30), nullable=False, server_default="ESTIMATED"),
        sa.Column("sync_method", sa.String(50), nullable=False, server_default="ARRIVAL_TIME"),
        sa.Column("sample_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("source_timestamp_available", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_sync_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("camera_id", name="uq_camera_time_sync_camera"),
    )
    op.create_index("idx_camera_time_sync_status", "camera_time_sync", ["sync_status"])
    op.create_index("idx_camera_time_sync_updated", "camera_time_sync", ["updated_at"])

    op.create_table(
        "stream_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("camera_id", sa.String(100), nullable=False),
        sa.Column("observed_fps", sa.Float(), nullable=True),
        sa.Column("target_fps", sa.Float(), nullable=True),
        sa.Column("decoded_frames", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("processed_frames", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("dropped_frames", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("dropped_ratio", sa.Float(), nullable=True),
        sa.Column("read_latency_ms", sa.Float(), nullable=True),
        sa.Column("queue_depth", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("queue_capacity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reconnect_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_frame_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_stream_metrics_camera_time", "stream_metrics", ["camera_id", "created_at"])


def downgrade():
    op.drop_table("stream_metrics")
    op.drop_index("idx_camera_time_sync_updated", table_name="camera_time_sync")
    op.drop_index("idx_camera_time_sync_status", table_name="camera_time_sync")
    op.drop_table("camera_time_sync")
