"""add camera catalogue integrations and external synchronization provenance

Revision ID: a9c4e7f1b2d3
Revises: f6b1c2d3e4a5
"""
from alembic import op
import sqlalchemy as sa


revision = "a9c4e7f1b2d3"
down_revision = "f6b1c2d3e4a5"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("alerts", sa.Column("source_timestamp", sa.DateTime(), nullable=True))
    op.add_column("alerts", sa.Column("source_pts_seconds", sa.Float(), nullable=True))
    op.add_column("alerts", sa.Column("timestamp_source", sa.String(length=40), nullable=True))
    op.add_column("alerts", sa.Column("timestamp_quality", sa.String(length=30), nullable=True))
    op.create_index("ix_alerts_source_timestamp", "alerts", ["source_timestamp"])
    op.create_index("ix_alerts_timestamp_quality", "alerts", ["timestamp_quality"])

    op.add_column("camera_time_sync", sa.Column("pts_seconds", sa.Float(), nullable=True))
    op.add_column("camera_time_sync", sa.Column("timestamp_quality", sa.String(length=30), nullable=True))
    op.add_column("camera_time_sync", sa.Column("discontinuity_count", sa.BigInteger(), nullable=False, server_default="0"))
    op.add_column("camera_time_sync", sa.Column("last_discontinuity_reason", sa.String(length=80), nullable=True))
    op.add_column("stream_metrics", sa.Column("discontinuity_count", sa.BigInteger(), nullable=False, server_default="0"))

    op.create_table(
        "camera_integrations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("provider_type", sa.String(length=50), nullable=False),
        sa.Column("config_encrypted", sa.Text(), nullable=False),
        sa.Column("config_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("auto_sync", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("sync_interval_seconds", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("last_sync_started_at", sa.DateTime(), nullable=True),
        sa.Column("last_sync_at", sa.DateTime(), nullable=True),
        sa.Column("last_sync_status", sa.String(length=30), nullable=True),
        sa.Column("last_error_code", sa.String(length=80), nullable=True),
        sa.Column("last_error_message", sa.String(length=500), nullable=True),
        sa.Column("last_camera_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_created_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_updated_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_offline_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_revision", sa.String(length=128), nullable=True),
        sa.Column("etag", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("user_id", "name", name="uq_camera_integration_user_name"),
    )
    op.create_index("ix_camera_integrations_id", "camera_integrations", ["id"])
    op.create_index("ix_camera_integrations_user_id", "camera_integrations", ["user_id"])
    op.create_index("ix_camera_integrations_provider_type", "camera_integrations", ["provider_type"])
    op.create_index("ix_camera_integrations_enabled", "camera_integrations", ["enabled"])
    op.create_index("ix_camera_integrations_auto_sync", "camera_integrations", ["auto_sync"])
    op.create_index("ix_camera_integrations_last_sync_at", "camera_integrations", ["last_sync_at"])
    op.create_index("ix_camera_integrations_last_sync_status", "camera_integrations", ["last_sync_status"])
    op.create_index("idx_camera_integration_due", "camera_integrations", ["enabled", "auto_sync", "last_sync_at"])

    op.add_column("cameras", sa.Column("integration_id", sa.Integer(), nullable=True))
    op.add_column("cameras", sa.Column("external_provider", sa.String(length=50), nullable=True))
    op.add_column("cameras", sa.Column("external_camera_id", sa.String(length=150), nullable=True))
    op.add_column("cameras", sa.Column("external_live_status", sa.String(length=30), nullable=True))
    op.add_column("cameras", sa.Column("external_metadata", sa.JSON(), nullable=True))
    op.add_column("cameras", sa.Column("external_metadata_hash", sa.String(length=64), nullable=True))
    op.add_column("cameras", sa.Column("external_last_seen_at", sa.DateTime(), nullable=True))
    op.add_column("cameras", sa.Column("external_last_synced_at", sa.DateTime(), nullable=True))
    op.add_column("cameras", sa.Column("externally_managed", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.create_foreign_key(
        "fk_cameras_integration_id_camera_integrations",
        "cameras",
        "camera_integrations",
        ["integration_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_cameras_integration_id", "cameras", ["integration_id"])
    op.create_index("ix_cameras_external_provider", "cameras", ["external_provider"])
    op.create_index("ix_cameras_external_camera_id", "cameras", ["external_camera_id"])
    op.create_index("ix_cameras_external_live_status", "cameras", ["external_live_status"])
    op.create_index("ix_cameras_external_metadata_hash", "cameras", ["external_metadata_hash"])
    op.create_index(
        "uq_camera_external_identity",
        "cameras",
        ["user_id", "external_provider", "external_camera_id"],
        unique=True,
    )

    op.create_table(
        "camera_integration_sync_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("integration_id", sa.Integer(), sa.ForeignKey("camera_integrations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("trigger", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("discovered_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("offline_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_revision", sa.String(length=128), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_camera_integration_sync_runs_id", "camera_integration_sync_runs", ["id"])
    op.create_index("ix_camera_integration_sync_runs_integration_id", "camera_integration_sync_runs", ["integration_id"])
    op.create_index("ix_camera_integration_sync_runs_status", "camera_integration_sync_runs", ["status"])
    op.create_index("ix_camera_integration_sync_runs_started_at", "camera_integration_sync_runs", ["started_at"])
    op.create_index("idx_camera_integration_run_time", "camera_integration_sync_runs", ["integration_id", "started_at"])


def downgrade():
    op.drop_table("camera_integration_sync_runs")
    op.drop_index("uq_camera_external_identity", table_name="cameras")
    op.drop_index("ix_cameras_external_metadata_hash", table_name="cameras")
    op.drop_index("ix_cameras_external_live_status", table_name="cameras")
    op.drop_index("ix_cameras_external_camera_id", table_name="cameras")
    op.drop_index("ix_cameras_external_provider", table_name="cameras")
    op.drop_index("ix_cameras_integration_id", table_name="cameras")
    op.drop_constraint("fk_cameras_integration_id_camera_integrations", "cameras", type_="foreignkey")
    for column in (
        "externally_managed", "external_last_synced_at", "external_last_seen_at",
        "external_metadata_hash", "external_metadata", "external_live_status",
        "external_camera_id", "external_provider", "integration_id",
    ):
        op.drop_column("cameras", column)
    op.drop_table("camera_integrations")
    op.drop_column("stream_metrics", "discontinuity_count")
    op.drop_column("camera_time_sync", "last_discontinuity_reason")
    op.drop_column("camera_time_sync", "discontinuity_count")
    op.drop_column("camera_time_sync", "timestamp_quality")
    op.drop_column("camera_time_sync", "pts_seconds")
    op.drop_index("ix_alerts_timestamp_quality", table_name="alerts")
    op.drop_index("ix_alerts_source_timestamp", table_name="alerts")
    op.drop_column("alerts", "timestamp_quality")
    op.drop_column("alerts", "timestamp_source")
    op.drop_column("alerts", "source_pts_seconds")
    op.drop_column("alerts", "source_timestamp")
