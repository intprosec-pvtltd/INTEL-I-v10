"""add camera onboarding and fleet management

Revision ID: f9a1b2c3d4e5
Revises: b8d4e6f1a2c3
"""
from alembic import op
import sqlalchemy as sa

revision = "f9a1b2c3d4e5"
down_revision = "b8d4e6f1a2c3"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("camera_credentials",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(150), nullable=False), sa.Column("username", sa.String(255)), sa.Column("secret_encrypted", sa.Text(), nullable=False),
        sa.Column("auth_type", sa.String(30), nullable=False, server_default="rtsp"), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("user_id", "name", name="uq_camera_credential_user_name"))
    op.create_index("ix_camera_credentials_user_id", "camera_credentials", ["user_id"])
    op.create_table("camera_groups",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(150), nullable=False), sa.Column("parent_id", sa.Integer(), sa.ForeignKey("camera_groups.id", ondelete="CASCADE")),
        sa.Column("group_type", sa.String(50), nullable=False, server_default="logical"), sa.Column("metadata", sa.JSON()), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("user_id", "parent_id", "name", name="uq_camera_group_parent_name"))
    op.create_index("ix_camera_groups_user_id", "camera_groups", ["user_id"]); op.create_index("ix_camera_groups_parent_id", "camera_groups", ["parent_id"])
    op.create_table("camera_ai_profiles",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("name", sa.String(150), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False), sa.Column("processing_fps", sa.Float(), nullable=False, server_default="5"), sa.Column("preferred_stream", sa.String(20), nullable=False, server_default="substream"),
        sa.Column("evidence_enabled", sa.Boolean(), nullable=False, server_default=sa.true()), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("user_id", "name", name="uq_camera_ai_profile_user_name"))
    op.create_index("ix_camera_ai_profiles_user_id", "camera_ai_profiles", ["user_id"])
    op.create_table("camera_onboarding_jobs",
        sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("job_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="QUEUED"), sa.Column("total", sa.Integer(), nullable=False, server_default="0"), sa.Column("completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("successful", sa.Integer(), nullable=False, server_default="0"), sa.Column("failed", sa.Integer(), nullable=False, server_default="0"), sa.Column("duplicate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("request_metadata", sa.JSON()), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("started_at", sa.DateTime()), sa.Column("completed_at", sa.DateTime()))
    op.create_index("idx_onboarding_jobs_user_time", "camera_onboarding_jobs", ["user_id", "created_at"]); op.create_index("ix_camera_onboarding_jobs_status", "camera_onboarding_jobs", ["status"])
    op.create_table("camera_onboarding_items",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True), sa.Column("job_id", sa.String(36), sa.ForeignKey("camera_onboarding_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False), sa.Column("camera_id", sa.String(100)), sa.Column("camera_name", sa.String(150)), sa.Column("source_encrypted", sa.Text()), sa.Column("input_data_encrypted", sa.Text()),
        sa.Column("status", sa.String(30), nullable=False, server_default="QUEUED"), sa.Column("error_code", sa.String(100)), sa.Column("error_message", sa.String(500)), sa.Column("result_metadata", sa.JSON()),
        sa.Column("registered_camera_id", sa.Integer(), sa.ForeignKey("cameras.id", ondelete="SET NULL")), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("job_id", "ordinal", name="uq_onboarding_item_ordinal"))
    op.create_index("ix_camera_onboarding_items_job_id", "camera_onboarding_items", ["job_id"]); op.create_index("ix_camera_onboarding_items_status", "camera_onboarding_items", ["status"])
    columns = [
        ("last_seen_at", sa.DateTime()), ("last_error_code", sa.String(100)), ("last_error_message", sa.String(500)), ("retry_count", sa.Integer(), "0"), ("timestamp_status", sa.String(30)),
        ("district", sa.String(150)), ("jurisdiction", sa.String(150)), ("hostname", sa.String(255)), ("ip_address", sa.String(45)), ("onvif_uuid", sa.String(255)), ("serial_number", sa.String(255)),
        ("mac_address", sa.String(32)), ("firmware_version", sa.String(100)), ("main_stream_encrypted", sa.Text()), ("sub_stream_encrypted", sa.Text()),
        ("group_id", sa.Integer()), ("credential_profile_id", sa.Integer()), ("ai_profile_id", sa.Integer()), ("processing_enabled", sa.Boolean(), "false"), ("analytics_enabled", sa.Boolean(), "false"),
        ("preferred_worker_pool", sa.String(100)), ("assigned_worker_id", sa.String(150)), ("worker_status", sa.String(30)), ("assigned_at", sa.DateTime()),
    ]
    for item in columns:
        name, kind, *default = item
        op.add_column("cameras", sa.Column(name, kind, nullable=False if default else True, server_default=default[0] if default else None))
    op.create_foreign_key("fk_cameras_group_id", "cameras", "camera_groups", ["group_id"], ["id"], ondelete="SET NULL")
    op.create_foreign_key("fk_cameras_credential_profile_id", "cameras", "camera_credentials", ["credential_profile_id"], ["id"], ondelete="SET NULL")
    op.create_foreign_key("fk_cameras_ai_profile_id", "cameras", "camera_ai_profiles", ["ai_profile_id"], ["id"], ondelete="SET NULL")
    for name in ("last_seen_at", "district", "ip_address", "onvif_uuid", "serial_number", "group_id", "credential_profile_id", "ai_profile_id", "processing_enabled", "preferred_worker_pool", "assigned_worker_id"):
        op.create_index(f"ix_cameras_{name}", "cameras", [name])


def downgrade():
    for name in ("assigned_worker_id", "preferred_worker_pool", "processing_enabled", "ai_profile_id", "credential_profile_id", "group_id", "serial_number", "onvif_uuid", "ip_address", "district", "last_seen_at"):
        op.drop_index(f"ix_cameras_{name}", table_name="cameras")
    for constraint in ("fk_cameras_ai_profile_id", "fk_cameras_credential_profile_id", "fk_cameras_group_id"): op.drop_constraint(constraint, "cameras", type_="foreignkey")
    for name in ("assigned_at", "worker_status", "assigned_worker_id", "preferred_worker_pool", "analytics_enabled", "processing_enabled", "ai_profile_id", "credential_profile_id", "group_id", "sub_stream_encrypted", "main_stream_encrypted", "firmware_version", "mac_address", "serial_number", "onvif_uuid", "ip_address", "hostname", "jurisdiction", "district", "timestamp_status", "retry_count", "last_error_message", "last_error_code", "last_seen_at"): op.drop_column("cameras", name)
    op.drop_table("camera_onboarding_items"); op.drop_table("camera_onboarding_jobs"); op.drop_table("camera_ai_profiles"); op.drop_table("camera_groups"); op.drop_table("camera_credentials")
