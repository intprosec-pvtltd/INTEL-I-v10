"""Add tenant-safe person correlation and persisted investigation summaries.

Revision ID: e5a7c9d2f410
Revises: d4f6a8c2e1b9
"""
from alembic import op
import sqlalchemy as sa

revision = "e5a7c9d2f410"
down_revision = "d4f6a8c2e1b9"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("persons", sa.Column("user_id", sa.BigInteger(), nullable=True))
    op.add_column("person_observations", sa.Column("user_id", sa.BigInteger(), nullable=True))
    op.execute("UPDATE persons SET user_id = 0 WHERE user_id IS NULL")
    op.execute("UPDATE person_observations SET user_id = 0 WHERE user_id IS NULL")
    op.alter_column("persons", "user_id", nullable=False)
    op.alter_column("person_observations", "user_id", nullable=False)
    op.create_index("ix_persons_user_id", "persons", ["user_id"])
    op.create_index("ix_person_observations_user_id", "person_observations", ["user_id"])

    op.create_table(
        "person_correlation_decisions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("camera_id", sa.String(100), nullable=False),
        sa.Column("local_track_id", sa.String(100), nullable=False),
        sa.Column("global_person_id", sa.String(100), nullable=False),
        sa.Column("candidate_global_person_id", sa.String(100), nullable=True),
        sa.Column("appearance_score", sa.Float(), nullable=True),
        sa.Column("temporal_score", sa.Float(), nullable=True),
        sa.Column("topology_score", sa.Float(), nullable=True),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(30), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    for name, cols in (
        ("ix_person_correlation_decisions_user_id", ["user_id"]),
        ("ix_person_correlation_decisions_camera_id", ["camera_id"]),
        ("ix_person_correlation_decisions_local_track_id", ["local_track_id"]),
        ("ix_person_correlation_decisions_global_person_id", ["global_person_id"]),
        ("ix_person_correlation_decisions_candidate_global_person_id", ["candidate_global_person_id"]),
        ("ix_person_correlation_decisions_final_score", ["final_score"]),
        ("ix_person_correlation_decisions_decision", ["decision"]),
        ("ix_person_correlation_decisions_created_at", ["created_at"]),
        ("idx_person_corr_user_time", ["user_id", "created_at"]),
        ("idx_person_corr_global_time", ["global_person_id", "created_at"]),
    ):
        op.create_index(name, "person_correlation_decisions", cols)

    op.create_table(
        "investigation_summaries",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("incident_id", sa.BigInteger(), nullable=True),
        sa.Column("subject_type", sa.String(30), nullable=False),
        sa.Column("subject_id", sa.String(100), nullable=False),
        sa.Column("source_version", sa.String(100), nullable=False),
        sa.Column("model_name", sa.String(150), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("structured_summary", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    for name, cols in (
        ("ix_investigation_summaries_user_id", ["user_id"]),
        ("ix_investigation_summaries_incident_id", ["incident_id"]),
        ("ix_investigation_summaries_subject_type", ["subject_type"]),
        ("ix_investigation_summaries_subject_id", ["subject_id"]),
        ("ix_investigation_summaries_status", ["status"]),
        ("ix_investigation_summaries_created_at", ["created_at"]),
        ("idx_summary_subject", ["user_id", "subject_type", "subject_id"]),
    ):
        op.create_index(name, "investigation_summaries", cols)


def downgrade():
    op.drop_table("investigation_summaries")
    op.drop_table("person_correlation_decisions")
    op.drop_index("ix_person_observations_user_id", table_name="person_observations")
    op.drop_index("ix_persons_user_id", table_name="persons")
    op.drop_column("person_observations", "user_id")
    op.drop_column("persons", "user_id")
