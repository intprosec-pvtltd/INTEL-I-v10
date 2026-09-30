"""add distributed camera desired state

Revision ID: b8d4e6f1a2c3
Revises: e7a3c9d1f5b2
"""
from alembic import op
import sqlalchemy as sa

revision = "b8d4e6f1a2c3"
down_revision = "e7a3c9d1f5b2"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "cameras",
        sa.Column(
            "desired_state",
            sa.String(length=20),
            nullable=False,
            server_default="STOPPED",
        ),
    )
    op.add_column(
        "cameras",
        sa.Column("desired_state_updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_cameras_desired_state",
        "cameras",
        ["desired_state"],
    )
    # Existing records must not start automatically during rollout.
    op.execute("UPDATE cameras SET desired_state = 'STOPPED'")


def downgrade():
    op.drop_index("ix_cameras_desired_state", table_name="cameras")
    op.drop_column("cameras", "desired_state_updated_at")
    op.drop_column("cameras", "desired_state")
