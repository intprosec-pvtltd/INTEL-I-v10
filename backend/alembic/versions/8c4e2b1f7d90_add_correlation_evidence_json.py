"""add correlation evidence JSON

Revision ID: 8c4e2b1f7d90
Revises: 7a1d3c5e9f20
"""
from alembic import op
import sqlalchemy as sa

revision = "8c4e2b1f7d90"
down_revision = "7a1d3c5e9f20"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "correlation_decisions",
        sa.Column("evidence_metadata", sa.JSON(), nullable=True),
    )


def downgrade():
    op.drop_column("correlation_decisions", "evidence_metadata")
