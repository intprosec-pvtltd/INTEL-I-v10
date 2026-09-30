"""add snapshot integrity metadata

Revision ID: 7a1d3c5e9f20
Revises: f2c7a9d4e1b3
"""
from alembic import op
import sqlalchemy as sa

revision = "7a1d3c5e9f20"
down_revision = "f2c7a9d4e1b3"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column("snapshots", sa.Column("sha256", sa.String(length=64), nullable=True))
    op.add_column("snapshots", sa.Column("is_original", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("snapshots", sa.Column("metadata", sa.JSON(), nullable=True))
    op.create_index("ix_snapshots_sha256", "snapshots", ["sha256"], unique=False)

def downgrade() -> None:
    op.drop_index("ix_snapshots_sha256", table_name="snapshots")
    op.drop_column("snapshots", "metadata")
    op.drop_column("snapshots", "is_original")
    op.drop_column("snapshots", "sha256")
