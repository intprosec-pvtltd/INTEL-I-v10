"""add unified camera connector integration fields

Revision ID: 7f2c1a9e4b60
Revises: 438a1c72de3e
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "7f2c1a9e4b60"
down_revision: Union[str, Sequence[str], None] = "438a1c72de3e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("connector_type", sa.String(50), nullable=True))
    op.add_column("cameras", sa.Column("vendor", sa.String(100), nullable=True))
    op.add_column("cameras", sa.Column("vendor_device_id", sa.String(150), nullable=True))
    op.add_column("cameras", sa.Column("stream_profile", sa.String(256), nullable=True))
    op.add_column("cameras", sa.Column("connector_config_encrypted", sa.Text(), nullable=True))
    op.add_column("cameras", sa.Column("resolved_source_type", sa.String(50), nullable=True))
    op.add_column("cameras", sa.Column("connection_state", sa.String(30), nullable=True))
    op.add_column("cameras", sa.Column("last_health_check", sa.DateTime(), nullable=True))

    op.create_index(
        "ix_cameras_connector_type",
        "cameras",
        ["connector_type"],
        unique=False,
    )
    op.execute(
        "UPDATE cameras SET connection_state = 'DISCONNECTED' "
        "WHERE connection_state IS NULL"
    )


def downgrade() -> None:
    op.drop_index("ix_cameras_connector_type", table_name="cameras")
    op.drop_column("cameras", "last_health_check")
    op.drop_column("cameras", "connection_state")
    op.drop_column("cameras", "resolved_source_type")
    op.drop_column("cameras", "connector_config_encrypted")
    op.drop_column("cameras", "stream_profile")
    op.drop_column("cameras", "vendor_device_id")
    op.drop_column("cameras", "vendor")
    op.drop_column("cameras", "connector_type")
