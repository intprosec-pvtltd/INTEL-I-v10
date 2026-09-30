"""complete phase 0-3 camera contract

Revision ID: b7e4c9d1a2f3
Revises: 9e2f6a7b4c10
"""
from alembic import op
import sqlalchemy as sa

revision = "b7e4c9d1a2f3"
down_revision = "9e2f6a7b4c10"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("cameras", sa.Column("direction", sa.String(length=20), nullable=True))
    op.add_column("cameras", sa.Column("stream_fps", sa.Float(), nullable=True))
    op.add_column("cameras", sa.Column("stream_width", sa.Integer(), nullable=True))
    op.add_column("cameras", sa.Column("stream_height", sa.Integer(), nullable=True))
    op.add_column("cameras", sa.Column("transport", sa.String(length=20), nullable=True))
    op.add_column("cameras", sa.Column("codec", sa.String(length=32), nullable=True))
    op.execute("UPDATE cameras SET connection_state = 'OFFLINE' WHERE connection_state IS NULL OR connection_state = 'DISCONNECTED'")
    op.execute("UPDATE cameras SET connection_state = 'ONLINE' WHERE connection_state = 'CONNECTED'")
    op.execute("UPDATE cameras SET transport = 'tcp' WHERE source_type = 'rtsp' AND (transport IS NULL OR transport = '')")


def downgrade():
    op.drop_column("cameras", "codec")
    op.drop_column("cameras", "transport")
    op.drop_column("cameras", "stream_height")
    op.drop_column("cameras", "stream_width")
    op.drop_column("cameras", "stream_fps")
    op.drop_column("cameras", "direction")
