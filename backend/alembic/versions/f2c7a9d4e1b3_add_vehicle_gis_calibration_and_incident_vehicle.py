"""add vehicle GIS calibration and incident vehicle identity

Revision ID: f2c7a9d4e1b3
Revises: d1e7f5a9c2b4
"""
from alembic import op
import sqlalchemy as sa

revision = "f2c7a9d4e1b3"
down_revision = "d1e7f5a9c2b4"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column("cameras", sa.Column("gis_calibration", sa.JSON(), nullable=True))
    op.add_column("incidents", sa.Column("global_vehicle_id", sa.String(length=100), nullable=True))
    op.create_index("ix_incidents_global_vehicle_id", "incidents", ["global_vehicle_id"], unique=False)

def downgrade() -> None:
    op.drop_index("ix_incidents_global_vehicle_id", table_name="incidents")
    op.drop_column("incidents", "global_vehicle_id")
    op.drop_column("cameras", "gis_calibration")
