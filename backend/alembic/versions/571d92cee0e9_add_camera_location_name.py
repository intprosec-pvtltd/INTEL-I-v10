"""add camera location name

Revision ID: 571d92cee0e9
Revises: 7f2c1a9e4b60
"""

from alembic import op
import sqlalchemy as sa


revision = "571d92cee0e9"
down_revision = "7f2c1a9e4b60"
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()

    inspector = sa.inspect(bind)

    return column_name in {
        column["name"]
        for column in inspector.get_columns(table_name)
    }


def upgrade() -> None:
    if not _column_exists("cameras", "location_name"):
        op.add_column(
            "cameras",
            sa.Column(
                "location_name",
                sa.String(length=255),
                nullable=True,
            ),
        )


def downgrade() -> None:
    if _column_exists("cameras", "location_name"):
        op.drop_column(
            "cameras",
            "location_name",
        )