from alembic import op
import sqlalchemy as sa

revision = "d1e7f5a9c2b4"
down_revision = "c4d9f0a1b2c3"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "person_watchlist_entries",
        sa.Column("reference_image_data", sa.LargeBinary(), nullable=True),
    )
    op.add_column(
        "person_watchlist_entries",
        sa.Column("reference_image_content_type", sa.String(100), nullable=True),
    )
    op.add_column(
        "person_watchlist_entries",
        sa.Column("reference_image_filename", sa.String(255), nullable=True),
    )
    op.add_column(
        "person_watchlist_entries",
        sa.Column("reference_image_size", sa.Integer(), nullable=True),
    )


def downgrade():
    op.drop_column(
        "person_watchlist_entries",
        "reference_image_size",
    )
    op.drop_column(
        "person_watchlist_entries",
        "reference_image_filename",
    )
    op.drop_column(
        "person_watchlist_entries",
        "reference_image_content_type",
    )
    op.drop_column(
        "person_watchlist_entries",
        "reference_image_data",
    )
