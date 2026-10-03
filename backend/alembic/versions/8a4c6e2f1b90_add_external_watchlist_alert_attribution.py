"""Persist external watchlist provenance on existing alerts."""
from alembic import op
import sqlalchemy as sa

revision = "8a4c6e2f1b90"
down_revision = "7d91e3a5c4b8"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("alerts", sa.Column("watchlist_source_id", sa.BigInteger(), nullable=True))
    op.add_column("alerts", sa.Column("watchlist_source_name", sa.String(150), nullable=True))
    op.add_column("alerts", sa.Column("external_record_id", sa.String(200), nullable=True))
    op.add_column("alerts", sa.Column("case_reference", sa.String(200), nullable=True))
    op.add_column("alerts", sa.Column("entity_type", sa.String(20), nullable=True))
    op.create_index("ix_alerts_watchlist_source_id", "alerts", ["watchlist_source_id"])

def downgrade():
    op.drop_index("ix_alerts_watchlist_source_id", table_name="alerts")
    op.drop_column("alerts", "entity_type")
    op.drop_column("alerts", "case_reference")
    op.drop_column("alerts", "external_record_id")
    op.drop_column("alerts", "watchlist_source_name")
    op.drop_column("alerts", "watchlist_source_id")
