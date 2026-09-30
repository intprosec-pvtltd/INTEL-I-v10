"""add secure hierarchical RBAC

Revision ID: f6b1c2d3e4a5
Revises: 690ae3d7577b
"""
from alembic import op
import sqlalchemy as sa

revision = "f6b1c2d3e4a5"
down_revision = "690ae3d7577b"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("department", sa.String(20), nullable=True))
    op.add_column("users", sa.Column("manager_id", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("token_version", sa.Integer(), server_default="0", nullable=False))
    op.add_column("users", sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False))
    op.create_foreign_key("fk_users_manager_id_users", "users", "users", ["manager_id"], ["id"], ondelete="SET NULL")
    op.create_index("ix_users_department", "users", ["department"])
    op.create_index("ix_users_manager_id", "users", ["manager_id"])
    op.execute("UPDATE users SET role='crime_staff', department='crime' WHERE role='user'")
    op.execute("UPDATE users SET role='super_admin', department=NULL WHERE role='admin'")


def downgrade():
    op.execute("UPDATE users SET role='admin' WHERE role='super_admin'")
    op.execute("UPDATE users SET role='user' WHERE role <> 'admin'")
    op.drop_index("ix_users_manager_id", table_name="users")
    op.drop_index("ix_users_department", table_name="users")
    op.drop_constraint("fk_users_manager_id_users", "users", type_="foreignkey")
    op.drop_column("users", "updated_at")
    op.drop_column("users", "token_version")
    op.drop_column("users", "manager_id")
    op.drop_column("users", "department")
