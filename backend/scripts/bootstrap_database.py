"""Safely initialize a new INTEL-I database or migrate an existing one."""
from __future__ import annotations

import subprocess
from sqlalchemy import inspect, text

from db.database import Base, engine
from db import model  # noqa: F401
from db import intelligence_model  # noqa: F401
from db import advanced_intelligence_model  # noqa: F401
from db import watchlist_model  # noqa: F401


def alembic(*args: str) -> None:
    subprocess.run(["alembic", *args], check=True)


def main() -> None:
    tables = set(inspect(engine).get_table_names())
    if not tables:
        Base.metadata.create_all(bind=engine)
        alembic("stamp", "head")
        print("[database] Fresh schema created and stamped at Alembic head")
        return
    if "alembic_version" not in tables:
        raise RuntimeError(
            "Database contains tables but has no alembic_version. "
            "Refusing to guess its migration state. Back it up and migrate it explicitly."
        )
    alembic("upgrade", "head")
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    print("[database] Alembic migration completed")


if __name__ == "__main__":
    main()
