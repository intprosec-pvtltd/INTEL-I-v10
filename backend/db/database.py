import os

from dotenv import load_dotenv

from sqlalchemy import create_engine

from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
)


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


if not DATABASE_URL:

    raise RuntimeError(
        "DATABASE_URL missing in .env"
    )


engine = create_engine(
    DATABASE_URL,

    pool_pre_ping=True,

    pool_size=10,

    max_overflow=20,
)


SessionLocal = sessionmaker(
    autocommit=False,

    autoflush=False,

    bind=engine,
)


Base = declarative_base()


def getDB():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()


def get_db():

    """
    Compatibility dependency.

    Both getDB() and get_db() are supported.
    """

    yield from getDB()