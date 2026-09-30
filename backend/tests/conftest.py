import os
os.environ.setdefault("DATABASE_URL", "sqlite:///./.pytest_inteli.db")
os.environ.setdefault("KAFKA_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "true")
