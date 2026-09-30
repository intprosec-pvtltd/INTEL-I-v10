#!/usr/bin/env bash
set -Eeuo pipefail

log() {
    printf '[entrypoint] %s\n' "$*"
}

fail() {
    printf '[entrypoint] ERROR: %s\n' "$*" >&2
    exit 1
}

: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be configured in RunPod}"
: "${REDIS_PASSWORD:?REDIS_PASSWORD must be configured in RunPod}"
: "${FRONTEND_URL:?FRONTEND_URL must be configured in RunPod}"

export POSTGRES_USER="${POSTGRES_USER:-intel_i}"
export POSTGRES_DB="${POSTGRES_DB:-cctv_db}"

# PostgreSQL must use the container disk because RunPod Network Volumes
# normally prevent PostgreSQL ownership and permission requirements.
export PGDATA="${PGDATA:-/var/lib/postgresql/data}"

# These directories use the persistent RunPod Network Volume.
export REDIS_DATA_DIR="${REDIS_DATA_DIR:-/workspace/redis-data}"
export UPLOAD_DIR="${UPLOAD_DIR:-/workspace/uploads}"
export KAFKA_ENABLED="${KAFKA_ENABLED:-false}"
export ENV="${ENV:-prod}"

# Transport-security boundary.
#
# TLS is terminated by the production ingress/reverse proxy (for example the
# RunPod HTTPS proxy). Uvicorn therefore receives plain HTTP on the private
# container hop, but it MUST trust forwarded scheme/client information only
# from the configured proxy addresses.
#
# In RunPod's managed proxy topology the proxy source address can vary, so "*"
# is the practical default. Do not expose port 8000 directly to the public
# Internet when FORWARDED_ALLOW_IPS="*".
export FORWARDED_ALLOW_IPS="${FORWARDED_ALLOW_IPS:-*}"
export UVICORN_TIMEOUT_KEEP_ALIVE="${UVICORN_TIMEOUT_KEEP_ALIVE:-15}"
export UVICORN_WS_PING_INTERVAL="${UVICORN_WS_PING_INTERVAL:-20}"
export UVICORN_WS_PING_TIMEOUT="${UVICORN_WS_PING_TIMEOUT:-20}"
export UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN="${UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN:-30}"

# INTEL-I RTSP ingestion is intentionally RTSP-over-TCP. This environment
# value is consumed by the existing camera/FFmpeg pipeline.
export RTSP_TRANSPORT="tcp"

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

[[ "$POSTGRES_USER" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] ||
    fail "Invalid POSTGRES_USER"

[[ "$POSTGRES_DB" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] ||
    fail "Invalid POSTGRES_DB"

# Production browser/API traffic must enter through HTTPS. This protects
# authentication cookies, MFA exchanges, password-reset tokens and WebSockets
# (WSS at the browser edge). The application itself remains behind the trusted
# TLS-terminating proxy.
if [[ "${ENV,,}" == "prod" ]]; then
    [[ "$FRONTEND_URL" == https://* ]] ||
        fail "Production FRONTEND_URL must use https://"

    if [[ -n "${CORS_ALLOWED_ORIGINS:-}" ]]; then
        IFS=',' read -r -a _cors_origins <<< "$CORS_ALLOWED_ORIGINS"
        for _origin in "${_cors_origins[@]}"; do
            _origin="${_origin#"${_origin%%[![:space:]]*}"}"
            _origin="${_origin%"${_origin##*[![:space:]]}"}"
            [[ -z "$_origin" || "$_origin" == https://* ]] ||
                fail "Production CORS_ALLOWED_ORIGINS must contain only https:// origins"
        done
        unset _cors_origins _origin
    fi
fi

[[ "$UVICORN_TIMEOUT_KEEP_ALIVE" =~ ^[0-9]+$ ]] ||
    fail "UVICORN_TIMEOUT_KEEP_ALIVE must be an integer"

[[ "$UVICORN_WS_PING_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    fail "UVICORN_WS_PING_INTERVAL must be numeric"

[[ "$UVICORN_WS_PING_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    fail "UVICORN_WS_PING_TIMEOUT must be numeric"

[[ "$UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN" =~ ^[0-9]+$ ]] ||
    fail "UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN must be an integer"

PG_VERSION="$(
    find /usr/lib/postgresql \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -printf '%f\n' |
        sort -V |
        tail -1
)"

[[ -n "$PG_VERSION" ]] ||
    fail "PostgreSQL binaries were not found"

PG_BIN="/usr/lib/postgresql/${PG_VERSION}/bin"

mkdir -p \
    "$REDIS_DATA_DIR" \
    "$UPLOAD_DIR" \
    /workspace/keys \
    /app/outbox

postgres_uid="$(id -u postgres)"

if [[ ! -d "$PGDATA" ]]; then
    runuser -u postgres -- mkdir -p "$PGDATA" ||
        fail "PostgreSQL cannot create $PGDATA"
fi

pgdata_owner_uid="$(stat -c '%u' "$PGDATA")"

[[ "$pgdata_owner_uid" == "$postgres_uid" ]] ||
    fail "$PGDATA must be owned by UID $postgres_uid; current owner UID is $pgdata_owner_uid"

if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
    log "Initializing PostgreSQL at $PGDATA"

    runuser -u postgres -- "$PG_BIN/initdb" \
        -D "$PGDATA" \
        --auth-local=trust \
        --auth-host=scram-sha-256
fi

if runuser -u postgres -- "$PG_BIN/pg_isready" \
    -q \
    -h 127.0.0.1
then
    log "PostgreSQL is already running"
else
    log "Starting PostgreSQL"

    runuser -u postgres -- "$PG_BIN/pg_ctl" \
        -D "$PGDATA" \
        -l /workspace/postgres.log \
        -o "-c listen_addresses=127.0.0.1 -c password_encryption=scram-sha-256" \
        start
fi

for _ in $(seq 1 60); do
    if runuser -u postgres -- "$PG_BIN/pg_isready" \
        -q \
        -h 127.0.0.1
    then
        break
    fi

    sleep 1
done

runuser -u postgres -- "$PG_BIN/pg_isready" \
    -q \
    -h 127.0.0.1 ||
    fail "PostgreSQL did not become ready"

log "Configuring PostgreSQL role and database"

python - <<'PY'
import os

import psycopg2
from psycopg2 import sql

role_name = os.environ["POSTGRES_USER"]
role_password = os.environ["POSTGRES_PASSWORD"]
database_name = os.environ["POSTGRES_DB"]

connection = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    host="/var/run/postgresql",
)
connection.autocommit = True

try:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s",
            (role_name,),
        )

        if cursor.fetchone() is None:
            cursor.execute(
                sql.SQL("CREATE ROLE {} WITH LOGIN").format(
                    sql.Identifier(role_name)
                )
            )

        cursor.execute(
            sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD %s").format(
                sql.Identifier(role_name)
            ),
            (role_password,),
        )

        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database_name,),
        )

        if cursor.fetchone() is None:
            cursor.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(database_name),
                    sql.Identifier(role_name),
                )
            )

        cursor.execute(
            sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(database_name),
                sql.Identifier(role_name),
            )
        )
finally:
    connection.close()

print("[entrypoint] PostgreSQL role and database configured")
PY

export DATABASE_URL="$(
    python - <<'PY'
import os
from urllib.parse import quote

print(
    "postgresql://{}:{}@127.0.0.1:5432/{}".format(
        quote(os.environ["POSTGRES_USER"], safe=""),
        quote(os.environ["POSTGRES_PASSWORD"], safe=""),
        quote(os.environ["POSTGRES_DB"], safe=""),
    )
)
PY
)"

export REDIS_URL="$(
    python - <<'PY'
import os
from urllib.parse import quote

print(
    "redis://:{}@127.0.0.1:6379/0".format(
        quote(os.environ["REDIS_PASSWORD"], safe="")
    )
)
PY
)"

if redis-cli \
    -h 127.0.0.1 \
    -a "$REDIS_PASSWORD" \
    --no-auth-warning \
    ping 2>/dev/null |
    grep -qx PONG
then
    log "Redis is already running"
else
    log "Starting password-protected Redis"

    redis-server \
        --daemonize yes \
        --bind 127.0.0.1 \
        --protected-mode yes \
        --appendonly yes \
        --appendfilename appendonly.aof \
        --dir "$REDIS_DATA_DIR" \
        --requirepass "$REDIS_PASSWORD"
fi

log "Waiting for Redis to become ready"

redis_ready=false

for attempt in $(seq 1 30)
do
    if redis-cli \
        -h 127.0.0.1 \
        -a "$REDIS_PASSWORD" \
        --no-auth-warning \
        ping 2>/dev/null |
        grep -qx PONG
    then
        redis_ready=true
        log "Redis is ready"
        break
    fi

    log "Waiting for Redis (${attempt}/30)"
    sleep 1
done

if [[ "$redis_ready" != "true" ]]
then
    fail "Redis did not become ready after 30 seconds"
fi

if [[ ! -s /workspace/keys/private_key.pem ||
      ! -s /workspace/keys/public_key.pem ]]
then
    log "Generating persistent JWT signing keys"

    openssl genpkey \
        -algorithm RSA \
        -pkeyopt rsa_keygen_bits:3072 \
        -out /workspace/keys/private_key.pem

    openssl rsa \
        -pubout \
        -in /workspace/keys/private_key.pem \
        -out /workspace/keys/public_key.pem

    chmod 600 /workspace/keys/private_key.pem
fi

export PRIVATE_KEY_PATH=/workspace/keys/private_key.pem
export PUBLIC_KEY_PATH=/workspace/keys/public_key.pem

if [[ -z "${CAMERA_SOURCE_ENCRYPTION_KEY:-}" ]]; then
    if [[ ! -s /workspace/keys/camera_source.key ]]; then
        python - <<'PY' > /workspace/keys/camera_source.key
from cryptography.fernet import Fernet

print(Fernet.generate_key().decode("ascii"))
PY

        chmod 600 /workspace/keys/camera_source.key
    fi

    export CAMERA_SOURCE_ENCRYPTION_KEY="$(
        tr -d '\r\n' < /workspace/keys/camera_source.key
    )"
fi

if [[ -z "${PERSON_EMBEDDING_KEY:-}" ]]; then
    if [[ ! -s /workspace/keys/person_embedding.key ]]; then
        python - <<'PY' > /workspace/keys/person_embedding.key
from cryptography.fernet import Fernet

print(Fernet.generate_key().decode("ascii"))
PY

        chmod 600 /workspace/keys/person_embedding.key
    fi

    export PERSON_EMBEDDING_KEY="$(
        tr -d '\r\n' < /workspace/keys/person_embedding.key
    )"
fi

python - <<'PY'
import os
from cryptography.fernet import Fernet

for name in (
    "CAMERA_SOURCE_ENCRYPTION_KEY",
    "PERSON_EMBEDDING_KEY",
):
    try:
        Fernet(os.environ[name].encode("ascii"))
    except Exception as exc:
        raise SystemExit(f"Invalid {name}: {exc}") from exc

print("[entrypoint] Persistent data-encryption keys validated")
PY

log "Validating model assets"
if [[ "${MODEL_AUTO_PROVISION:-false}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]|[Oo][Nn])$ ]]; then
    log "Provisioning enabled model artifacts from the approved manifest"
    python /app/scripts/provision_models.py
fi

python /app/scripts/preflight_models.py

log "Checking whether the database needs to be restored from a backup"

TABLE_COUNT="$(
    psql "$DATABASE_URL" -Atqc \
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" \
        2>/dev/null || echo 0
)"

if [[ "${TABLE_COUNT:-0}" == "0" ]]; then
    LATEST_BACKUP="$(ls -t /workspace/backups/*.dump 2>/dev/null | head -n1 || true)"

    if [[ -n "$LATEST_BACKUP" ]]; then
        log "Empty database detected (likely a RunPod pod migration wiped the" \
            "container disk) - restoring latest backup: $LATEST_BACKUP"

        bash /app/scripts/db_restore.sh "$LATEST_BACKUP" ||
            fail "Automatic restore from $LATEST_BACKUP failed"
    else
        log "Empty database and no backup found on /workspace - starting fresh" \
            "(expected only on the very first deployment)"
    fi
fi

log "Preparing database schema"
cd /app
python -m scripts.bootstrap_database

mkdir -p /workspace/backups

backup_on_exit() {
    log "Received shutdown signal - creating a final database backup before exit"

    bash /app/scripts/db_backup.sh ||
        log "WARNING: final backup failed - the network volume may still have an" \
            "earlier periodic backup"

    if [[ -n "${UVICORN_PID:-}" ]]; then
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
    fi

    exit 0
}

trap backup_on_exit SIGTERM SIGINT

auto_backup_loop() {
    local interval="${AUTO_BACKUP_INTERVAL_SECONDS:-1800}"
    local keep="${AUTO_BACKUP_KEEP:-10}"

    while true; do
        sleep "$interval"

        bash /app/scripts/db_backup.sh ||
            log "WARNING: periodic backup failed"

        ls -t /workspace/backups/*.dump 2>/dev/null |
            tail -n "+$((keep + 1))" |
            xargs -r rm -f
    done
}

log "Starting FastAPI on port 8000"

python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips "$FORWARDED_ALLOW_IPS" \
    --timeout-keep-alive "$UVICORN_TIMEOUT_KEEP_ALIVE" \
    --ws-ping-interval "$UVICORN_WS_PING_INTERVAL" \
    --ws-ping-timeout "$UVICORN_WS_PING_TIMEOUT" \
    --timeout-graceful-shutdown "$UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN" &
UVICORN_PID=$!

auto_backup_loop &

wait "$UVICORN_PID"