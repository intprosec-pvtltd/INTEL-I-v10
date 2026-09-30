import os
import json
import logging
from typing import Any

from dotenv import load_dotenv
from redis import Redis
from redis.exceptions import RedisError

load_dotenv()

logger = logging.getLogger(__name__)

ENV = os.getenv("ENV", "dev").lower()
IS_PROD = ENV == "prod"

REDIS_URL = os.getenv("REDIS_URL")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "cctv").strip() or "cctv"

redis_client: Redis | None = None


def _fail_or_default(action: str, error: Exception, default):
    logger.exception("Redis %s failed", action)

    if IS_PROD:
        raise RuntimeError(f"Redis {action} failed") from error

    return default


if REDIS_ENABLED:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required when REDIS_ENABLED=true")
    try:
        redis_client = Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            health_check_interval=30,
            retry_on_timeout=True,
        )

        redis_client.ping()

    except RedisError as e:
        logger.exception("Redis client init failed")

        if IS_PROD:
            raise RuntimeError("Redis client init failed") from e

        redis_client = None


def make_key(key: str) -> str:
    clean_key = str(key).strip().lstrip(":")

    if not clean_key:
        raise ValueError("Redis key cannot be empty")

    return f"{REDIS_PREFIX}:{clean_key}"


def redis_ping() -> bool:
    if not REDIS_ENABLED or redis_client is None:
        if IS_PROD:
            logger.error("Redis ping failed because Redis is disabled or unavailable")
        return False
    try:
        return bool(redis_client.ping())
    except RedisError as e:
        return _fail_or_default("ping", e, False)


def set_json(key: str, value: dict[str, Any], ttl: int | None = None) -> bool:
    if not REDIS_ENABLED or redis_client is None:
        if IS_PROD:
            raise RuntimeError("Redis set_json failed because Redis is unavailable")
        return False

    try:
        real_key = make_key(key)

        redis_client.set(
            real_key,
            json.dumps(value, default=str),
            ex=ttl if ttl and ttl > 0 else None,
        )

        return True

    except (RedisError, TypeError, ValueError) as e:
        return _fail_or_default("set_json", e, False)


def get_json(key: str) -> dict[str, Any] | None:
    if not REDIS_ENABLED or redis_client is None:
        if IS_PROD:
            raise RuntimeError("Redis get_json failed because Redis is unavailable")
        return None

    try:
        data = redis_client.get(make_key(key))

        if not data:
            return None

        parsed = json.loads(data)

        if not isinstance(parsed, dict):
            logger.warning("Redis get_json ignored non-dict value | key=%s", key)
            return None

        return parsed

    except (RedisError, json.JSONDecodeError, TypeError, ValueError) as e:
        return _fail_or_default("get_json", e, None)


def delete_key(key: str) -> bool:
    if not REDIS_ENABLED or redis_client is None:
        if IS_PROD:
            raise RuntimeError("Redis delete_key failed because Redis is unavailable")
        return False

    try:
        redis_client.delete(make_key(key))
        return True

    except (RedisError, ValueError) as e:
        return _fail_or_default("delete_key", e, False)


def incr_key(key: str, ttl: int | None = None) -> int:
    if not REDIS_ENABLED or redis_client is None:
        if IS_PROD:
            raise RuntimeError("Redis incr_key failed because Redis is unavailable")
        return 0

    try:
        real_key = make_key(key)
        value = redis_client.incr(real_key)

        if ttl and ttl > 0:
            redis_client.expire(real_key, ttl)

        return int(value)

    except (RedisError, ValueError) as e:
        return _fail_or_default("incr_key", e, 0)


def scan_keys(pattern: str, count: int = 100) -> list[str]:

    if not REDIS_ENABLED or redis_client is None:
        if IS_PROD:
            raise RuntimeError("Redis scan_keys failed because Redis is unavailable")
        return []

    try:
        safe_count = max(10, min(int(count), 1000))
        real_pattern = make_key(pattern)

        keys: list[str] = []

        prefix = f"{REDIS_PREFIX}:"

        for key in redis_client.scan_iter(match=real_pattern, count=safe_count):
            if key.startswith(prefix):
                key = key[len(prefix):]

            keys.append(key)

        return keys

    except (RedisError, ValueError) as e:
        return _fail_or_default("scan_keys", e, [])


def delete_by_pattern(pattern: str, count: int = 100) -> int:
    if not REDIS_ENABLED or redis_client is None:
        if IS_PROD:
            raise RuntimeError("Redis delete_by_pattern failed because Redis is unavailable")
        return 0

    try:
        safe_count = max(10, min(int(count), 1000))
        real_pattern = make_key(pattern)

        deleted = 0

        for key in redis_client.scan_iter(match=real_pattern, count=safe_count):
            deleted += int(redis_client.delete(key))

        return deleted

    except (RedisError, ValueError) as e:
        return _fail_or_default("delete_by_pattern", e, 0)
