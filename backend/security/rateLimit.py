import os
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

load_dotenv()


def user_or_ip_key(request: Request) -> str:
    user = getattr(request.state, "user", None)

    if user and getattr(user, "id", None):
        return f"user:{user.id}"

    client_ip = get_remote_address(request)
    return f"ip:{client_ip}"


ENV = os.getenv("ENV", "dev").lower()
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_LIMIT = os.getenv("RATE_LIMIT_DEFAULT", "200/minute")
IN_MEMORY_FALLBACK = ENV != "prod"
RATE_LIMIT_STORAGE_URI = (
    os.getenv("RATE_LIMIT_STORAGE_URI", "").strip()
    or (REDIS_URL if ENV == "prod" else "memory://")
)

if RATE_LIMIT_ENABLED and not DEFAULT_LIMIT:
    raise RuntimeError("RATE_LIMIT_DEFAULT missing")

if ENV == "prod" and RATE_LIMIT_ENABLED:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL missing in production")

    if REDIS_URL == "redis://localhost:6379/0":
        raise RuntimeError("Production REDIS_URL must not use localhost default")

limiter = Limiter(
    key_func=user_or_ip_key,
    default_limits=[DEFAULT_LIMIT],
    storage_uri=RATE_LIMIT_STORAGE_URI,
    headers_enabled=True,
    enabled=RATE_LIMIT_ENABLED,
    in_memory_fallback_enabled=IN_MEMORY_FALLBACK,
)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Please try again later.",
            "error": "rate_limit_exceeded",
        },
    )


def setup_rate_limiter(app):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)
