import os
import uuid
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import (
    HTTPException,
    Depends,
    status,
    WebSocket,
    Request,
    WebSocketException,
)
from sqlalchemy.orm import Session

from db.database import getDB
from db.model import User


load_dotenv()


# ============================================================
# JWT KEY LOADING
# ============================================================

def _load_pem_key(
    path_env_name: str,
    raw_env_name: str,
    default_path: str,
    key_name: str,
) -> str:
    key_path_raw = os.getenv(path_env_name)

    if key_path_raw and key_path_raw.strip():
        key_path = Path(key_path_raw.strip())

        if not key_path.is_absolute():
            key_path = Path.cwd() / key_path

        if not key_path.is_file():
            raise RuntimeError(
                f"{key_name} file not found: {key_path}"
            )

        try:
            key = key_path.read_text(
                encoding="utf-8"
            ).strip()
        except OSError as exc:
            raise RuntimeError(
                f"Unable to read {key_name} file: {key_path}"
            ) from exc
    else:
        key_raw = os.getenv(raw_env_name)

        if not key_raw:
            raise RuntimeError(
                f"{path_env_name} or {raw_env_name} "
                f"must be configured"
            )

        key = key_raw.strip()

        if (
            len(key) >= 2
            and key[0] == '"'
            and key[-1] == '"'
        ):
            key = key[1:-1].strip()

        # Convert literal \n into real newlines.
        key = key.replace("\\n", "\n")

        # Normalize multiline indentation.
        lines = [
            line.strip()
            for line in key.splitlines()
            if line.strip()
        ]

        key = "\n".join(lines).strip()

    # --------------------------------------------------------
    # 3. Validate PEM structure
    # --------------------------------------------------------
    if key_name == "PRIVATE_KEY":
        begin_marker = "-----BEGIN PRIVATE KEY-----"
        end_marker = "-----END PRIVATE KEY-----"

        # Also allow PKCS#1 RSA private key.
        rsa_begin_marker = "-----BEGIN RSA PRIVATE KEY-----"
        rsa_end_marker = "-----END RSA PRIVATE KEY-----"

        valid = (
            (
                begin_marker in key
                and end_marker in key
            )
            or
            (
                rsa_begin_marker in key
                and rsa_end_marker in key
            )
        )

    else:
        begin_marker = "-----BEGIN PUBLIC KEY-----"
        end_marker = "-----END PUBLIC KEY-----"

        # Also allow RSA PUBLIC KEY format.
        rsa_begin_marker = "-----BEGIN RSA PUBLIC KEY-----"
        rsa_end_marker = "-----END RSA PUBLIC KEY-----"

        valid = (
            (
                begin_marker in key
                and end_marker in key
            )
            or
            (
                rsa_begin_marker in key
                and rsa_end_marker in key
            )
        )

    if not valid:
        raise RuntimeError(
            f"Invalid {key_name} format. "
            f"Expected a valid RSA PEM key."
        )

    return key


PRIVATE_KEY = _load_pem_key(
    path_env_name="PRIVATE_KEY_PATH",
    raw_env_name="PRIVATE_KEY",
    default_path="keys/private_key.pem",
    key_name="PRIVATE_KEY",
)

PUBLIC_KEY = _load_pem_key(
    path_env_name="PUBLIC_KEY_PATH",
    raw_env_name="PUBLIC_KEY",
    default_path="keys/public_key.pem",
    key_name="PUBLIC_KEY",
)


# ============================================================
# JWT CONFIGURATION
# ============================================================

ALGORITHM = os.getenv(
    "ALGORITHM",
    "RS256",
).strip().upper()

if ALGORITHM != "RS256":
    raise RuntimeError(
        "Only RS256 is allowed in production"
    )


JWT_ISSUER = os.getenv(
    "JWT_ISSUER",
    "intel-i-backend",
).strip()

JWT_AUDIENCE = os.getenv(
    "JWT_AUDIENCE",
    "intel-i-frontend",
).strip()


if not JWT_ISSUER:
    raise RuntimeError(
        "JWT_ISSUER missing in .env"
    )

if not JWT_AUDIENCE:
    raise RuntimeError(
        "JWT_AUDIENCE missing in .env"
    )


try:
    ACCESS_TOKEN_EXPIRE_HOURS = int(
        os.getenv(
            "ACCESS_TOKEN_EXPIRE_HOURS",
            "24",
        )
    )

    REFRESH_TOKEN_EXPIRE_DAYS = int(
        os.getenv(
            "REFRESH_TOKEN_EXPIRE_DAYS",
            "30",
        )
    )

except ValueError as exc:
    raise RuntimeError(
        "ACCESS_TOKEN_EXPIRE_HOURS and "
        "REFRESH_TOKEN_EXPIRE_DAYS must be valid integers"
    ) from exc


if (
    ACCESS_TOKEN_EXPIRE_HOURS <= 0
    or ACCESS_TOKEN_EXPIRE_HOURS > 24
):
    raise RuntimeError(
        "ACCESS_TOKEN_EXPIRE_HOURS must be between 1 and 24"
    )


if (
    REFRESH_TOKEN_EXPIRE_DAYS <= 0
    or REFRESH_TOKEN_EXPIRE_DAYS > 30
):
    raise RuntimeError(
        "REFRESH_TOKEN_EXPIRE_DAYS must be between 1 and 30"
    )


# ============================================================
# PASSWORD HASHING
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    if not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is required",
        )

    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    if not plain_password or not hashed_password:
        return False

    try:
        return pwd_context.verify(
            plain_password,
            hashed_password,
        )
    except Exception:
        return False


# ============================================================
# ACCESS TOKEN
# ============================================================

def create_access_token(
    data: Dict[str, Any],
) -> str:

    subject = data.get("sub")

    if not subject:
        raise ValueError(
            "Access token requires subject"
        )

    now = datetime.utcnow()

    expire = now + timedelta(
        hours=ACCESS_TOKEN_EXPIRE_HOURS
    )

    payload = data.copy()

    payload.update(
        {
            "jti": str(uuid.uuid4()),
            "sub": str(subject),
            "type": "access",
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
            "iat": now,
            "exp": expire,
        }
    )

    return jwt.encode(
        payload,
        PRIVATE_KEY,
        algorithm=ALGORITHM,
    )


# ============================================================
# REFRESH TOKEN
# ============================================================

def create_refresh_token(
    data: Dict[str, Any],
) -> Tuple[str, str, datetime]:

    subject = data.get("sub")

    if not subject:
        raise ValueError(
            "Refresh token requires subject"
        )

    now = datetime.utcnow()

    expire = now + timedelta(
        days=REFRESH_TOKEN_EXPIRE_DAYS
    )

    token_jti = str(uuid.uuid4())

    payload = data.copy()

    payload.update(
        {
            "jti": token_jti,
            "sub": str(subject),
            "type": "refresh",
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
            "iat": now,
            "exp": expire,
        }
    )

    token = jwt.encode(
        payload,
        PRIVATE_KEY,
        algorithm=ALGORITHM,
    )

    return (
        token,
        token_jti,
        expire,
    )


# ============================================================
# TOKEN DECODING
# ============================================================

def decode_token(
    token: str,
) -> Dict[str, Any]:

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing",
        )

    try:
        return jwt.decode(
            token,
            PUBLIC_KEY,
            algorithms=[ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ============================================================
# CURRENT USER
# ============================================================

def get_current_user(
    request: Request,
    db: Session = Depends(getDB),
) -> User:

    token = request.cookies.get(
        "access_token"
    )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    payload = decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
        )

    subject = payload.get("sub")

    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        )

    try:
        user_id = int(subject)

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account disabled",
        )

    if int(payload.get("ver", -1)) != int(user.token_version or 0):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session revoked")

    from security.rbac import enforce_request_permission
    enforce_request_permission(user, request.method, request.url.path)

    request.state.user = user

    return user

# ============================================================
# CURRENT USER - WEBSOCKET
# ============================================================

async def get_current_user_ws(
    websocket: WebSocket,
) -> User:

    # Cookie-authenticated WebSockets require an explicit Origin check to
    # prevent cross-site WebSocket hijacking. Non-browser clients may omit the
    # header only outside production.
    configured_origins = {
        value.strip().rstrip("/")
        for value in (
            os.getenv("CORS_ALLOWED_ORIGINS", "")
            + ","
            + os.getenv("FRONTEND_URL", "")
        ).split(",")
        if value.strip()
    }
    origin = str(websocket.headers.get("origin") or "").strip().rstrip("/")
    is_production = os.getenv("ENV", "dev").strip().lower() in {
        "prod", "production"
    }
    if (origin and origin not in configured_origins) or (
        is_production and not origin
    ):
        await websocket.close(code=1008)
        raise WebSocketException(code=1008)

    token = websocket.cookies.get(
        "access_token"
    )

    if not token:
        await websocket.close(
            code=1008
        )
        raise WebSocketException(
            code=1008
        )

    try:
        payload = decode_token(token)

    except Exception:
        await websocket.close(
            code=1008
        )
        raise WebSocketException(
            code=1008
        )

    if payload.get("type") != "access":
        await websocket.close(
            code=1008
        )
        raise WebSocketException(
            code=1008
        )

    subject = payload.get("sub")

    if not subject:
        await websocket.close(
            code=1008
        )
        raise WebSocketException(
            code=1008
        )

    try:
        user_id = int(subject)

    except (ValueError, TypeError):
        await websocket.close(
            code=1008
        )
        raise WebSocketException(
            code=1008
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # WebSockets are long-lived.
    # Never keep a SQLAlchemy session open for the entire
    # WebSocket lifetime.
    # --------------------------------------------------------

    from db.database import SessionLocal

    db = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(User.id == user_id)
            .first()
        )

        if not user or not user.is_active:
            await websocket.close(
                code=1008
            )
            raise WebSocketException(
                code=1008
            )

        # Extract only what the WebSocket actually needs
        # before closing/detaching the ORM session.
        if int(payload.get("ver", -1)) != int(user.token_version or 0):
            await websocket.close(code=1008)
            raise WebSocketException(code=1008)
        authenticated_user_id = user.id

    finally:
        db.close()

    websocket.state.user_id = authenticated_user_id

    return authenticated_user_id

# ============================================================
# PASSWORD STRENGTH
# ============================================================

COMMON_PASSWORDS = {
    "password",
    "password123",
    "password123!",
    "admin123456",
    "qwerty123456",
    "welcome12345",
}


def validate_password_strength(
    password: str,
):

    if len(password) < 12:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 12 characters",
        )

    if not re.search(
        r"[A-Z]",
        password,
    ):
        raise HTTPException(
            status_code=400,
            detail="Password must contain uppercase letter",
        )

    if not re.search(
        r"[a-z]",
        password,
    ):
        raise HTTPException(
            status_code=400,
            detail="Password must contain lowercase letter",
        )

    if not re.search(
        r"\d",
        password,
    ):
        raise HTTPException(
            status_code=400,
            detail="Password must contain number",
        )

    if not re.search(
        r"[^\w\s]",
        password,
    ):
        raise HTTPException(
            status_code=400,
            detail="Password must contain special character",
        )

    if password.lower() in COMMON_PASSWORDS:
        raise HTTPException(
            status_code=400,
            detail="Password is too common",
        )

    return True
