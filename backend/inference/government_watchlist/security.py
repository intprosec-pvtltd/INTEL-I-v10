import os
from cryptography.fernet import Fernet, InvalidToken


def _fernet():
    key = os.getenv("EXTERNAL_WATCHLIST_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("External watchlist credential encryption is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except Exception as exc:
        raise RuntimeError("External watchlist credential encryption key is invalid") from exc


def encrypt_credential(value: str | None) -> bytes | None:
    if value is None or value == "":
        return None
    return _fernet().encrypt(value.encode("utf-8"))


def decrypt_credential(value: bytes | None) -> str | None:
    if not value:
        return None
    try:
        return _fernet().decrypt(value).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError) as exc:
        raise RuntimeError("Stored external credential cannot be decrypted") from exc
