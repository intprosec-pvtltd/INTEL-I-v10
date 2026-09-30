import os
from cryptography.fernet import Fernet, InvalidToken


ENV_KEY_NAME = "CAMERA_SOURCE_ENCRYPTION_KEY"


def _get_fernet() -> Fernet:
    """
    Load the camera-source encryption key from the environment.

    The key must never be hard-coded in source code.
    """

    key = os.getenv(ENV_KEY_NAME)

    if not key:
        raise RuntimeError(
            f"{ENV_KEY_NAME} is not configured"
        )

    try:
        return Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Invalid {ENV_KEY_NAME}"
        ) from exc


def encrypt_camera_source(source: str) -> str:
    """
    Encrypt an RTSP/HTTP camera source.

    The plaintext source is never logged.
    """

    if source is None:
        raise ValueError(
            "Camera source cannot be None"
        )

    value = str(source).strip()

    if not value:
        raise ValueError(
            "Camera source cannot be empty"
        )

    if len(value) > 4096:
        raise ValueError(
            "Camera source is too long"
        )

    fernet = _get_fernet()

    encrypted = fernet.encrypt(
        value.encode("utf-8")
    )

    return encrypted.decode("utf-8")


def decrypt_camera_source(
    encrypted_source: str,
) -> str:
    """
    Decrypt a previously encrypted camera source.

    Only backend camera-worker code should call this.
    """

    if not encrypted_source:
        raise ValueError(
            "Encrypted camera source is empty"
        )

    fernet = _get_fernet()

    try:
        decrypted = fernet.decrypt(
            encrypted_source.encode("utf-8")
        )

    except InvalidToken as exc:
        raise ValueError(
            "Unable to decrypt camera source"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            "Camera source decryption failed"
        ) from exc

    source = decrypted.decode("utf-8").strip()

    if not source:
        raise ValueError(
            "Decrypted camera source is empty"
        )

    return source