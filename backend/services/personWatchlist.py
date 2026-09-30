from __future__ import annotations

import logging

import cv2
import numpy as np

from fastapi import UploadFile
from sqlalchemy.orm import Session

from db.advanced_intelligence_model import PersonWatchlistEntry
from services.personRecognition import (
    VERSION,
    encrypt_embedding,
    encrypt_bytes,
    extract_single_embedding,
)


logger = logging.getLogger("person-watchlist")

MAX_IMAGE_SIZE = 10 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def validate_user_id(user_id: int) -> int:
    """Validate and normalize the authenticated record owner."""

    try:
        normalized_user_id = int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("A valid authenticated user ID is required.") from exc

    if normalized_user_id <= 0:
        raise ValueError("A valid authenticated user ID is required.")

    return normalized_user_id


async def read_reference_image(
    image: UploadFile,
) -> tuple[bytes, np.ndarray]:
    """Read and validate a watchlist reference image."""

    if image is None:
        raise ValueError("Reference face image is required.")

    content_type = str(image.content_type or "").split(";", 1)[0].strip().lower()

    if content_type and content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            "Unsupported image content type. Allowed: JPEG, PNG, WEBP."
        )

    data = await image.read()

    if not data:
        raise ValueError("Uploaded image is empty.")

    if len(data) > MAX_IMAGE_SIZE:
        raise ValueError("Image size must not exceed 10 MB.")

    buffer = np.frombuffer(data, dtype=np.uint8)

    if buffer.size == 0:
        raise ValueError("Uploaded image is empty.")

    frame = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Unable to decode uploaded image.")

    return data, frame


def normalize_embedding(
    embedding,
) -> np.ndarray:
    """Return a finite, normalized float32 face embedding."""

    if embedding is None:
        raise ValueError("Unable to generate face embedding.")

    vector = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    if vector.size == 0:
        raise ValueError("Generated face embedding is empty.")

    if not np.all(np.isfinite(vector)):
        raise ValueError("Generated face embedding contains invalid values.")

    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Generated face embedding is invalid.")

    normalized = vector / norm

    return normalized.astype(
        np.float32,
        copy=False,
    )


async def enroll_person_watchlist(
    db: Session,
    user_id: int,
    full_name: str,
    category: str,
    status: str,
    description: str,
    source: str,
    external_reference: str,
    image: UploadFile,
):
    """
    Enroll a person watchlist entry for the authenticated user.

    The caller must pass current_user.id. The function must never use a
    hard-coded user ID because the list endpoint is owner-filtered.
    """

    owner_user_id = validate_user_id(user_id)

    image_bytes, frame = await read_reference_image(
        image
    )

    embedding = extract_single_embedding(
        frame
    )

    embedding = normalize_embedding(
        embedding
    )

    encrypted_embedding = encrypt_embedding(
        embedding
    )

    encrypted_image = encrypt_bytes(
        image_bytes
    )

    content_type = (
        str(image.content_type or "")
        .split(";", 1)[0]
        .strip()
        .lower()
        or "application/octet-stream"
    )

    entry = PersonWatchlistEntry(
        user_id=owner_user_id,
        full_name=full_name,
        category=category,
        status=status,
        description=description or None,
        source=source or None,
        external_reference=external_reference or None,
        embedding_encrypted=encrypted_embedding,
        embedding_dimension=int(embedding.size),
        face_model="OpenCV FaceRecognizerSF",
        face_model_version=VERSION,
        reference_image_data=encrypted_image,
        reference_image_content_type=content_type,
        reference_image_filename=(
            image.filename or "reference.jpg"
        ),
        reference_image_size=len(image_bytes),
        metadata_json={
            "enrollment": {
                "image_filename": (
                    image.filename or "reference.jpg"
                ),
                "content_type": content_type,
                "image_size": len(image_bytes),
                "embedding_dtype": "float32",
                "embedding_dimension": int(
                    embedding.size
                ),
                "embedding_normalized": True,
            }
        },
    )

    try:
        db.add(entry)
        db.commit()
        db.refresh(entry)

    except Exception:
        db.rollback()

        logger.exception(
            "Failed to save person watchlist entry",
            extra={
                "user_id": owner_user_id,
            },
        )

        raise

    return {
        "id": entry.id,
        "user_id": entry.user_id,
        "full_name": entry.full_name,
        "category": entry.category,
        "status": entry.status,
        "description": entry.description,
        "source": entry.source,
        "external_reference": (
            entry.external_reference
        ),
        "embedding_dimension": (
            entry.embedding_dimension
        ),
        "face_model": entry.face_model,
        "face_model_version": (
            entry.face_model_version
        ),
        "reference_image_available": bool(
            entry.reference_image_data
        ),
        "reference_image_content_type": (
            entry.reference_image_content_type
        ),
        "reference_image_filename": (
            entry.reference_image_filename
        ),
        "reference_image_size": (
            entry.reference_image_size
        ),
        "created_at": (
            entry.created_at.isoformat()
            if entry.created_at
            else None
        ),
        "updated_at": (
            entry.updated_at.isoformat()
            if entry.updated_at
            else None
        ),
    }


async def replace_person_watchlist_image(
    db: Session,
    entry: PersonWatchlistEntry,
    image: UploadFile,
):
    """Replace the image and regenerate its corresponding embedding."""

    if entry is None:
        raise ValueError(
            "Person watchlist entry is required."
        )

    image_bytes, frame = await read_reference_image(
        image
    )

    embedding = extract_single_embedding(
        frame
    )

    embedding = normalize_embedding(
        embedding
    )

    encrypted_embedding = encrypt_embedding(
        embedding
    )

    encrypted_image = encrypt_bytes(
        image_bytes
    )

    content_type = (
        str(image.content_type or "")
        .split(";", 1)[0]
        .strip()
        .lower()
        or "application/octet-stream"
    )

    entry.embedding_encrypted = (
        encrypted_embedding
    )

    entry.embedding_dimension = int(
        embedding.size
    )

    entry.face_model = (
        "OpenCV FaceRecognizerSF"
    )

    entry.face_model_version = VERSION

    entry.reference_image_data = (
        encrypted_image
    )

    entry.reference_image_content_type = (
        content_type
    )

    entry.reference_image_filename = (
        image.filename or "reference.jpg"
    )

    entry.reference_image_size = len(
        image_bytes
    )

    metadata = dict(
        entry.metadata_json or {}
    )

    enrollment = dict(
        metadata.get("enrollment") or {}
    )

    enrollment.update(
        {
            "image_filename": (
                image.filename or "reference.jpg"
            ),
            "content_type": content_type,
            "image_size": len(image_bytes),
            "embedding_dtype": "float32",
            "embedding_dimension": int(
                embedding.size
            ),
            "embedding_normalized": True,
            "image_replaced": True,
        }
    )

    metadata["enrollment"] = enrollment

    entry.metadata_json = metadata

    try:
        db.commit()
        db.refresh(entry)

    except Exception:
        db.rollback()

        logger.exception(
            "Failed to replace person reference image",
            extra={
                "person_watchlist_id": (
                    getattr(entry, "id", None)
                ),
                "user_id": (
                    getattr(entry, "user_id", None)
                ),
            },
        )

        raise

    return {
        "id": entry.id,
        "user_id": entry.user_id,
        "full_name": entry.full_name,
        "reference_image_available": True,
        "reference_image_content_type": (
            entry.reference_image_content_type
        ),
        "reference_image_filename": (
            entry.reference_image_filename
        ),
        "reference_image_size": (
            entry.reference_image_size
        ),
        "embedding_dimension": (
            entry.embedding_dimension
        ),
        "face_model": entry.face_model,
        "face_model_version": (
            entry.face_model_version
        ),
        "updated_at": (
            entry.updated_at.isoformat()
            if entry.updated_at
            else None
        ),
    }