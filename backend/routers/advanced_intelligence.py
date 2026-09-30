from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from sqlalchemy.orm import Session

from db.database import get_db

from db.advanced_intelligence_model import (
    IncidentEvidence,
    PersonWatchlistEntry,
)
from db.model import Incident, User, Snapshot, Alert
from auth.auth import get_current_user

from services.personWatchlist import (
    enroll_person_watchlist,
    replace_person_watchlist_image,
)
from services.personRecognition import decrypt_bytes


logger = logging.getLogger(
    "advanced-intelligence"
)


router = APIRouter(
    prefix="/api/intelligence",
    tags=["Person Watchlist"],
)


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_CATEGORIES = {
    "MISSING",
    "WANTED",
    "OTHER",
}

ALLOWED_STATUSES = {
    "ACTIVE",
    "CLEARED",
    "EXPIRED",
    "DISABLED",
}

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_NAME_LENGTH = 150
MAX_DESCRIPTION_LENGTH = 5000
MAX_SOURCE_LENGTH = 150
MAX_REFERENCE_LENGTH = 150
MAX_IMAGE_SIZE = 10 * 1024 * 1024


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_person_watchlist_entry(
    entry: PersonWatchlistEntry,
) -> dict:

    if entry is None:
        return None

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

        "metadata": (
            entry.metadata_json
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

        "reference_image_url": (
            f"/api/intelligence/person-watchlist/{entry.id}/image"
            if entry.reference_image_data
            else None
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


# ============================================================
# IMAGE VALIDATION
# ============================================================

async def validate_face_image(
    image: UploadFile,
) -> bytes:

    if image is None:

        raise HTTPException(
            status_code=422,
            detail=(
                "Reference face image is required."
            ),
        )

    filename = (
        image.filename or ""
    ).strip().lower()

    if not filename:

        raise HTTPException(
            status_code=422,
            detail="Image filename is required.",
        )

    if not any(
        filename.endswith(
            extension
        )
        for extension
        in ALLOWED_IMAGE_EXTENSIONS
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid image format. "
                "Allowed formats: JPG, JPEG, PNG, WEBP."
            ),
        )

    if (
        image.content_type
        and image.content_type
        not in ALLOWED_IMAGE_TYPES
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported image content type. "
                "Allowed types: JPEG, PNG, WEBP."
            ),
        )

    data = await image.read()

    if not data:

        raise HTTPException(
            status_code=422,
            detail="Uploaded image is empty.",
        )

    if len(data) > MAX_IMAGE_SIZE:

        raise HTTPException(
            status_code=422,
            detail=(
                "Image size must not exceed 10 MB."
            ),
        )

    # Reset file pointer so the service can read it again.
    try:
        await image.seek(0)
    except Exception:
        pass

    return data


# ============================================================
# GET WATCHLIST
# ============================================================

@router.get(
    "/person-watchlist"
)
def get_person_watchlist(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    user_id = current_user.id

    try:

        entries = (
            db.query(
                PersonWatchlistEntry
            )
            .filter(
                PersonWatchlistEntry.user_id
                == user_id
            )
            .order_by(
                PersonWatchlistEntry.updated_at.desc()
            )
            .limit(1000)
            .all()
        )

        return {
            "success": True,

            "count": len(entries),

            "entries": [
                serialize_person_watchlist_entry(
                    entry
                )
                for entry in entries
            ],
        }

    except Exception as exc:

        logger.exception(
            "Failed to load person watchlist"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load person watchlist."
            ),
        ) from exc


# ============================================================
# ENROLL PERSON
# ============================================================

@router.post(
    "/person-watchlist/enroll"
)
async def enroll_person(

    full_name: str = Form(...),

    category: str = Form(
        "OTHER"
    ),

    status: str = Form(
        "ACTIVE"
    ),

    description: str = Form(
        ""
    ),

    source: str = Form(
        ""
    ),

    external_reference: str = Form(
        ""
    ),

    image: UploadFile = File(...),

    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    clean_full_name = (
        full_name.strip()
        if full_name
        else ""
    )

    if not clean_full_name:

        raise HTTPException(
            status_code=422,
            detail="Full name is required.",
        )

    if len(clean_full_name) > MAX_NAME_LENGTH:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Full name must not exceed "
                f"{MAX_NAME_LENGTH} characters."
            ),
        )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    clean_category = (
        category.strip().upper()
        if category
        else "OTHER"
    )

    if (
        clean_category
        not in ALLOWED_CATEGORIES
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid category. "
                "Allowed values: "
                "MISSING, WANTED, OTHER."
            ),
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    clean_status = (
        status.strip().upper()
        if status
        else "ACTIVE"
    )

    if (
        clean_status
        not in ALLOWED_STATUSES
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid status. "
                "Allowed values: "
                "ACTIVE, CLEARED, EXPIRED, DISABLED."
            ),
        )

    # --------------------------------------------------------
    # OPTIONAL FIELDS
    # --------------------------------------------------------

    clean_description = (
        description.strip()
        if description
        else ""
    )

    clean_source = (
        source.strip()
        if source
        else ""
    )

    clean_external_reference = (
        external_reference.strip()
        if external_reference
        else ""
    )

    if (
        len(clean_description)
        > MAX_DESCRIPTION_LENGTH
    ):

        raise HTTPException(
            status_code=422,
            detail="Description is too long.",
        )

    if (
        len(clean_source)
        > MAX_SOURCE_LENGTH
    ):

        raise HTTPException(
            status_code=422,
            detail="Source is too long.",
        )

    if (
        len(clean_external_reference)
        > MAX_REFERENCE_LENGTH
    ):

        raise HTTPException(
            status_code=422,
            detail="Reference ID is too long.",
        )

    await validate_face_image(
        image
    )

    try:

        person = await enroll_person_watchlist(
            db=db,
            user_id=current_user.id,
            full_name=clean_full_name,
            category=clean_category,
            status=clean_status,
            description=clean_description,
            source=clean_source,
            external_reference=clean_external_reference,
            image=image,
        )

        return {
            "success": True,

            "message": (
                "Person enrolled successfully."
            ),

            "person": person,
        }

    except ValueError as exc:

        db.rollback()

        logger.warning(
            "Person enrollment validation error: %s",
            exc,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except HTTPException:

        db.rollback()

        raise

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Person enrollment failed"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to enroll person."
            ),
        ) from exc


# ============================================================
# GET REFERENCE IMAGE
# ============================================================

@router.get(
    "/person-watchlist/{person_id}/image"
)
def get_person_watchlist_image(
    person_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return the protected reference image for one watchlist
    entry. The image is never included in the list endpoint.
    """
    from fastapi.responses import Response

    user_id = current_user.id

    entry = (
        db.query(PersonWatchlistEntry)
        .filter(
            PersonWatchlistEntry.id == person_id,
            PersonWatchlistEntry.user_id == user_id,
        )
        .first()
    )

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Person watchlist entry not found.",
        )

    if not entry.reference_image_data:
        raise HTTPException(
            status_code=404,
            detail="Reference image is not available for this person.",
        )

    content_type = (
        entry.reference_image_content_type
        or "application/octet-stream"
    )

    try:
        image_data = decrypt_bytes(entry.reference_image_data)
    except Exception as exc:
        logger.exception("Unable to decrypt person reference image")
        raise HTTPException(
            status_code=500,
            detail="Unable to load reference image.",
        ) from exc

    response = Response(
        content=image_data,
        media_type=content_type,
    )

    response.headers["Content-Disposition"] = "inline"
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


# ============================================================
# REPLACE REFERENCE IMAGE
# ============================================================

@router.post(
    "/person-watchlist/{person_id}/image"
)
async def replace_person_watchlist_reference_image(
    person_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload/replace the reference image for an existing person.

    The face embedding is regenerated from the new image so the
    stored photo and recognition embedding always stay aligned.
    """

    user_id = current_user.id

    entry = (
        db.query(PersonWatchlistEntry)
        .filter(
            PersonWatchlistEntry.id == person_id,
            PersonWatchlistEntry.user_id == user_id,
        )
        .first()
    )

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Person watchlist entry not found.",
        )

    try:
        await validate_face_image(image)
        result = await replace_person_watchlist_image(
            db=db,
            entry=entry,
            image=image,
        )
        return {
            "success": True,
            "message": "Reference image updated successfully.",
            "person": result,
        }
    except ValueError as exc:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to replace person reference image")
        raise HTTPException(
            status_code=400,
            detail="Failed to update reference image.",
        ) from exc


# ============================================================
# UPDATE PERSON
# ============================================================

@router.patch(
    "/person-watchlist/{person_id}"
)
def update_person_watchlist(

    person_id: int,

    full_name: Optional[str] = Form(
        None
    ),

    category: Optional[str] = Form(
        None
    ),

    status: Optional[str] = Form(
        None
    ),

    description: Optional[str] = Form(
        None
    ),

    source: Optional[str] = Form(
        None
    ),

    external_reference: Optional[str] = Form(
        None
    ),

    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    user_id = current_user.id

    entry = (
        db.query(
            PersonWatchlistEntry
        )
        .filter(
            PersonWatchlistEntry.id
            == person_id,

            PersonWatchlistEntry.user_id
            == user_id,
        )
        .first()
    )

    if entry is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Person watchlist entry not found."
            ),
        )

    if full_name is not None:

        clean_full_name = full_name.strip()

        if not clean_full_name:
            raise HTTPException(
                status_code=422,
                detail="Full name cannot be empty.",
            )

        if len(clean_full_name) > MAX_NAME_LENGTH:
            raise HTTPException(
                status_code=422,
                detail="Full name is too long.",
            )

        entry.full_name = clean_full_name

    if category is not None:

        clean_category = (
            category.strip().upper()
        )

        if (
            clean_category
            not in ALLOWED_CATEGORIES
        ):

            raise HTTPException(
                status_code=422,
                detail="Invalid category.",
            )

        entry.category = clean_category

    if status is not None:

        clean_status = (
            status.strip().upper()
        )

        if (
            clean_status
            not in ALLOWED_STATUSES
        ):

            raise HTTPException(
                status_code=422,
                detail="Invalid status.",
            )

        entry.status = clean_status

    if description is not None:

        entry.description = (
            description.strip()
        )

    if source is not None:

        entry.source = (
            source.strip()
        )

    if external_reference is not None:

        entry.external_reference = (
            external_reference.strip()
        )

    try:

        db.commit()

        db.refresh(entry)

        return {
            "success": True,

            "message": (
                "Person watchlist updated."
            ),

            "person": (
                serialize_person_watchlist_entry(
                    entry
                )
            ),
        }

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Failed updating person watchlist"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to update person watchlist."
            ),
        ) from exc


# ============================================================
# DISABLE PERSON
# ============================================================

@router.delete(
    "/person-watchlist/{person_id}"
)
def delete_person_watchlist(

    person_id: int,

    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    user_id = current_user.id

    entry = (
        db.query(
            PersonWatchlistEntry
        )
        .filter(
            PersonWatchlistEntry.id
            == person_id,

            PersonWatchlistEntry.user_id
            == user_id,
        )
        .first()
    )

    if entry is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Person watchlist entry not found."
            ),
        )

    try:

        entry.status = "DISABLED"

        db.commit()

        return {
            "success": True,

            "message": (
                "Person watchlist entry disabled."
            ),

            "id": person_id,
        }

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Failed disabling person watchlist entry"
        )

        raise HTTPException(
            status_code=400,
            detail="Failed to disable person.",
        ) from exc

# ============================================================
# INCIDENT / EVIDENCE API
# ============================================================

def serialize_incident_evidence(evidence: IncidentEvidence) -> dict:
    return {
        "id": evidence.id,
        "incident_id": evidence.incident_id,
        "user_id": evidence.user_id,
        "alert_id": evidence.alert_id,
        "snapshot_id": evidence.snapshot_id,
        "snapshot_path": f"/snapshot/{evidence.snapshot_id}" if evidence.snapshot_id else None,
        "evidence_type": evidence.evidence_type,
        "object_type": evidence.object_type,
        "object_reference": evidence.object_reference,
        "description": evidence.description,
        "metadata": evidence.metadata_json,
        "created_at": (
            evidence.created_at.isoformat()
            if evidence.created_at
            else None
        ),
    }


def serialize_incident(incident: Incident) -> dict:
    evidence_items = getattr(incident, "evidence_items", None) or []
    description = (
        incident.evaluation_reason
        or incident.evidence
        or ""
    )
    severity = (
        incident.evaluation_severity
        or "INFO"
    )
    # The primary incident snapshot must remain the camera detection.  Person
    # watchlist incidents can also contain an enrolled reference image.
    first_snapshot_id = next(
        (
            item.snapshot_id
            for item in evidence_items
            if getattr(item, "snapshot_id", None)
            and str(getattr(item, "evidence_type", "") or "").upper()
            != "WATCHLIST_REFERENCE"
        ),
        None,
    )
    if first_snapshot_id is None:
        first_snapshot_id = next(
            (
                item.snapshot_id
                for item in evidence_items
                if getattr(item, "snapshot_id", None)
            ),
            None,
        )
    return {
        "id": incident.id,
        "user_id": incident.user_id,
        "cam_id": incident.cam_id,
        "primary_track_id": incident.primary_track_id,
        "global_vehicle_id": getattr(incident, "global_vehicle_id", None),
        "incident_type": incident.incident_type,
        "status": incident.status,
        "evidence": [serialize_incident_evidence(x) for x in evidence_items],
        "snapshot_id": first_snapshot_id,
        "snapshot_path": f"/snapshot/{first_snapshot_id}" if first_snapshot_id else None,
        "evidence_text": incident.evidence,
        "description": description,
        "severity": severity,
        "started_at": incident.started_at.isoformat() if incident.started_at else None,
        "ended_at": incident.ended_at.isoformat() if incident.ended_at else None,
        "evaluation_status": incident.evaluation_status,
        "evaluation_score": incident.evaluation_score,
        "evaluation_confidence": incident.evaluation_confidence,
        "evaluation_severity": incident.evaluation_severity,
        "evaluation_reason": incident.evaluation_reason,
        "evaluation_evidence": incident.evaluation_evidence,
        "evaluated_at": incident.evaluated_at.isoformat() if incident.evaluated_at else None,
        "evaluated_by": incident.evaluated_by,
    }


@router.get("/incidents")
def get_incidents(
    limit: int = Query(200, ge=1, le=1000),
    status: Optional[str] = Query(None),
    incident_type: Optional[str] = Query(None),
    cam_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id
    query = db.query(Incident).filter(Incident.user_id == user_id)

    if status:
        query = query.filter(Incident.status == status.strip().upper())
    if incident_type:
        query = query.filter(Incident.incident_type == incident_type.strip())
    if cam_id:
        query = query.filter(Incident.cam_id == cam_id.strip())

    incidents = (
        query
        .order_by(Incident.started_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "success": True,
        "count": len(incidents),
        "incidents": [serialize_incident(x) for x in incidents],
    }


@router.get("/incidents/{incident_id}")
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id
    incident = (
        db.query(Incident)
        .filter(
            Incident.id == incident_id,
            Incident.user_id == user_id,
        )
        .first()
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return {"success": True, "incident": serialize_incident(incident)}


@router.post("/incidents")
def create_incident(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id

    incident_type = str(
        payload.get("incident_type") or payload.get("type") or "GENERAL"
    ).strip()
    if not incident_type:
        raise HTTPException(status_code=422, detail="incident_type is required.")

    allowed_statuses = {"OPEN", "CLOSED", "RESOLVED", "ESCALATED", "UNDER_REVIEW"}
    status = str(payload.get("status") or "OPEN").strip().upper()
    if status not in allowed_statuses:
        raise HTTPException(status_code=422, detail="Invalid incident status.")

    incident = Incident(
        user_id=user_id,
        cam_id=(str(payload["cam_id"]).strip() if payload.get("cam_id") is not None else None),
        primary_track_id=(str(payload["primary_track_id"]).strip() if payload.get("primary_track_id") is not None else None),
        global_vehicle_id=(str(payload["global_vehicle_id"]).strip() if payload.get("global_vehicle_id") is not None else None),
        incident_type=incident_type,
        status=status,
        evidence=(str(payload["evidence"]) if payload.get("evidence") is not None else None),
        evaluation_status=str(payload.get("evaluation_status") or "PENDING"),
        evaluation_score=(float(payload["evaluation_score"]) if payload.get("evaluation_score") is not None else None),
        evaluation_confidence=(float(payload["evaluation_confidence"]) if payload.get("evaluation_confidence") is not None else None),
        evaluation_severity=(str(payload["evaluation_severity"]) if payload.get("evaluation_severity") is not None else None),
        evaluation_reason=(str(payload["evaluation_reason"]) if payload.get("evaluation_reason") is not None else None),
        evaluation_evidence=payload.get("evaluation_evidence"),
    )

    try:
        db.add(incident)
        db.flush()

        evidence_payloads = payload.get("evidence_items") or payload.get("evidence_records") or []
        if isinstance(evidence_payloads, list):
            for item in evidence_payloads[:100]:
                if not isinstance(item, dict):
                    continue
                snapshot_id = (
                    int(item["snapshot_id"])
                    if item.get("snapshot_id") is not None
                    else None
                )
                alert_id = (
                    int(item["alert_id"])
                    if item.get("alert_id") is not None
                    else None
                )
                if snapshot_id is not None:
                    owned_snapshot = (
                        db.query(Snapshot)
                        .join(Alert, Alert.id == Snapshot.alert_id)
                        .filter(
                            Snapshot.id == snapshot_id,
                            Alert.user_id == user_id,
                        )
                        .first()
                    )
                    if owned_snapshot is None:
                        raise HTTPException(
                            status_code=403,
                            detail="Snapshot does not belong to the authenticated user.",
                        )

                evidence = IncidentEvidence(
                    incident_id=incident.id,
                    user_id=user_id,
                    alert_id=alert_id,
                    snapshot_id=snapshot_id,
                    evidence_type=str(item.get("evidence_type") or "OTHER").strip(),
                    object_type=str(item["object_type"]).strip() if item.get("object_type") is not None else None,
                    object_reference=str(item["object_reference"]).strip() if item.get("object_reference") is not None else None,
                    description=str(item["description"]) if item.get("description") is not None else None,
                    metadata_json=item.get("metadata"),
                )
                db.add(evidence)

        db.commit()
        db.refresh(incident)
    except Exception as exc:
        db.rollback()
        logger.exception("Failed creating incident")
        raise HTTPException(status_code=400, detail="Failed to create incident.") from exc

    return {
        "success": True,
        "message": "Incident created successfully.",
        "incident": serialize_incident(incident),
    }


@router.patch("/incidents/{incident_id}")
def update_incident(
    incident_id: int,
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id
    incident = (
        db.query(Incident)
        .filter(
            Incident.id == incident_id,
            Incident.user_id == user_id,
        )
        .first()
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    allowed_statuses = {"OPEN", "CLOSED", "RESOLVED", "ESCALATED", "UNDER_REVIEW"}

    if "cam_id" in payload:
        incident.cam_id = str(payload["cam_id"]).strip() if payload["cam_id"] is not None else None
    if "primary_track_id" in payload:
        incident.primary_track_id = str(payload["primary_track_id"]).strip() if payload["primary_track_id"] is not None else None
    if "global_vehicle_id" in payload:
        incident.global_vehicle_id = str(payload["global_vehicle_id"]).strip() if payload["global_vehicle_id"] is not None else None
    if "incident_type" in payload:
        value = str(payload["incident_type"] or "").strip()
        if not value:
            raise HTTPException(status_code=422, detail="incident_type cannot be empty.")
        incident.incident_type = value
    if "status" in payload:
        new_status = str(payload["status"] or "").strip().upper()
        if new_status not in allowed_statuses:
            raise HTTPException(status_code=422, detail="Invalid incident status.")
        incident.status = new_status
        if new_status in {"CLOSED", "RESOLVED"} and incident.ended_at is None:
            incident.ended_at = datetime.now(timezone.utc).replace(tzinfo=None)
    if "evidence" in payload:
        incident.evidence = str(payload["evidence"]) if payload["evidence"] is not None else None
    if "evaluation_status" in payload:
        incident.evaluation_status = str(payload["evaluation_status"] or "").strip()
    if "evaluation_score" in payload:
        incident.evaluation_score = float(payload["evaluation_score"]) if payload["evaluation_score"] is not None else None
    if "evaluation_confidence" in payload:
        incident.evaluation_confidence = float(payload["evaluation_confidence"]) if payload["evaluation_confidence"] is not None else None
    if "evaluation_severity" in payload:
        incident.evaluation_severity = str(payload["evaluation_severity"] or "").strip() or None
    if "evaluation_reason" in payload:
        incident.evaluation_reason = str(payload["evaluation_reason"]) if payload["evaluation_reason"] is not None else None
    if "evaluation_evidence" in payload:
        incident.evaluation_evidence = payload["evaluation_evidence"]
    if "evaluated_by" in payload:
        incident.evaluated_by = str(payload["evaluated_by"]) if payload["evaluated_by"] is not None else None
    if payload.get("evaluated_at") is not None:
        incident.evaluated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    try:
        db.commit()
        db.refresh(incident)
    except Exception as exc:
        db.rollback()
        logger.exception("Failed updating incident")
        raise HTTPException(status_code=400, detail="Failed to update incident.") from exc

    return {
        "success": True,
        "message": "Incident updated successfully.",
        "incident": serialize_incident(incident),
    }


@router.get("/incidents/{incident_id}/evidence")
def get_incident_evidence(
    incident_id: int,
    limit: int = Query(500, ge=1, le=2000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id
    incident = (
        db.query(Incident)
        .filter(Incident.id == incident_id, Incident.user_id == user_id)
        .first()
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    evidence = (
        db.query(IncidentEvidence)
        .filter(
            IncidentEvidence.incident_id == incident_id,
            IncidentEvidence.user_id == user_id,
        )
        .order_by(IncidentEvidence.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "success": True,
        "count": len(evidence),
        "evidence": [serialize_incident_evidence(x) for x in evidence],
    }


@router.post("/incidents/{incident_id}/evidence")
def add_incident_evidence(
    incident_id: int,
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id
    incident = (
        db.query(Incident)
        .filter(Incident.id == incident_id, Incident.user_id == user_id)
        .first()
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    evidence_type = str(payload.get("evidence_type") or "OTHER").strip()
    if not evidence_type:
        raise HTTPException(status_code=422, detail="evidence_type is required.")

    snapshot_id = (
        int(payload["snapshot_id"])
        if payload.get("snapshot_id") is not None
        else None
    )
    alert_id = (
        int(payload["alert_id"])
        if payload.get("alert_id") is not None
        else None
    )
    if snapshot_id is not None:
        owned_snapshot = (
            db.query(Snapshot)
            .join(Alert, Alert.id == Snapshot.alert_id)
            .filter(
                Snapshot.id == snapshot_id,
                Alert.user_id == user_id,
            )
            .first()
        )
        if owned_snapshot is None:
            raise HTTPException(
                status_code=403,
                detail="Snapshot does not belong to the authenticated user.",
            )

    evidence = IncidentEvidence(
        incident_id=incident_id,
        user_id=user_id,
        alert_id=alert_id,
        snapshot_id=snapshot_id,
        evidence_type=evidence_type,
        object_type=str(payload["object_type"]).strip() if payload.get("object_type") is not None else None,
        object_reference=str(payload["object_reference"]).strip() if payload.get("object_reference") is not None else None,
        description=str(payload["description"]) if payload.get("description") is not None else None,
        metadata_json=payload.get("metadata"),
    )

    try:
        db.add(evidence)
        db.commit()
        db.refresh(evidence)
    except Exception as exc:
        db.rollback()
        logger.exception("Failed adding incident evidence")
        raise HTTPException(status_code=400, detail="Failed to add incident evidence.") from exc

    return {
        "success": True,
        "message": "Incident evidence added successfully.",
        "evidence": serialize_incident_evidence(evidence),
    }
