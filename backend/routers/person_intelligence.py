from __future__ import annotations

import hashlib
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from db.database import getDB
from db.model import Camera, User
from db.intelligence_model import Person, PersonObservation, InvestigationSummary
from services.gis_engine import haversine_distance_meters
from services.investigationSummary import investigation_summary_worker

router = APIRouter(prefix="/api/persons", tags=["Person Intelligence"])


def _serialize_observation(row: PersonObservation, camera: Camera | None):
    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    return {
        "observation_id": int(row.id), "global_person_id": row.global_person_id,
        "camera_id": row.camera_id, "camera_name": getattr(camera, "name", None) or row.camera_id,
        "local_track_id": row.local_track_id, "confidence": row.confidence,
        "timestamp": row.frame_timestamp.isoformat(), "bbox": row.bbox,
        "latitude": getattr(camera, "latitude", None), "longitude": getattr(camera, "longitude", None),
        "location_name": getattr(camera, "location_name", None), "appearance": metadata,
    }


@router.get("/correlated")
def correlated_persons(limit: int = Query(100, ge=1, le=500), db: Session = Depends(getDB), current_user: User = Depends(get_current_user)):
    rows = db.query(Person).filter(Person.user_id == current_user.id).order_by(Person.last_seen_at.desc()).limit(limit).all()
    return {"count": len(rows), "persons": [{"global_person_id": p.global_person_id, "first_seen": p.first_seen_at.isoformat(), "last_seen": p.last_seen_at.isoformat(), "observation_count": p.observation_count, "status": p.status, "confidence_status": (p.metadata_json or {}).get("confidence_status", "POSSIBLE")} for p in rows]}


@router.get("/{global_person_id}/journey")
def person_journey(global_person_id: str, db: Session = Depends(getDB), current_user: User = Depends(get_current_user)):
    gid = str(global_person_id).strip().upper()
    person = db.query(Person).filter(Person.user_id == current_user.id, Person.global_person_id == gid).first()
    if person is None:
        raise HTTPException(404, "Correlated person not found")
    rows = db.query(PersonObservation).filter(PersonObservation.user_id == current_user.id, PersonObservation.global_person_id == gid).order_by(PersonObservation.frame_timestamp.asc()).limit(10000).all()
    camera_ids = {r.camera_id for r in rows}
    cameras = {c.cam_id: c for c in db.query(Camera).filter(Camera.user_id == current_user.id, Camera.cam_id.in_(camera_ids)).all()} if camera_ids else {}
    observations = [_serialize_observation(r, cameras.get(r.camera_id)) for r in rows]
    segments=[]
    for left,right in zip(observations, observations[1:]):
        distance = haversine_distance_meters(left.get("latitude"), left.get("longitude"), right.get("latitude"), right.get("longitude"))
        segments.append({"source_camera_id": left["camera_id"], "destination_camera_id": right["camera_id"], "source_timestamp": left["timestamp"], "destination_timestamp": right["timestamp"], "distance_meters": distance, "observed": True})
    return {"global_person_id": gid, "journey_label": "Observed Person Journey Across Integrated CCTV Coverage", "observations": observations, "route": [o for o in observations if o.get("latitude") is not None and o.get("longitude") is not None], "segments": segments, "observation_count": len(observations)}


class SummaryRequest(BaseModel):
    subject_type: str = Field(pattern="^(VEHICLE|PERSON|INCIDENT)$")
    subject_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    incident_id: int | None = Field(default=None, ge=1)


@router.post("/investigation-summaries", status_code=202)
def create_summary(payload: SummaryRequest, db: Session = Depends(getDB), current_user: User = Depends(get_current_user)):
    if payload.subject_type == "PERSON":
        rows = db.query(PersonObservation).filter(PersonObservation.user_id == current_user.id, PersonObservation.global_person_id == payload.subject_id.upper()).order_by(PersonObservation.frame_timestamp.asc()).limit(5000).all()
        evidence = [{"camera_id": r.camera_id, "timestamp": r.frame_timestamp.isoformat(), "confidence": r.confidence, "appearance": r.metadata_json} for r in rows]
    elif payload.subject_type == "VEHICLE":
        from db.intelligence_model import VehicleObservation
        camera_ids = {c.cam_id for c in db.query(Camera).filter(Camera.user_id == current_user.id).all()}
        rows = db.query(VehicleObservation).filter(VehicleObservation.global_vehicle_id == payload.subject_id.upper(), VehicleObservation.camera_id.in_(camera_ids)).order_by(VehicleObservation.frame_timestamp.asc()).limit(5000).all()
        evidence = [{"camera_id": r.camera_id, "timestamp": r.frame_timestamp.isoformat(), "confidence": r.confidence, "metadata": r.metadata_json} for r in rows]
    else:
        from db.model import Incident
        from db.advanced_intelligence_model import IncidentEvidence
        try:
            incident_id = payload.incident_id or int(payload.subject_id)
        except (TypeError, ValueError):
            raise HTTPException(422, "A numeric incident ID is required")
        incident = db.query(Incident).filter(Incident.id == incident_id, Incident.user_id == current_user.id).first()
        if incident is None:
            raise HTTPException(404, "Incident not found")
        items = db.query(IncidentEvidence).filter(IncidentEvidence.incident_id == incident_id, IncidentEvidence.user_id == current_user.id).order_by(IncidentEvidence.created_at.asc()).limit(5000).all()
        evidence = [{"incident_id": incident_id, "camera_id": incident.cam_id, "incident_type": incident.incident_type, "status": incident.status, "evaluation_status": incident.evaluation_status, "evaluation_confidence": incident.evaluation_confidence, "evidence_type": item.evidence_type, "description": item.description, "metadata": item.metadata_json, "created_at": item.created_at.isoformat()} for item in items]
    if not evidence:
        raise HTTPException(409, "Persisted correlation evidence is required before summary generation")
    version = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()
    row = InvestigationSummary(user_id=current_user.id, incident_id=payload.incident_id, subject_type=payload.subject_type, subject_id=payload.subject_id.upper(), source_version=version, model_name=investigation_summary_worker.model, status="PENDING")
    db.add(row); db.commit(); db.refresh(row)
    if not investigation_summary_worker.submit(int(row.id), {"subject_type": payload.subject_type, "subject_id": payload.subject_id.upper(), "observations": evidence}):
        row.status = "DISABLED"; row.error_code = "OLLAMA_SUMMARY_DISABLED"; db.commit()
    return {"summary_id": int(row.id), "status": row.status, "source_version": version}


@router.get("/investigation-summaries/{summary_id}")
def get_summary(summary_id: int, db: Session = Depends(getDB), current_user: User = Depends(get_current_user)):
    row = db.query(InvestigationSummary).filter(InvestigationSummary.id == summary_id, InvestigationSummary.user_id == current_user.id).first()
    if row is None: raise HTTPException(404, "Summary not found")
    return {"summary_id": int(row.id), "status": row.status, "subject_type": row.subject_type, "subject_id": row.subject_id, "source_version": row.source_version, "model_name": row.model_name, "summary": row.summary_text, "structured_summary": row.structured_summary, "error_code": row.error_code, "created_at": row.created_at.isoformat(), "completed_at": row.completed_at.isoformat() if row.completed_at else None}
