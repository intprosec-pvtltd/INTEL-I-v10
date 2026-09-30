from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from db.database import getDB
from db.model import Camera, CameraConnection, User
from db.intelligence_model import VehicleObservation
from services.gis_engine import CameraPoint, evaluate_camera_transition
from schemas.gis import (
    CameraConnectionCreate,
    CameraConnectionListResponse,
    CameraConnectionResponse,
    GISTransitionEvaluationRequest,
    GISTransitionEvaluationResponse,
    GISJourneyObservation,
    GISJourneySegment,
    GISVehicleJourneyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gis", tags=["GIS"])


def _camera_id(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 100:
        raise HTTPException(status_code=400, detail="Invalid camera ID")
    return value


def _owned_camera(db: Session, camera_id: str, user_id: int) -> Camera:
    camera_id = _camera_id(camera_id)
    camera = (
        db.query(Camera)
        .filter(Camera.cam_id == camera_id, Camera.user_id == user_id)
        .first()
    )
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


def _serialize_connection(connection: CameraConnection) -> CameraConnectionResponse:
    return CameraConnectionResponse(
        id=int(connection.id),
        source_camera_id=str(connection.source_camera_id),
        destination_camera_id=str(connection.destination_camera_id),
        distance_km=connection.distance_km,
        expected_min_seconds=connection.expected_min_seconds,
        expected_max_seconds=connection.expected_max_seconds,
        expected_seconds=connection.expected_seconds,
        road_name=connection.road_name,
        direction=connection.direction,
        direction_tolerance=connection.direction_tolerance,
        road_distance_km=connection.road_distance_km,
        road_duration_seconds=connection.road_duration_seconds,
        route_geometry=connection.route_geometry,
        routing_provider=connection.routing_provider,
        confidence=connection.confidence,
        active=bool(connection.active),
        created_at=connection.created_at,
        updated_at=connection.updated_at,
    )


def _camera_metadata(camera: Camera) -> dict:
    return {
        "camera_id": str(camera.cam_id),
        "latitude": camera.latitude,
        "longitude": camera.longitude,
        "altitude": getattr(camera, "altitude", None),
        "heading": getattr(camera, "heading", None),
        "fov": getattr(camera, "fov", None),
        "road_name": getattr(camera, "road_name", None),
        "location_name": getattr(camera, "location_name", None),
    }


@router.get("/connections", response_model=CameraConnectionListResponse)
def list_camera_connections(
    camera_id: str | None = Query(default=None, min_length=1, max_length=100),
    active_only: bool = Query(default=True),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    owned_ids = {
        str(c.cam_id)
        for c in db.query(Camera.cam_id).filter(Camera.user_id == current_user.id).all()
    }

    query = db.query(CameraConnection)
    if camera_id:
        camera_id = _camera_id(camera_id)
        if camera_id not in owned_ids:
            raise HTTPException(status_code=404, detail="Camera not found")
        query = query.filter(CameraConnection.source_camera_id == camera_id)

    if active_only:
        query = query.filter(CameraConnection.active.is_(True))

    query = query.filter(
        CameraConnection.source_camera_id.in_(owned_ids),
        CameraConnection.destination_camera_id.in_(owned_ids),
    )

    connections = query.order_by(CameraConnection.id.desc()).limit(limit).all()
    return CameraConnectionListResponse(
        count=len(connections),
        connections=[_serialize_connection(item) for item in connections],
    )


@router.post(
    "/connections",
    response_model=CameraConnectionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_or_update_camera_connection(
    payload: CameraConnectionCreate,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    source = _owned_camera(db, payload.source_camera_id, current_user.id)
    destination = _owned_camera(db, payload.destination_camera_id, current_user.id)

    if source.cam_id == destination.cam_id:
        raise HTTPException(status_code=400, detail="A camera cannot connect to itself")

    if (
        payload.expected_min_seconds is not None
        and payload.expected_max_seconds is not None
        and payload.expected_min_seconds > payload.expected_max_seconds
    ):
        raise HTTPException(status_code=422, detail="expected_min_seconds cannot exceed expected_max_seconds")

    connection = (
        db.query(CameraConnection)
        .filter(
            CameraConnection.source_camera_id == source.cam_id,
            CameraConnection.destination_camera_id == destination.cam_id,
        )
        .first()
    )

    values = payload.model_dump()
    if connection is None:
        connection = CameraConnection(**values)
        db.add(connection)
    else:
        for key, value in values.items():
            setattr(connection, key, value)

    try:
        db.commit()
        db.refresh(connection)
    except Exception:
        db.rollback()
        logger.exception("Failed to persist GIS camera connection")
        raise HTTPException(status_code=500, detail="Unable to save camera connection")

    return _serialize_connection(connection)


@router.get(
    "/connections/{source_camera_id}/{destination_camera_id}",
    response_model=CameraConnectionResponse,
)
def get_camera_connection(
    source_camera_id: str,
    destination_camera_id: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    _owned_camera(db, source_camera_id, current_user.id)
    _owned_camera(db, destination_camera_id, current_user.id)

    connection = (
        db.query(CameraConnection)
        .filter(
            CameraConnection.source_camera_id == _camera_id(source_camera_id),
            CameraConnection.destination_camera_id == _camera_id(destination_camera_id),
        )
        .first()
    )
    if connection is None:
        raise HTTPException(status_code=404, detail="Camera connection not found")
    return _serialize_connection(connection)


@router.delete("/connections/{source_camera_id}/{destination_camera_id}")
def deactivate_camera_connection(
    source_camera_id: str,
    destination_camera_id: str,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    _owned_camera(db, source_camera_id, current_user.id)
    _owned_camera(db, destination_camera_id, current_user.id)

    connection = (
        db.query(CameraConnection)
        .filter(
            CameraConnection.source_camera_id == _camera_id(source_camera_id),
            CameraConnection.destination_camera_id == _camera_id(destination_camera_id),
        )
        .first()
    )
    if connection is None:
        raise HTTPException(status_code=404, detail="Camera connection not found")

    connection.active = False
    db.commit()
    return {"message": "Camera connection deactivated"}


@router.post("/evaluate-transition", response_model=GISTransitionEvaluationResponse)
def evaluate_transition(
    payload: GISTransitionEvaluationRequest,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    source = _owned_camera(db, payload.source_camera_id, current_user.id)
    destination = _owned_camera(db, payload.destination_camera_id, current_user.id)

    if payload.max_speed_kmh < payload.min_speed_kmh:
        raise HTTPException(status_code=422, detail="max_speed_kmh must be >= min_speed_kmh")

    result = evaluate_camera_transition(
        source_camera=CameraPoint(**_camera_metadata(source)),
        destination_camera=CameraPoint(**_camera_metadata(destination)),
        source_timestamp=payload.source_timestamp,
        destination_timestamp=payload.destination_timestamp,
        source_heading=payload.source_heading,
        destination_heading=payload.destination_heading,
        min_speed_kmh=payload.min_speed_kmh,
        max_speed_kmh=payload.max_speed_kmh,
    )

    return GISTransitionEvaluationResponse(
        source_camera_id=str(source.cam_id),
        destination_camera_id=str(destination.cam_id),
        **result.as_dict(),
    )


@router.get("/vehicle/{global_vehicle_id}/journey", response_model=GISVehicleJourneyResponse)
def persistent_vehicle_journey(
    global_vehicle_id: str,
    limit: int = Query(default=500, ge=1, le=2000),
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    global_vehicle_id = str(global_vehicle_id or "").strip().upper()
    if not global_vehicle_id or len(global_vehicle_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid global vehicle ID")

    owned_ids = {
        str(c.cam_id)
        for c in db.query(Camera.cam_id).filter(Camera.user_id == current_user.id).all()
    }
    if not owned_ids:
        raise HTTPException(status_code=404, detail="No cameras available")

    rows = (
        db.query(VehicleObservation)
        .filter(
            VehicleObservation.global_vehicle_id == global_vehicle_id,
            VehicleObservation.camera_id.in_(owned_ids),
        )
        .order_by(VehicleObservation.frame_timestamp.asc(), VehicleObservation.id.asc())
        .limit(limit)
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Vehicle journey not found")

    cameras = {
        str(c.cam_id): c
        for c in db.query(Camera).filter(Camera.user_id == current_user.id, Camera.cam_id.in_(owned_ids)).all()
    }

    observations: list[GISJourneyObservation] = []
    for row in rows:
        camera = cameras.get(str(row.camera_id))
        observations.append(
            GISJourneyObservation(
                observation_id=int(row.id),
                camera_id=str(row.camera_id),
                vehicle_type=row.vehicle_type,
                confidence=row.confidence,
                frame_timestamp=row.frame_timestamp,
                latitude=getattr(camera, "latitude", None) if camera else None,
                longitude=getattr(camera, "longitude", None) if camera else None,
                local_track_id=row.local_track_id,
            )
        )

    segments: list[GISJourneySegment] = []
    previous = None
    for current in observations:
        if previous is not None and previous.camera_id != current.camera_id:
            source_camera = cameras.get(previous.camera_id)
            destination_camera = cameras.get(current.camera_id)
            if source_camera and destination_camera:
                source_point = CameraPoint(**_camera_metadata(source_camera))
                destination_point = CameraPoint(**_camera_metadata(destination_camera))
                result = evaluate_camera_transition(
                    source_point,
                    destination_point,
                    previous.frame_timestamp.timestamp(),
                    current.frame_timestamp.timestamp(),
                )
                segments.append(
                    GISJourneySegment(
                        from_camera_id=previous.camera_id,
                        to_camera_id=current.camera_id,
                        from_timestamp=previous.frame_timestamp,
                        to_timestamp=current.frame_timestamp,
                        elapsed_seconds=max(0.0, (current.frame_timestamp - previous.frame_timestamp).total_seconds()),
                        distance_km=(result.distance_meters / 1000.0) if result.distance_meters is not None else None,
                        bearing_degrees=result.bearing_degrees,
                        feasible=result.feasible,
                        gis_score=result.gis_score,
                        direction_score=result.direction_score,
                        travel_time_score=result.travel_time_score,
                        reason=result.reason,
                    )
                )
        previous = current

    return GISVehicleJourneyResponse(
        global_vehicle_id=global_vehicle_id,
        observation_count=len(observations),
        transition_count=len(segments),
        observations=observations,
        segments=segments,
    )
