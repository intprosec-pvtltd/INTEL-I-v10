from __future__ import annotations
from datetime import datetime, timedelta
import logging
from difflib import SequenceMatcher
import os
import re
from sqlalchemy.orm import Session
from db.watchlist_model import Watchlist, WatchlistEntry, indian_time

logger = logging.getLogger('watchlist-service')

PLATE_RE = re.compile(r'[^A-Z0-9]')
MAX_PLATE_LENGTH = 50


def watchlist_journey_lookback_seconds() -> int:
    """Return the long-term identity-history lookback.

    ``WATCHLIST_JOURNEY_WINDOW_SECONDS`` remains supported for existing
    deployments.  New deployments should use the clearer day-based setting.
    The old implementation silently capped this value at 24 hours, which
    split the same watchlist subject into unrelated incidents after a day.
    """

    raw_days = os.getenv("WATCHLIST_JOURNEY_LOOKBACK_DAYS")
    if raw_days is not None and raw_days.strip():
        try:
            days = int(raw_days)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid WATCHLIST_JOURNEY_LOOKBACK_DAYS=%r; using 365 days",
                raw_days,
            )
            days = 365
        return max(1, min(3650, days)) * 86400

    legacy_seconds = os.getenv("WATCHLIST_JOURNEY_WINDOW_SECONDS")
    if legacy_seconds is not None and legacy_seconds.strip():
        try:
            return max(300, min(315_360_000, int(legacy_seconds)))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid WATCHLIST_JOURNEY_WINDOW_SECONDS=%r; using 365 days",
                legacy_seconds,
            )

    return 365 * 86400


def normalize_plate(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('plate must be a string')
    normalized = PLATE_RE.sub('', value.upper().strip())
    if not normalized or len(normalized) > MAX_PLATE_LENGTH:
        raise ValueError('invalid plate')
    return normalized


def plate_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _entry_active(entry: WatchlistEntry, now: datetime) -> bool:
    if entry.status != 'ACTIVE':
        return False
    if entry.effective_from and now < entry.effective_from:
        return False
    if entry.effective_until and now > entry.effective_until:
        return False
    return True


def find_plate_matches(db: Session, user_id: int, plate: str, include_possible: bool = True):
    normalized = normalize_plate(plate)
    now = indian_time()
    rows = (
        db.query(WatchlistEntry)
        .filter(
            WatchlistEntry.user_id == user_id,
            WatchlistEntry.plate_normalized.isnot(None),
        )
        .all()
    )

    exact = []
    possible = []
    for entry in rows:
        if not _entry_active(entry, now):
            continue
        score = plate_similarity(normalized, entry.plate_normalized)
        if entry.plate_normalized == normalized:
            exact.append((entry, 1.0))
        elif include_possible and score >= 0.82:
            possible.append((entry, score))

    possible.sort(key=lambda item: item[1], reverse=True)
    return {
        'plate': plate,
        'plate_normalized': normalized,
        'exact': exact,
        'possible': possible[:10],
    }


def serialize_entry(entry: WatchlistEntry):
    category = str(entry.category or 'OTHER').upper()
    priority = str(entry.priority or '').upper()
    severity = priority if priority in {'HIGH','MEDIUM','LOW'} else {'STOLEN':'HIGH','WANTED':'HIGH','BLACKLISTED':'MEDIUM','SUSPICIOUS':'LOW','OTHER':'LOW'}.get(category,'LOW')
    return {
        'id': int(entry.id),
        'watchlist_id': int(entry.watchlist_id),
        'plate': entry.plate_display,
        'plate_normalized': entry.plate_normalized,
        'category': entry.category,
        'status': entry.status,
        'priority': entry.priority,
        'severity': severity,
        'description': entry.description,
        'source': entry.source,
        'effective_from': entry.effective_from.isoformat() if entry.effective_from else None,
        'effective_until': entry.effective_until.isoformat() if entry.effective_until else None,
        'version': entry.version,
        'created_at': entry.created_at.isoformat(),
        'updated_at': entry.updated_at.isoformat(),
    }


def create_exact_watchlist_alert(
    db: Session,
    *,
    user_id: int,
    camera_id: str,
    plate: str,
    track_id: str | None,
    source_type: str | None,
    global_vehicle_id: str | None = None,
    confidence: float = 1.0,
    correlation_confidence: float | None = None,
    snapshot_data=None,
    vehicle_attributes: dict | None = None,
):
    """Create an exact-match alert and outbox event in one DB transaction."""
    import json
    from db.model import Alert, OutboxEvent, Incident, Camera
    from db.advanced_intelligence_model import IncidentEvidence
    from db.crud import save_snapshot_to_db

    result = find_plate_matches(db, user_id, plate, include_possible=False)
    if not result['exact']:
        logger.info(
            "[INTEL-I][WATCHLIST] NO_EXACT_MATCH "
            "user=%s camera=%s track=%s plate_present=true",
            user_id,
            camera_id,
            track_id,
        )
        return None

    entry, _ = result['exact'][0]

    logger.info(
        "[INTEL-I][WATCHLIST] EXACT_MATCH "
        "user=%s camera=%s track=%s plate_present=true entry_id=%s category=%s confidence=%.3f",
        user_id,
        camera_id,
        track_id,
        entry.id,
        entry.category,
        float(confidence or 0.0),
    )

    # Exact watchlist matches are always HIGH priority per the operational
    # alert contract. The entry priority is retained as metadata by the
    # watchlist record, but an exact match must not be downgraded in the UI.
    category_level = {'STOLEN':'HIGH','WANTED':'HIGH','BLACKLISTED':'MEDIUM','SUSPICIOUS':'LOW','OTHER':'LOW'}
    entry_priority = str(entry.priority or '').upper()
    level = entry_priority if entry_priority in {'HIGH','MEDIUM','LOW'} else category_level.get(str(entry.category).upper(),'LOW')
    clean_track = str(track_id) if track_id is not None else 'global'
    cutoff = indian_time() - timedelta(seconds=120)

    duplicate = (
        db.query(Alert)
        .filter(
            Alert.user_id == user_id,
            Alert.cam_id == str(camera_id),
            Alert.rule == 'WATCHLIST_EXACT_MATCH',
            Alert.track_id == clean_track,
            Alert.created_at >= cutoff,
            Alert.watchlist_entry_id == entry.id,
        )
        .order_by(Alert.created_at.desc())
        .first()
    )
    if duplicate:
        return None

    alert = Alert(
        user_id=user_id,
        cam_id=str(camera_id),
        alert_type='WATCHLIST_MATCH',
        level=level,
        confidence_score=max(0.0, min(1.0, float(confidence))),
        track_id=clean_track,
        rule='WATCHLIST_EXACT_MATCH',
        source_type=str(source_type).strip() if source_type else None,
        zone=None,
        watchlist_entry_id=entry.id,
        plate=entry.plate_display,
        watchlist_category=entry.category,
        watchlist_status=entry.status,
        watchlist_match_type='EXACT',
        watchlist_match_confidence=float(confidence),
        watchlist_version=entry.version,
    )
    db.add(alert)
    db.flush()

    # Persist the evidence frame only after the exact match has been
    # confirmed. This keeps normal ANPR observations lightweight while
    # ensuring an operational watchlist alert has an evidence image.
    snapshot_obj = None
    if snapshot_data:
        try:
            snapshot_obj = save_snapshot_to_db(
                db=db,
                alert_id=int(alert.id),
                snapshot_data=snapshot_data,
            )
        except Exception:
            db.rollback()
            raise
    journey_window_seconds = watchlist_journey_lookback_seconds()
    # The watchlist record is the durable anchor.  Re-ID GlobalVehicleIDs can
    # change after process restarts, while this entry remains stable for the
    # configured month/year-scale history.
    identity_key = f'VEHICLE-WATCHLIST-{int(entry.id)}'
    incident_query = db.query(Incident).filter(
        Incident.user_id == int(user_id),
        Incident.incident_type == 'VEHICLE_WATCHLIST_MATCH',
        Incident.status.in_(['OPEN', 'UNDER_REVIEW', 'ESCALATED']),
        Incident.primary_track_id == identity_key,
        Incident.started_at >= indian_time() - timedelta(seconds=journey_window_seconds),
    )
    incident = incident_query.order_by(Incident.started_at.desc()).first()
    previous_incident = None
    if incident is None:
        previous_incident = (
            db.query(Incident)
            .filter(
                Incident.user_id == int(user_id),
                Incident.incident_type == 'VEHICLE_WATCHLIST_MATCH',
                Incident.status.in_(['RESOLVED', 'CLOSED']),
                Incident.primary_track_id == identity_key,
                Incident.started_at
                >= indian_time() - timedelta(seconds=journey_window_seconds),
            )
            .order_by(Incident.started_at.desc())
            .first()
        )

    camera = db.query(Camera).filter(
        Camera.user_id == int(user_id),
        Camera.cam_id == str(camera_id),
    ).first()
    observed_at = alert.created_at or indian_time()
    observation = {
        'sequence': 1,
        'camera_id': str(camera_id),
        'camera_name': str(getattr(camera, 'camera_name', None) or camera_id),
        'latitude': getattr(camera, 'latitude', None),
        'longitude': getattr(camera, 'longitude', None),
        'location_name': getattr(camera, 'location_name', None),
        'timestamp': observed_at.isoformat() if hasattr(observed_at, 'isoformat') else str(observed_at),
        'track_id': clean_track,
        'snapshot_id': int(snapshot_obj.id) if snapshot_obj else None,
        'confidence': float(confidence),
        'correlation_confidence': float(correlation_confidence or 0.0),
    }

    created_incident = incident is None
    if created_incident:
        previous_evidence = (
            previous_incident.evaluation_evidence
            if previous_incident is not None
            and isinstance(previous_incident.evaluation_evidence, dict)
            else {}
        )
        previous_journey = dict(previous_evidence.get('watchlist_journey') or {})
        previous_route = list(previous_journey.get('route') or [])
        observation['sequence'] = int(
            previous_journey.get('observation_count') or len(previous_route)
        ) + 1
        combined_route = (previous_route + [observation])[-200:]
        related_incident_ids = list(
            previous_journey.get('related_incident_ids') or []
        )
        if previous_incident is not None:
            related_incident_ids.append(int(previous_incident.id))
        related_incident_ids = list(dict.fromkeys(related_incident_ids))[-100:]
        journey = {
            'identity_type': 'VEHICLE',
            'identity_id': identity_key,
            'global_vehicle_id': global_vehicle_id,
            'watchlist_entry_id': int(entry.id),
            'reference': entry.plate_display,
            'category': entry.category,
            'tracking_status': 'ACTIVE_TRACKING' if len(combined_route) > 1 else 'WAITING_FOR_NEXT_CAMERA',
            'first_camera_id': previous_journey.get('first_camera_id') or str(camera_id),
            'last_camera_id': str(camera_id),
            'first_seen': previous_journey.get('first_seen') or observation['timestamp'],
            'last_seen': observation['timestamp'],
            'camera_count': len({str(item.get('camera_id')) for item in combined_route}),
            'observation_count': int(previous_journey.get('observation_count') or 0) + 1,
            'previous_incident_id': int(previous_incident.id) if previous_incident is not None else None,
            'related_incident_ids': related_incident_ids,
            'route': combined_route,
        }
        incident = Incident(
            user_id=user_id,
            cam_id=str(camera_id),
            primary_track_id=identity_key,
            global_vehicle_id=global_vehicle_id,
            incident_type='VEHICLE_WATCHLIST_MATCH',
            status='OPEN',
            evaluation_status='PENDING',
            evaluation_severity=level,
            evaluation_confidence=float(confidence),
            evaluation_reason=f'Vehicle watchlist match: {entry.category}',
            evaluation_evidence={'watchlist_journey': journey},
        )
        db.add(incident)
        db.flush()
    else:
        evidence_state = (
            dict(incident.evaluation_evidence)
            if isinstance(incident.evaluation_evidence, dict)
            else {}
        )
        journey = dict(evidence_state.get('watchlist_journey') or {})
        route = list(journey.get('route') or [])
        observation['sequence'] = int(
            journey.get('observation_count') or len(route)
        ) + 1
        route.append(observation)
        journey.update({
            'identity_type': 'VEHICLE',
            'identity_id': identity_key,
            'global_vehicle_id': global_vehicle_id or journey.get('global_vehicle_id'),
            'watchlist_entry_id': int(entry.id),
            'reference': entry.plate_display,
            'category': entry.category,
            'tracking_status': 'ACTIVE_TRACKING' if len(route) > 1 else 'WAITING_FOR_NEXT_CAMERA',
            'first_camera_id': journey.get('first_camera_id') or str(camera_id),
            'last_camera_id': str(camera_id),
            'first_seen': journey.get('first_seen') or observation['timestamp'],
            'last_seen': observation['timestamp'],
            'camera_count': len({str(item.get('camera_id')) for item in route}),
            'observation_count': int(journey.get('observation_count') or 0) + 1,
            'route': route[-200:],
        })
        evidence_state['watchlist_journey'] = journey
        incident.evaluation_evidence = evidence_state
        incident.cam_id = str(camera_id)
        incident.global_vehicle_id = incident.global_vehicle_id or global_vehicle_id
        incident.evaluation_confidence = max(
            float(incident.evaluation_confidence or 0.0),
            float(confidence),
        )

    db.add(IncidentEvidence(
        incident_id=int(incident.id),
        user_id=user_id,
        alert_id=int(alert.id),
        snapshot_id=int(snapshot_obj.id) if snapshot_obj else None,
        evidence_type='SNAPSHOT' if snapshot_obj else 'ALERT',
        object_type='VEHICLE',
        object_reference=entry.plate_display,
        description=f'Exact vehicle watchlist match: {entry.category}',
        metadata_json={
            'watchlist_entry_id': int(entry.id),
            'category': entry.category,
            'confidence': float(confidence),
            'global_vehicle_id': global_vehicle_id,
            'journey_observation': observation,
        },
    ))

    payload = {
        'type': 'alert',
        'event': 'WATCHLIST_MATCH',
        'id': int(alert.id),
        'alert_id': int(alert.id),
        'user_id': int(user_id),
        'camera_id': str(camera_id),
        'cam_id': str(camera_id),
        'source_type': str(source_type).strip() if source_type else 'unknown',
        'alert_type': 'WATCHLIST_MATCH',
        'alert_level': level,
        'level': level,
        'severity': level,
        'rule': 'WATCHLIST_EXACT_MATCH',
        'alert_rule': 'WATCHLIST_EXACT_MATCH',
        'plate': entry.plate_display,
        'category': entry.category,
        'watchlist_category': entry.category,
        'status': entry.status,
        'watchlist_status': entry.status,
        'priority': level,
        'match_type': 'EXACT',
        'watchlist_match_type': 'EXACT',
        'match_confidence': float(confidence),
        'watchlist_match_confidence': float(confidence),
        'confidence_score': max(0.0, min(1.0, float(confidence))),
        'confidence': max(0.0, min(1.0, float(confidence))),
        'watchlist_entry_id': int(entry.id),
        'watchlist_version': int(entry.version),
        'global_vehicle_id': global_vehicle_id,
        'vehicle_attributes': vehicle_attributes or {},
        'incident_id': int(incident.id),
        'incident_created': created_incident,
        'watchlist_journey': journey,
        'track_id': clean_track,
        'correlation_confidence': correlation_confidence,
        'message': f'Watchlist exact match: {entry.category}',
        'snapshot_id': int(snapshot_obj.id) if snapshot_obj else None,
        'snapshot_saved': bool(snapshot_obj),
        'snapshot_path': f'/snapshot/{snapshot_obj.id}' if snapshot_obj else None,
        'has_snapshot': bool(snapshot_obj),
        'created_at': alert.created_at.isoformat(),
    }

    outbox = OutboxEvent(
        topic=os.getenv('KAFKA_SAVED_ALERT_TOPIC', 'cctv.alerts.saved'),
        event_key=str(alert.id),
        payload=json.dumps(payload, default=str),
        status='PENDING',
        attempts=0,
        next_attempt_at=indian_time(),
    )
    db.add(outbox)
    db.commit()
    db.refresh(alert)

    logger.warning(
        "[INTEL-I][WATCHLIST] ALERT_PERSISTED "
        "alert_id=%s incident_id=%s entry_id=%s plate_present=true category=%s "
        "track=%s severity=%s snapshot_id=%s",
        alert.id,
        incident.id,
        entry.id,
        entry.category,
        clean_track,
        level,
        snapshot_obj.id if snapshot_obj else None,
    )

    return payload
