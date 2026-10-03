from __future__ import annotations
import base64
import binascii
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from db.advanced_intelligence_model import PersonWatchlistEntry
from db.watchlist_model import Watchlist, WatchlistEntry, indian_time
from db.external_watchlist_model import ExternalWatchlistRecord, ExternalWatchlistSource, ExternalWatchlistSyncHistory
from integrations.government_watchlist.mapper import normalize_record
from integrations.government_watchlist.providers.rest import RestJsonProvider
from integrations.government_watchlist.security import decrypt_credential
from services.personRecognition import VERSION, encrypt_bytes, encrypt_embedding, extract_single_embedding
from services.watchlist_service import normalize_plate

logger = logging.getLogger("government-watchlist")
PROVIDER = RestJsonProvider()
try:
    from monitoring import metrics as _metrics
except Exception:
    _metrics = None

MAX_IMAGE_BYTES = 10 * 1024 * 1024


def safe_source(source):
    return {"id": int(source.id), "name": source.name, "description": source.description,
            "provider_type": source.provider_type, "base_url": source.base_url, "enabled": bool(source.enabled),
            "read_only": bool(source.read_only), "auth_type": source.auth_type,
            "credential_configured": bool(source.credential_ciphertext), "api_key_header": source.api_key_header,
            "entity_types": source.entity_types or [], "field_mapping": source.field_mapping or {},
            "records_path": source.records_path, "pagination": source.pagination or {},
            "tls_verify": bool(source.tls_verify), "sync_mode": source.sync_mode, "sync_interval_seconds": source.sync_interval_seconds,
            "status": "DISABLED" if not source.enabled else source.status, "last_sync_at": source.last_sync_at.isoformat() if source.last_sync_at else None,
            "last_success_at": source.last_success_at.isoformat() if source.last_success_at else None,
            "next_sync_at": (source.last_sync_at + timedelta(seconds=source.sync_interval_seconds)).isoformat() if source.enabled and source.sync_mode == "INTERVAL" and source.last_sync_at else None,
            "last_error": source.last_error, "created_at": source.created_at.isoformat() if source.created_at else None}


def deactivate_source_records(db: Session, source: ExternalWatchlistSource) -> None:
    """Stop disabled source records from matching while preserving native entries."""
    rows = db.query(ExternalWatchlistRecord).filter(ExternalWatchlistRecord.source_id == source.id, ExternalWatchlistRecord.active.is_(True)).all()
    for row in rows:
        row.active = False
        if row.entity_type == "PERSON":
            person = db.query(PersonWatchlistEntry).filter(PersonWatchlistEntry.user_id == source.owner_user_id,
                PersonWatchlistEntry.external_reference == f"extwl:{source.id}:{row.external_id}").first()
            if person is not None:
                person.status = "DISABLED"; person.updated_at = indian_time()
        elif row.local_watchlist_entry_id:
            vehicle = db.query(WatchlistEntry).filter(WatchlistEntry.id == row.local_watchlist_entry_id,
                WatchlistEntry.user_id == source.owner_user_id).first()
            if vehicle is not None:
                vehicle.status = "DISABLED"; vehicle.updated_at = indian_time()
    db.flush()


def test_source(source):
    rows = PROVIDER.fetch_records(source, decrypt_credential(source.credential_ciphertext))
    return {"connected": True, "records_available": len(rows), "provider": "REST_JSON"}


def _vehicle_watchlist(db: Session, source: ExternalWatchlistSource) -> Watchlist:
    name = f"External: {source.name}"[:150]
    obj = db.query(Watchlist).filter(Watchlist.user_id == source.owner_user_id, Watchlist.name == name).first()
    if not obj:
        obj = Watchlist(user_id=source.owner_user_id, name=name, description="Read-only synchronized external watchlist", is_active=True)
        db.add(obj); db.flush()
    return obj


def _safe_metadata(raw: dict, record: dict) -> dict:
    # Keep bounded, non-binary, non-secret source context; images remain encrypted in the existing person table.
    allowed = {"external_id", "entity_type", "category", "active", "priority", "case_reference", "name", "plate", "make", "model", "color", "description"}
    mapped = {key: value for key, value in record.items() if key in allowed and value is not None}
    raw_json = json.dumps(mapped, ensure_ascii=False, default=str)
    if len(raw_json) > 8000:
        raw_json = raw_json[:8000]
    return {"external_watchlist": mapped, "source_record_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()}


def _save_person(db: Session, source: ExternalWatchlistSource, ext: ExternalWatchlistRecord, record: dict, raw: dict):
    image_value = record.get("image_data")
    person = db.query(PersonWatchlistEntry).filter(PersonWatchlistEntry.user_id == source.owner_user_id, PersonWatchlistEntry.external_reference == f"extwl:{source.id}:{ext.external_id}").first()
    status = "ACTIVE" if record["active"] else "DISABLED"
    category_raw = record["category"]
    category = "MISSING" if "MISSING" in category_raw else ("WANTED" if any(word in category_raw for word in ("WANTED", "SUSPECT", "BLACKLIST")) else "OTHER")
    image_changed = False
    if image_value:
        try:
            encoded = str(image_value)
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Person image must be valid base64 image data") from exc
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Person image is empty or exceeds 10 MB")
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Person image could not be decoded")
        vector = extract_single_embedding(frame)
        if vector is None or np.asarray(vector).size == 0:
            raise ValueError("Existing INTEL-I face pipeline could not create an embedding")
        content_type = str(record.get("image_content_type") or "image/jpeg").lower().split(";", 1)[0]
        if content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Person image content type must be JPEG, PNG, or WEBP")
        enc_image = encrypt_bytes(image_bytes)
        enc_embedding = encrypt_embedding(np.asarray(vector, dtype=np.float32))
        image_changed = True
    if person is None and not image_value:
        # Keep the external entity in the integration registry. FRS matching requires a validated reference photo.
        return None
    metadata = _safe_metadata(raw, record)
    metadata.update({"external_source_id": int(source.id), "external_record_id": ext.external_id, "source_name": source.name})
    if person is None:
        person = PersonWatchlistEntry(user_id=source.owner_user_id, full_name=record["name"], category=category, status=status,
            description=record.get("description"), source=source.name, external_reference=f"extwl:{source.id}:{ext.external_id}",
            embedding_encrypted=enc_embedding, embedding_dimension=int(np.asarray(vector).size), face_model="INTEL-I configured face pipeline",
            face_model_version=VERSION, reference_image_data=enc_image, reference_image_content_type=content_type,
            reference_image_filename=f"external-{ext.external_id}.jpg", reference_image_size=len(image_bytes), metadata_json=metadata)
        db.add(person); db.flush()
    else:
        person.full_name = record["name"] or person.full_name; person.category = category; person.status = status
        person.description = record.get("description"); person.source = source.name; person.metadata_json = metadata
        if image_changed:
            person.embedding_encrypted = enc_embedding; person.embedding_dimension = int(np.asarray(vector).size)
            person.reference_image_data = enc_image; person.reference_image_content_type = content_type
            person.reference_image_filename = f"external-{ext.external_id}.jpg"; person.reference_image_size = len(image_bytes)
        person.updated_at = indian_time()
    return person


def _save_vehicle(db: Session, source: ExternalWatchlistSource, ext: ExternalWatchlistRecord, record: dict, raw: dict):
    watchlist = _vehicle_watchlist(db, source)
    entry = None
    if ext.local_watchlist_entry_id:
        entry = db.query(WatchlistEntry).filter(WatchlistEntry.id == ext.local_watchlist_entry_id, WatchlistEntry.user_id == source.owner_user_id).first()
    metadata = _safe_metadata(raw, record)
    status = "ACTIVE" if record["active"] else "DISABLED"
    category = record["category"][:50]
    priority = record.get("priority") if record.get("priority") in {"HIGH", "MEDIUM", "LOW"} else "HIGH"
    if entry is None:
        entry = WatchlistEntry(watchlist_id=watchlist.id, user_id=source.owner_user_id, plate_normalized=normalize_plate(record["plate"]),
            plate_display=record["plate"], category=category, status=status, priority=priority,
            description=record.get("description"), source=source.name, version=1,
            metadata_json={**metadata, "external_source_id": int(source.id), "external_record_id": ext.external_id})
        db.add(entry); db.flush(); ext.local_watchlist_entry_id = int(entry.id)
    else:
        entry.plate_normalized = normalize_plate(record["plate"]); entry.plate_display = record["plate"]
        entry.category = category; entry.status = status; entry.priority = priority; entry.description = record.get("description")
        entry.source = source.name; entry.metadata_json = {**metadata, "external_source_id": int(source.id), "external_record_id": ext.external_id}
        entry.version += 1; entry.updated_at = indian_time()
    watchlist.updated_at = indian_time()
    return entry


def sync_source(db: Session, source: ExternalWatchlistSource, actor_user_id: int, *, action="MANUAL"):
    start = time.monotonic(); stats = {"received": 0, "created": 0, "updated": 0, "unchanged": 0, "rejected": 0}
    source_id = int(source.id)
    claimed = db.query(ExternalWatchlistSource).filter(ExternalWatchlistSource.id == source_id).with_for_update(skip_locked=True).first()
    if claimed is None:
        return {**stats, "deactivated": 0, "duration_ms": 0, "status": "SYNCING", "error": "A sync is already in progress"}
    if claimed.status == "SYNCING" and claimed.updated_at and (datetime.utcnow() - claimed.updated_at).total_seconds() < 600:
        return {**stats, "deactivated": 0, "duration_ms": 0, "status": "SYNCING", "error": "A sync is already in progress"}
    source = claimed
    source.status = "SYNCING"; source.last_error = None; source.last_sync_at = indian_time(); source.updated_at = indian_time(); db.commit()
    try:
        raw_rows = PROVIDER.fetch_records(source, decrypt_credential(source.credential_ciphertext))
        stats["received"] = len(raw_rows)
        for raw in raw_rows:
            try:
                item = normalize_record(raw, source.field_mapping or {})
                if item["entity_type"] not in (source.entity_types or ["PERSON", "VEHICLE"]):
                    stats["rejected"] += 1; continue
                existing = db.query(ExternalWatchlistRecord).filter(ExternalWatchlistRecord.source_id == source.id, ExternalWatchlistRecord.external_id == item["external_id"]).first()
                created = existing is None
                digest = hashlib.sha256(json.dumps(_safe_metadata(raw, item), sort_keys=True).encode()).hexdigest()
                old_digest = (existing.raw_metadata or {}).get("source_record_sha256") if existing else None
                if existing and old_digest == digest and not (item.get("image_data")):
                    existing.last_synced_at = indian_time(); stats["unchanged"] += 1; db.commit(); continue
                if existing is None:
                    existing = ExternalWatchlistRecord(source_id=source.id, owner_user_id=source.owner_user_id, external_id=item["external_id"],
                        entity_type=item["entity_type"], category=item["category"], source_name=source.name, raw_metadata={})
                    db.add(existing); db.flush()
                existing.entity_type = item["entity_type"]; existing.category = item["category"]; existing.active = item["active"]
                existing.priority = item.get("priority"); existing.case_reference = item.get("case_reference"); existing.name = item.get("name")
                existing.plate_display = item.get("plate"); existing.plate_normalized = normalize_plate(item["plate"]) if item.get("plate") else None
                existing.make = item.get("make"); existing.model = item.get("model"); existing.color = item.get("color")
                existing.source_name = source.name; existing.raw_metadata = _safe_metadata(raw, item); existing.last_synced_at = indian_time()
                if item["entity_type"] == "PERSON":
                    local = _save_person(db, source, existing, item, raw)
                    existing.image_available = bool(local and local.reference_image_data)
                    if local is not None: existing.local_watchlist_entry_id = int(local.id)
                else:
                    local = _save_vehicle(db, source, existing, item, raw)
                    existing.local_watchlist_entry_id = int(local.id)
                stats["created" if created else "updated"] += 1
                db.commit()
            except Exception as exc:
                db.rollback()
                source = db.query(ExternalWatchlistSource).filter(ExternalWatchlistSource.id == source.id).first()
                stats["rejected"] += 1
                logger.warning("External watchlist record rejected | source_id=%s reason=%s", source.id, type(exc).__name__)
        source = db.query(ExternalWatchlistSource).filter(ExternalWatchlistSource.id == source.id).first()
        db.commit()
        # Use INTEL-I's existing FRS worker reload API after the shared person
        # watchlist table has been updated. Failure stays isolated from sync/API.
        frs_watchlist_status = "DATABASE_UPDATED_WORKER_POLLING"
        try:
            from services import liveFRS
            if liveFRS.enabled():
                liveFRS._get_client().reload_watchlist()
                frs_watchlist_status = "RELOAD_REQUESTED"
        except Exception as reload_exc:
            frs_watchlist_status = "RELOAD_FAILED_WORKER_POLLING"
            logger.warning("FRS watchlist reload request failed; worker polling remains active | source_id=%s error=%s", source.id, type(reload_exc).__name__)
        source.status = "CONNECTED" if not stats["rejected"] else "DEGRADED"
        source.last_sync_at = indian_time(); source.last_success_at = source.last_sync_at; source.last_error = None
        db.add(ExternalWatchlistSyncHistory(source_id=source.id, actor_user_id=actor_user_id, action=action, result=source.status,
            records_received=stats["received"], records_created=stats["created"], records_updated=stats["updated"], records_unchanged=stats["unchanged"],
            records_rejected=stats["rejected"], duration_ms=int((time.monotonic()-start)*1000), message="Sync completed"))
        db.commit()
        duration = time.monotonic() - start
        if _metrics is not None:
            sid = str(source.id)
            _metrics.external_watchlist_sync_total.labels(result=source.status.lower(), source_id=sid).inc()
            _metrics.external_watchlist_sync_duration_seconds.labels(source_id=sid).observe(duration)
            _metrics.external_watchlist_source_up.labels(source_id=sid).set(1)
            _metrics.external_watchlist_last_success_timestamp.labels(source_id=sid).set(source.last_success_at.timestamp())
            for kind in ("PERSON", "VEHICLE"):
                count = db.query(ExternalWatchlistRecord).filter(ExternalWatchlistRecord.source_id == source.id, ExternalWatchlistRecord.entity_type == kind, ExternalWatchlistRecord.active.is_(True)).count()
                _metrics.external_watchlist_records.labels(source_id=sid, entity_type=kind.lower()).set(count)
        # FRS worker already polls the shared person watchlist table and refreshes its model-compatible index.
        return {**stats, "deactivated": 0, "duration_ms": int(duration*1000), "status": source.status,
                "frs_watchlist_status": frs_watchlist_status, "anpr_watchlist_status": "UPDATED_IN_SHARED_DATABASE"}
    except Exception as exc:
        db.rollback()
        source = db.query(ExternalWatchlistSource).filter(ExternalWatchlistSource.id == source.id).first()
        source.status = "DEGRADED" if source.last_success_at else "ERROR"
        source.last_sync_at = indian_time(); source.last_error = "External source unavailable or invalid response"
        db.add(ExternalWatchlistSyncHistory(source_id=source.id, actor_user_id=actor_user_id, action=action, result=source.status,
            records_received=stats["received"], records_rejected=stats["rejected"], duration_ms=int((time.monotonic()-start)*1000), message=source.last_error))
        db.commit()
        if _metrics is not None:
            sid = str(source.id)
            _metrics.external_watchlist_sync_total.labels(result=source.status.lower(), source_id=sid).inc()
            _metrics.external_watchlist_sync_failures_total.labels(source_id=sid).inc()
            _metrics.external_watchlist_sync_duration_seconds.labels(source_id=sid).observe(time.monotonic() - start)
            _metrics.external_watchlist_source_up.labels(source_id=sid).set(0)
        logger.warning("External watchlist sync failed | source_id=%s reason=%s", source.id, type(exc).__name__)
        return {**stats, "deactivated": 0, "duration_ms": int((time.monotonic()-start)*1000), "status": source.status, "error": source.last_error}
