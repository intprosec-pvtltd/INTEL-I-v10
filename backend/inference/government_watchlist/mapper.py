from __future__ import annotations
import re
from typing import Any

DEFAULT_MAPPING = {
    "external_id": "id", "entity_type": "entity_type", "category": "category",
    "active": "active", "name": "name", "plate": "registration_number",
    "priority": "priority", "case_reference": "case_reference", "image_data": "image_base64",
    "image_content_type": "image_content_type", "make": "make", "model": "model", "color": "color",
    "description": "description",
}

def get_path(value: Any, path: str) -> Any:
    current = value
    for token in str(path or "").split("."):
        if not token or not isinstance(current, dict):
            return None
        current = current.get(token)
    return current

def normalize_record(raw: dict, field_mapping: dict) -> dict:
    mapping = {**DEFAULT_MAPPING, **(field_mapping or {})}
    row = {key: get_path(raw, path) for key, path in mapping.items() if path}
    external_id = str(row.get("external_id") or "").strip()
    if not external_id or len(external_id) > 200:
        raise ValueError("Record is missing a valid external ID")
    entity = str(row.get("entity_type") or "").strip().upper()
    if entity in {"PERSON", "PEOPLE"}:
        entity = "PERSON"
    elif entity in {"VEHICLE", "VEHICLES"}:
        entity = "VEHICLE"
    else:
        raise ValueError("Entity type must be PERSON or VEHICLE")
    category = re.sub(r"[^A-Z0-9_]+", "_", str(row.get("category") or "OTHER").strip().upper())[:50]
    active = row.get("active", True)
    if isinstance(active, str):
        active = active.strip().lower() in {"true", "1", "yes", "active", "wanted", "missing", "stolen"}
    if entity == "PERSON" and not str(row.get("name") or "").strip():
        raise ValueError("Person record is missing a name")
    plate = re.sub(r"[^A-Z0-9]", "", str(row.get("plate") or "").upper()) if row.get("plate") is not None else ""
    if entity == "VEHICLE" and not plate:
        raise ValueError("Vehicle record is missing a registration number")
    return {**row, "external_id": external_id, "entity_type": entity, "category": category,
            "active": bool(active), "name": str(row.get("name") or "").strip()[:150] or None,
            "plate": plate[:50] or None, "priority": str(row.get("priority") or "").upper()[:20] or None,
            "case_reference": str(row.get("case_reference") or "")[:200] or None,
            "make": str(row.get("make") or "")[:100] or None,
            "model": str(row.get("model") or "")[:100] or None,
            "color": str(row.get("color") or "")[:60] or None,
            "description": str(row.get("description") or "")[:2000] or None}
