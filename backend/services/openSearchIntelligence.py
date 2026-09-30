from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from db.model import Alert, Incident
from db.watchlist_model import WatchlistEntry
from db.advanced_intelligence_model import PersonWatchlistEntry, IncidentEvidence

logger = logging.getLogger("opensearch-intelligence")

try:
    from opensearchpy import OpenSearch, helpers
except Exception:  # optional until OPENSEARCH_ENABLED=true
    OpenSearch = None
    helpers = None


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_text(value: Any, limit: int = 8000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


@dataclass(frozen=True)
class SearchHit:
    entity_type: str
    entity_id: str
    score: float
    title: str
    text: str
    href: str | None
    source: dict[str, Any]


class OpenSearchIntelligence:
    """Tenant-filtered search/index layer for INTEL-I.

    Security boundary:
      * no image bytes, biometric embeddings, credentials or raw secrets are indexed;
      * every operational document carries user_id;
      * every operational query applies an exact user_id filter;
      * knowledge documents are from an allowlisted documentation corpus only.
    """

    INDEX_VERSION = "v1"

    def __init__(self) -> None:
        self.enabled = _bool("OPENSEARCH_ENABLED", False)
        self.required = _bool("OPENSEARCH_REQUIRED", False)
        self.url = os.getenv("OPENSEARCH_URL", "https://opensearch:9200").strip().rstrip("/")
        self.username = os.getenv("OPENSEARCH_USERNAME", "").strip()
        self.password = os.getenv("OPENSEARCH_PASSWORD", "")
        self.verify_certs = _bool("OPENSEARCH_VERIFY_CERTS", True)
        self.ca_certs = os.getenv("OPENSEARCH_CA_CERTS", "").strip() or None
        self.index_prefix = re.sub(r"[^a-z0-9_-]", "-", os.getenv("OPENSEARCH_INDEX_PREFIX", "inteli").lower())[:40]
        self.timeout = max(2, min(30, int(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "8"))))
        self.embedding_enabled = _bool("OPENSEARCH_VECTOR_ENABLED", False)
        self.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma").strip()[:120]
        self.embedding_dimension = max(1, min(16000, int(os.getenv("RAG_EMBEDDING_DIMENSION", "768"))))
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        self.max_results = max(1, min(100, int(os.getenv("OPENSEARCH_MAX_RESULTS", "25"))))
        self._client = None
        self._lock = threading.RLock()
        if self.enabled:
            self._validate_config()

    def _validate_config(self) -> None:
        parsed = urlparse(self.url)
        allowed = {x.strip().lower() for x in os.getenv("OPENSEARCH_ALLOWED_HOSTS", "opensearch,127.0.0.1,localhost").split(",") if x.strip()}
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise RuntimeError("OPENSEARCH_URL is invalid")
        if parsed.hostname.lower() not in allowed:
            raise RuntimeError("OPENSEARCH_URL host is not allowlisted")
        if parsed.username or parsed.password:
            raise RuntimeError("credentials in OPENSEARCH_URL are forbidden")
        if os.getenv("ENV", "dev").strip().lower() == "prod" and parsed.scheme != "https":
            raise RuntimeError("production OpenSearch must use HTTPS")
        if OpenSearch is None:
            raise RuntimeError("opensearch-py is required when OPENSEARCH_ENABLED=true")

    def _index(self, kind: str) -> str:
        return f"{self.index_prefix}-{kind}-{self.INDEX_VERSION}"

    def client(self):
        if not self.enabled:
            return None
        with self._lock:
            if self._client is None:
                parsed = urlparse(self.url)
                auth = (self.username, self.password) if self.username else None
                self._client = OpenSearch(
                    hosts=[{"host": parsed.hostname, "port": parsed.port or (443 if parsed.scheme == "https" else 80), "scheme": parsed.scheme}],
                    http_auth=auth,
                    use_ssl=parsed.scheme == "https",
                    verify_certs=self.verify_certs,
                    ca_certs=self.ca_certs,
                    timeout=self.timeout,
                    max_retries=2,
                    retry_on_timeout=True,
                )
            return self._client

    def ensure_indices(self) -> None:
        if not self.enabled:
            return
        client = self.client()
        for kind in ("person-watchlist", "vehicle-watchlist", "evidence", "alerts", "knowledge"):
            name = self._index(kind)
            if client.indices.exists(index=name):
                continue
            properties: dict[str, Any] = {
                "user_id": {"type": "long"},
                "entity_type": {"type": "keyword"},
                "entity_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "category": {"type": "keyword"},
                "status": {"type": "keyword"},
                "camera_id": {"type": "keyword"},
                "plate": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "href": {"type": "keyword", "index": False},
                "metadata": {"type": "object", "enabled": False},
            }
            settings: dict[str, Any] = {"number_of_shards": 1, "number_of_replicas": int(os.getenv("OPENSEARCH_REPLICAS", "0"))}
            if self.embedding_enabled:
                settings["index.knn"] = True
                properties["embedding"] = {"type": "knn_vector", "dimension": self.embedding_dimension, "space_type": "cosinesimil"}
            client.indices.create(index=name, body={"settings": settings, "mappings": {"dynamic": "strict", "properties": properties}})

    def _embed(self, text: str) -> list[float] | None:
        if not self.embedding_enabled:
            return None
        parsed = urlparse(self.ollama_base_url)
        allowed = {x.strip().lower() for x in os.getenv("OLLAMA_ALLOWED_HOSTS", "127.0.0.1,localhost,ollama").split(",") if x.strip()}
        if parsed.hostname is None or parsed.hostname.lower() not in allowed:
            raise RuntimeError("Ollama embedding host is not allowlisted")
        with httpx.Client(timeout=min(30, self.timeout * 3), follow_redirects=False) as client:
            res = client.post(f"{self.ollama_base_url}/api/embed", json={"model": self.embedding_model, "input": text[:8000]})
            res.raise_for_status()
            vectors = res.json().get("embeddings") or []
        if not vectors or not isinstance(vectors[0], list):
            raise RuntimeError("embedding response missing vector")
        vector = [float(x) for x in vectors[0]]
        if len(vector) != self.embedding_dimension:
            raise RuntimeError(f"embedding dimension mismatch configured={self.embedding_dimension} actual={len(vector)}")
        return vector

    def _doc(self, *, user_id: int, entity_type: str, entity_id: Any, title: str, text: str, href: str | None = None, **fields: Any) -> dict[str, Any]:
        body = {
            "user_id": int(user_id),
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "title": _safe_text(title, 1000),
            "text": _safe_text(text, 8000),
            "category": _safe_text(fields.pop("category", ""), 200) or None,
            "status": _safe_text(fields.pop("status", ""), 100) or None,
            "camera_id": _safe_text(fields.pop("camera_id", ""), 150) or None,
            "plate": _safe_text(fields.pop("plate", ""), 80) or None,
            "created_at": fields.pop("created_at", None),
            "updated_at": fields.pop("updated_at", None),
            "href": href,
            "metadata": fields or {},
        }
        if self.embedding_enabled:
            try:
                body["embedding"] = self._embed(f"{body['title']}\n{body['text']}")
            except (httpx.RequestError, httpx.HTTPStatusError):
                if _bool("OPENSEARCH_VECTOR_REQUIRED", False):
                    raise
                # Text indexing must not block durable alert delivery when the
                # optional embedding server is temporarily unavailable.
                logger.warning("Embedding service unavailable; indexing document without vector", exc_info=True)
        return body

    def sync_user(self, db: Session, user_id: int) -> dict[str, int]:
        if not self.enabled:
            return {"indexed": 0}
        self.ensure_indices()
        uid = int(user_id)
        actions: list[dict[str, Any]] = []

        for row in db.query(PersonWatchlistEntry).filter(PersonWatchlistEntry.user_id == uid).all():
            text = " ".join(filter(None, [row.full_name, row.category, row.status, row.description, row.source, row.external_reference]))
            actions.append({"_op_type": "index", "_index": self._index("person-watchlist"), "_id": f"{uid}:{row.id}", "_source": self._doc(user_id=uid, entity_type="person_watchlist", entity_id=row.id, title=row.full_name, text=text, category=row.category, status=row.status, created_at=_iso(row.created_at), updated_at=_iso(row.updated_at), href="/person-watchlist")})

        for row in db.query(WatchlistEntry).filter(WatchlistEntry.user_id == uid).all():
            text = " ".join(filter(None, [row.plate_display, row.plate_normalized, row.category, row.status, row.priority, row.description, row.source]))
            actions.append({"_op_type": "index", "_index": self._index("vehicle-watchlist"), "_id": f"{uid}:{row.id}", "_source": self._doc(user_id=uid, entity_type="vehicle_watchlist", entity_id=row.id, title=f"Vehicle watchlist {row.plate_display}", text=text, category=row.category, status=row.status, plate=row.plate_normalized, created_at=_iso(row.created_at), updated_at=_iso(row.updated_at), href="/watchlist")})

        for row in db.query(Alert).filter(Alert.user_id == uid).all():
            text = " ".join(filter(None, [row.alert_type, row.level, row.rule, row.cam_id, row.plate, row.watchlist_category, row.watchlist_status, row.track_id]))
            actions.append({"_op_type": "index", "_index": self._index("alerts"), "_id": f"{uid}:{row.id}", "_source": self._doc(user_id=uid, entity_type="alert", entity_id=row.id, title=f"Alert {row.id}: {row.rule}", text=text, category=row.watchlist_category, status=row.watchlist_status, camera_id=row.cam_id, plate=row.plate, created_at=_iso(row.created_at), updated_at=_iso(row.created_at), href=f"/alert-history?alert_id={row.id}")})

        for row in db.query(IncidentEvidence).filter(IncidentEvidence.user_id == uid).all():
            metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
            text = " ".join(filter(None, [row.evidence_type, row.object_type, row.object_reference, row.description, json.dumps(metadata, ensure_ascii=True, default=str)[:3000]]))
            actions.append({"_op_type": "index", "_index": self._index("evidence"), "_id": f"{uid}:{row.id}", "_source": self._doc(user_id=uid, entity_type="evidence", entity_id=row.id, title=f"Evidence {row.id}: {row.evidence_type}", text=text, status="", created_at=_iso(row.created_at), updated_at=_iso(row.created_at), href=f"/incidents?incident_id={row.incident_id}", incident_id=row.incident_id, alert_id=row.alert_id, snapshot_id=row.snapshot_id, object_type=row.object_type, object_reference=row.object_reference)})

        if actions:
            helpers.bulk(self.client(), actions, chunk_size=250, request_timeout=max(30, self.timeout))
        return {"indexed": len(actions)}


    def index_alert_by_id(self, db: Session, *, user_id: int, alert_id: int) -> bool:
        """Idempotently index one persisted alert for durable event-driven search."""
        if not self.enabled:
            return False
        self.ensure_indices()
        uid = int(user_id)
        row = db.query(Alert).filter(Alert.id == int(alert_id), Alert.user_id == uid).first()
        if row is None:
            return False
        text = " ".join(filter(None, [row.alert_type, row.level, row.rule, row.cam_id, row.plate, row.watchlist_category, row.watchlist_status, row.track_id]))
        body = self._doc(
            user_id=uid, entity_type="alert", entity_id=row.id,
            title=f"Alert {row.id}: {row.rule}", text=text,
            category=row.watchlist_category, status=row.watchlist_status,
            camera_id=row.cam_id, plate=row.plate, created_at=_iso(row.created_at),
            updated_at=_iso(row.created_at), href=f"/alert-history?alert_id={row.id}",
        )
        self.client().index(index=self._index("alerts"), id=f"{uid}:{row.id}", body=body, refresh=False)
        return True

    def index_knowledge(self, root: Path) -> dict[str, int]:
        if not self.enabled:
            return {"indexed": 0}
        self.ensure_indices()
        allow_roots = [root / "docs"]
        root_files = [p for p in root.glob("*.md") if p.is_file()]
        files: list[Path] = root_files
        for base in allow_roots:
            if base.exists():
                files.extend(p for p in base.rglob("*.md") if p.is_file())
        actions = []
        for path in sorted(set(files)):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(raw) > 2_000_000:
                continue
            chunks = self._chunks(raw)
            rel = path.relative_to(root).as_posix()
            for i, chunk in enumerate(chunks):
                actions.append({"_op_type": "index", "_index": self._index("knowledge"), "_id": f"{rel}:{i}", "_source": self._doc(user_id=0, entity_type="knowledge", entity_id=f"{rel}:{i}", title=rel, text=chunk, href=None, source_path=rel, chunk=i)})
        if actions:
            helpers.bulk(self.client(), actions, chunk_size=100, request_timeout=max(30, self.timeout))
        return {"indexed": len(actions)}

    @staticmethod
    def _chunks(text: str, max_chars: int = 3500, overlap: int = 300) -> list[str]:
        clean = text.replace("\x00", "").strip()
        if not clean:
            return []
        out = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + max_chars)
            chunk = clean[start:end].strip()
            if chunk:
                out.append(chunk)
            if end >= len(clean):
                break
            start = max(start + 1, end - overlap)
        return out[:5000]

    def search(self, *, user_id: int, query: str, kinds: Iterable[str] | None = None, limit: int = 10, include_knowledge: bool = False) -> list[SearchHit]:
        if not self.enabled:
            return []
        self.ensure_indices()
        q = _safe_text(query, 1200)
        if not q:
            return []
        valid = {"person-watchlist", "vehicle-watchlist", "evidence", "alerts"}
        selected = [k for k in (kinds or valid) if k in valid]
        if include_knowledge:
            selected.append("knowledge")
        indices = [self._index(k) for k in selected]
        n = max(1, min(self.max_results, int(limit)))

        should: list[dict[str, Any]] = [{"multi_match": {"query": q, "fields": ["title^3", "text", "plate^4", "camera_id^2", "category^2", "status"], "type": "best_fields", "fuzziness": "AUTO"}}]
        if self.embedding_enabled:
            vector = self._embed(q)
            should.append({"knn": {"embedding": {"vector": vector, "k": max(n, 10)}}})

        body = {
            "size": n,
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                    "filter": [{"terms": {"user_id": [0, int(user_id)] if include_knowledge else [int(user_id)]}}],
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        response = self.client().search(index=",".join(indices), body=body)
        hits: list[SearchHit] = []
        for item in response.get("hits", {}).get("hits", []):
            src = item.get("_source") or {}
            hits.append(SearchHit(entity_type=str(src.get("entity_type") or ""), entity_id=str(src.get("entity_id") or ""), score=float(item.get("_score") or 0.0), title=str(src.get("title") or ""), text=str(src.get("text") or ""), href=src.get("href"), source=src))
        return hits

    def health(self) -> dict[str, Any]:
        base = {"enabled": self.enabled, "required": self.required, "vector_enabled": self.embedding_enabled, "embedding_model": self.embedding_model if self.embedding_enabled else None}
        if not self.enabled:
            return {**base, "status": "DISABLED"}
        try:
            client = self.client()
            info = client.cluster.health()
            return {**base, "status": "READY", "cluster_status": info.get("status"), "cluster_name": info.get("cluster_name")}
        except Exception as exc:
            logger.warning("OpenSearch health failed: %s", exc)
            return {**base, "status": "ERROR", "error": type(exc).__name__}


opensearch_intelligence = OpenSearchIntelligence()
