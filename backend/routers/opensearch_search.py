from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from db.database import getDB
from db.model import User
from services.openSearchIntelligence import opensearch_intelligence

router = APIRouter(prefix="/api/search", tags=["INTEL-I Search"])
BACKEND_ROOT = Path(__file__).resolve().parents[1]


class ReindexRequest(BaseModel):
    include_knowledge: bool = True


@router.get("")
def search(
    q: str = Query(min_length=1, max_length=1200),
    types: str = Query(default="person-watchlist,vehicle-watchlist,evidence,alerts"),
    limit: int = Query(default=20, ge=1, le=100),
    include_knowledge: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
):
    if not opensearch_intelligence.enabled:
        raise HTTPException(status_code=503, detail="OpenSearch is disabled")
    kinds = [x.strip() for x in types.split(",") if x.strip()]
    try:
        hits = opensearch_intelligence.search(user_id=int(current_user.id), query=q, kinds=kinds, limit=limit, include_knowledge=include_knowledge)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Search service temporarily unavailable") from exc
    return {
        "query": q,
        "count": len(hits),
        "results": [
            {
                "type": h.entity_type,
                "id": h.entity_id,
                "score": h.score,
                "title": h.title,
                "text": h.text,
                "href": h.href,
            }
            for h in hits
        ],
    }


@router.post("/reindex")
def reindex(
    payload: ReindexRequest,
    db: Session = Depends(getDB),
    current_user: User = Depends(get_current_user),
):
    if not opensearch_intelligence.enabled:
        raise HTTPException(status_code=503, detail="OpenSearch is disabled")
    try:
        operational = opensearch_intelligence.sync_user(db, int(current_user.id))
        knowledge = opensearch_intelligence.index_knowledge(BACKEND_ROOT) if payload.include_knowledge else {"indexed": 0}
        return {"operational": operational, "knowledge": knowledge}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Reindex failed") from exc


@router.get("/health")
def health(current_user: User = Depends(get_current_user)):
    return opensearch_intelligence.health()
