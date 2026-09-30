from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
from auth.auth import get_current_user
from db.database import getDB
from db.model import AuditLog, User
from security.rbac import ADMIN_ROLES, normalize_role
from connectors.government import government_connectors
from connectors.government.base import GovernmentConnectorError, GovernmentConnectorNotConfigured

router = APIRouter(prefix="/api/government-data", tags=["government-data"])

class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: dict[str, Any]
    purpose: str = Field(min_length=5, max_length=250)


def _authorized(user: User = Depends(get_current_user)) -> User:
    # Tight default. Deployments can later introduce a dedicated investigator role.
    if normalize_role(user.role) not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Government data access is restricted")
    return user

@router.get("/readiness")
def readiness(user: User = Depends(_authorized)):
    return {name: connector.readiness() for name, connector in government_connectors.items()}

@router.post("/{provider}/search")
def search(provider: str, payload: Query, db: Session = Depends(getDB), user: User = Depends(_authorized)):
    connector = government_connectors.get(provider.upper())
    if connector is None:
        raise HTTPException(status_code=404, detail="Unsupported government data provider")
    try:
        result = connector.search(criteria=payload.criteria, purpose=payload.purpose, actor_user_id=int(user.id))
    except GovernmentConnectorNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GovernmentConnectorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    db.add(AuditLog(user_id=int(user.id), action="GOVERNMENT_DATA_QUERY", resource_type="government_connector", resource_id=result.provider, details={"request_id": result.request_id, "purpose": payload.purpose, "record_count": len(result.records), "latency_ms": round(result.latency_ms, 2)}))
    db.commit()
    return {"provider": result.provider, "request_id": result.request_id, "record_count": len(result.records), "records": result.records, "latency_ms": round(result.latency_ms, 2)}
