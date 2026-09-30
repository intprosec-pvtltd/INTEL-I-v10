from fastapi import APIRouter, Depends, HTTPException
from auth.auth import get_current_user
from db.model import User
from services.runtimeModelReadiness import cached_model_readiness, verify_models

router = APIRouter(prefix="/api/models", tags=["Model Deployment"])

@router.get("/readiness")
def readiness(current_user: User = Depends(get_current_user)):
    state = cached_model_readiness()
    if state.get("status") == "NOT_VERIFIED":
        state = verify_models()
    return state

@router.post("/verify")
def verify(current_user: User = Depends(get_current_user)):
    # Authenticated on-demand verification; deliberately not public because it
    # can consume GPU/CPU for several seconds.
    return verify_models(force=True)
