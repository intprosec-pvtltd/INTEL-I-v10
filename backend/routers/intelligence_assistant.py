from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth.auth import get_current_user
from db.database import getDB
from db.model import User
from services.intelligenceAssistant import intelligence_assistant

router = APIRouter(prefix="/api/intelligence-assistant", tags=["INTEL-I Intelligence Assistant"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1200)


@router.post("/chat")
def chat(payload: ChatRequest, db: Session = Depends(getDB), current_user: User = Depends(get_current_user)):
    try:
        return intelligence_assistant.answer(db=db, user_id=int(current_user.id), question=payload.question)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid intelligence question")
    except RuntimeError as exc:
        code = str(exc)
        if code == "RATE_LIMITED":
            raise HTTPException(status_code=429, detail="Intelligence assistant rate limit exceeded")
        if code == "ASSISTANT_DISABLED":
            raise HTTPException(status_code=503, detail="Intelligence assistant is disabled")
        raise HTTPException(status_code=503, detail="Intelligence assistant temporarily unavailable")


@router.get("/health")
def health(current_user: User = Depends(get_current_user)):
    return intelligence_assistant.health()


@router.get("/suggestions")
def suggestions(current_user: User = Depends(get_current_user)):
    return {"suggestions": [
        "What is happening now?", "Show newest alerts", "Show high-priority alerts",
        "Which cameras are offline?", "What happened in the last hour?",
        "Show recent watchlist matches", "Is the AI system healthy?",
    ]}
