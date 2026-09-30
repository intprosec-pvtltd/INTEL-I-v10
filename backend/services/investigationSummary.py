from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from db.database import SessionLocal
from db.intelligence_model import InvestigationSummary

logger = logging.getLogger("investigation-summary")


class InvestigationSummaryWorker:
    def __init__(self):
        self.enabled = os.getenv("OLLAMA_SUMMARY_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        self.model = os.getenv("OLLAMA_SUMMARY_MODEL", "qwen2.5:7b").strip()[:150]
        self.timeout = max(5.0, min(120.0, float(os.getenv("OLLAMA_SUMMARY_TIMEOUT_SECONDS", "45"))))
        parsed = urlparse(self.base_url)
        allowed = {x.strip().lower() for x in os.getenv("OLLAMA_ALLOWED_HOSTS", "127.0.0.1,localhost,ollama").split(",") if x.strip()}
        if self.enabled and (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.lower() not in allowed):
            raise RuntimeError("OLLAMA_BASE_URL host is not allowed")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ollama-summary")

    def submit(self, summary_id: int, evidence: dict) -> bool:
        if not self.enabled:
            return False
        self._executor.submit(self._run, int(summary_id), evidence)
        return True

    def _run(self, summary_id: int, evidence: dict):
        db = SessionLocal()
        try:
            row = db.query(InvestigationSummary).filter(InvestigationSummary.id == summary_id).first()
            if row is None or row.status != "PENDING":
                return
            prompt = (
                "You are an evidence summarizer for a CCTV investigation system. "
                "Use only supplied facts. Do not infer identity, guilt, intent, unseen route, or certainty. "
                "Call paths observed journeys across integrated CCTV coverage. Return strict JSON: "
                "{summary, key_observations, confidence_notes, limitations}. Evidence: "
                + json.dumps(evidence, ensure_ascii=True, default=str)[:24000]
            )
            with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
                response = client.post(f"{self.base_url}/api/chat", json={"model": self.model, "stream": False, "format": "json", "messages": [{"role": "user", "content": prompt}], "options": {"temperature": 0, "num_predict": 900}})
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "")
            if not isinstance(content, str) or len(content) > 30000:
                raise ValueError("invalid_summary_response")
            parsed = json.loads(content)
            summary = str(parsed.get("summary") or "").strip()[:12000]
            if not summary:
                raise ValueError("empty_summary")
            row.summary_text = summary
            row.structured_summary = {
                "key_observations": list(parsed.get("key_observations") or [])[:100],
                "confidence_notes": list(parsed.get("confidence_notes") or [])[:50],
                "limitations": list(parsed.get("limitations") or [])[:50],
            }
            row.status = "COMPLETED"
            row.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            db.commit()
        except Exception as exc:
            db.rollback()
            row = db.query(InvestigationSummary).filter(InvestigationSummary.id == summary_id).first()
            if row:
                row.status = "FAILED"; row.error_code = type(exc).__name__[:100]; row.completed_at = datetime.now(timezone.utc).replace(tzinfo=None); db.commit()
            logger.warning("Investigation summary failed | id=%s error=%s", summary_id, type(exc).__name__)
        finally:
            db.close()

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


investigation_summary_worker = InvestigationSummaryWorker()
