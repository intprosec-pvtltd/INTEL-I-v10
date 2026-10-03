from __future__ import annotations
import os
import threading
from datetime import datetime, timedelta
from db.database import SessionLocal
from db.external_watchlist_model import ExternalWatchlistSource
from integrations.government_watchlist.service import sync_source

_stop = threading.Event()
_thread = None
_lock = threading.Lock()


def _loop():
    while not _stop.wait(15):
        try:
            now = datetime.utcnow()
            with SessionLocal() as db:
                sources = db.query(ExternalWatchlistSource).filter(ExternalWatchlistSource.enabled.is_(True), ExternalWatchlistSource.sync_mode == "INTERVAL").all()
                for source in sources:
                    due = source.last_sync_at is None or source.last_sync_at <= now - timedelta(seconds=max(30, source.sync_interval_seconds or 300))
                    if due:
                        sync_source(db, source, source.created_by or source.owner_user_id, action="SCHEDULED")
        except Exception:
            # Connector/scheduler failure must never affect the camera/API process.
            import logging
            logging.getLogger("government-watchlist-scheduler").exception("Scheduled external watchlist sync iteration failed")


def start_scheduler():
    global _thread
    if os.getenv("EXTERNAL_WATCHLIST_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    with _lock:
        if _thread and _thread.is_alive():
            return
        _stop.clear(); _thread = threading.Thread(target=_loop, name="government-watchlist-sync", daemon=True); _thread.start()


def stop_scheduler():
    _stop.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=3)
