from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from db.database import getDB
from db.model import User
from db.watchlist_model import Watchlist, WatchlistEntry, indian_time
from auth.auth import get_current_user
from schemas.watchlist import WatchlistCreate, WatchlistUpdate, WatchlistEntryCreate, WatchlistEntryUpdate, WatchlistMatchRequest
from services.watchlist_service import normalize_plate, find_plate_matches, serialize_entry

router = APIRouter(prefix='/api/watchlist', tags=['Watchlist'])


def _watchlist_or_404(db, user_id, watchlist_id):
    obj = db.query(Watchlist).filter(Watchlist.id == watchlist_id, Watchlist.user_id == user_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail='Watchlist not found')
    return obj


def _entry_or_404(db, user_id, entry_id):
    obj = db.query(WatchlistEntry).filter(WatchlistEntry.id == entry_id, WatchlistEntry.user_id == user_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail='Watchlist entry not found')
    return obj

@router.get('')
def list_watchlists(current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    rows = db.query(Watchlist).filter(Watchlist.user_id == current_user.id).order_by(Watchlist.updated_at.desc()).all()
    return {'watchlists': [{'id': int(x.id), 'name': x.name, 'description': x.description, 'is_active': x.is_active, 'entry_count': len(x.entries), 'created_at': x.created_at.isoformat(), 'updated_at': x.updated_at.isoformat()} for x in rows]}

@router.post('')
def create_watchlist(payload: WatchlistCreate, current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    obj = Watchlist(user_id=current_user.id, name=payload.name.strip(), description=payload.description, is_active=payload.is_active)
    db.add(obj); db.commit(); db.refresh(obj)
    return {'watchlist': {'id': int(obj.id), 'name': obj.name, 'description': obj.description, 'is_active': obj.is_active, 'entry_count': 0}}

@router.patch('/{watchlist_id}')
def update_watchlist(watchlist_id: int, payload: WatchlistUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    obj = _watchlist_or_404(db, current_user.id, watchlist_id)
    data = payload.model_dump(exclude_unset=True)
    if 'name' in data: obj.name = data['name'].strip()
    if 'description' in data: obj.description = data['description']
    if 'is_active' in data: obj.is_active = data['is_active']
    obj.updated_at = indian_time()
    db.commit(); db.refresh(obj)
    return {'watchlist': {'id': int(obj.id), 'name': obj.name, 'description': obj.description, 'is_active': obj.is_active, 'entry_count': len(obj.entries)}}

@router.delete('/{watchlist_id}')
def delete_watchlist(watchlist_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    obj = _watchlist_or_404(db, current_user.id, watchlist_id)
    db.delete(obj); db.commit()
    return {'success': True}

@router.get('/{watchlist_id}/entries')
def list_entries(watchlist_id: int, status: str | None = Query(default=None), category: str | None = Query(default=None), search: str | None = Query(default=None), current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    _watchlist_or_404(db, current_user.id, watchlist_id)
    q = db.query(WatchlistEntry).filter(WatchlistEntry.watchlist_id == watchlist_id, WatchlistEntry.user_id == current_user.id)
    if status: q = q.filter(WatchlistEntry.status == status.upper())
    if category: q = q.filter(WatchlistEntry.category == category.upper())
    if search:
        normalized = normalize_plate(search)
        q = q.filter(WatchlistEntry.plate_normalized.ilike(f'%{normalized}%'))
    rows = q.order_by(WatchlistEntry.updated_at.desc()).limit(1000).all()
    return {'entries': [serialize_entry(x) for x in rows]}

@router.post('/{watchlist_id}/entries')
def create_entry(watchlist_id: int, payload: WatchlistEntryCreate, current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    wl = _watchlist_or_404(db, current_user.id, watchlist_id)
    normalized = normalize_plate(payload.plate)
    duplicate = db.query(WatchlistEntry).filter(WatchlistEntry.user_id == current_user.id, WatchlistEntry.plate_normalized == normalized, WatchlistEntry.status == 'ACTIVE').first()
    if duplicate:
        raise HTTPException(status_code=409, detail='An active watchlist entry already exists for this plate')
    obj = WatchlistEntry(
        watchlist_id=wl.id, user_id=current_user.id, plate_normalized=normalized, plate_display=payload.plate.strip().upper(),
        category=payload.category, status=payload.status, priority=payload.priority, description=payload.description,
        source=payload.source, effective_from=payload.effective_from, effective_until=payload.effective_until,
        version=1, metadata_json=payload.metadata,
    )
    db.add(obj); wl.updated_at = indian_time()
    try:
        db.commit(); db.refresh(obj)
    except IntegrityError:
        db.rollback(); raise HTTPException(status_code=409, detail='Duplicate watchlist entry')
    return {'entry': serialize_entry(obj)}

@router.patch('/entries/{entry_id}')
def update_entry(entry_id: int, payload: WatchlistEntryUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    obj = _entry_or_404(db, current_user.id, entry_id)
    data = payload.model_dump(exclude_unset=True)
    if 'plate' in data and data['plate'] is not None:
        normalized = normalize_plate(data['plate'])
        duplicate = db.query(WatchlistEntry).filter(WatchlistEntry.user_id == current_user.id, WatchlistEntry.plate_normalized == normalized, WatchlistEntry.id != obj.id, WatchlistEntry.status == 'ACTIVE').first()
        if duplicate: raise HTTPException(status_code=409, detail='Another active entry already uses this plate')
        obj.plate_normalized = normalized; obj.plate_display = data['plate'].strip().upper()
    for key in ('category','status','priority','description','source','effective_from','effective_until','metadata'):
        if key in data:
            setattr(obj, 'metadata_json' if key == 'metadata' else key, data[key])
    obj.version += 1; obj.updated_at = indian_time()
    db.commit(); db.refresh(obj)
    wl = db.query(Watchlist).filter(Watchlist.id == obj.watchlist_id).first()
    if wl: wl.updated_at = indian_time(); db.commit()
    return {'entry': serialize_entry(obj)}

@router.delete('/entries/{entry_id}')
def delete_entry(entry_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    obj = _entry_or_404(db, current_user.id, entry_id)
    wl_id = obj.watchlist_id
    db.delete(obj); db.commit()
    wl = db.query(Watchlist).filter(Watchlist.id == wl_id).first()
    if wl: wl.updated_at = indian_time(); db.commit()
    return {'success': True}


@router.post('/match')
def match_plate(payload: WatchlistMatchRequest, current_user: User = Depends(get_current_user), db: Session = Depends(getDB)):
    result = find_plate_matches(db, current_user.id, payload.plate, payload.include_possible)
    return {
        'plate': result['plate'],
        'plate_normalized': result['plate_normalized'],
        'exact_matches': [{'entry': serialize_entry(e), 'match_type': 'EXACT', 'confidence': score} for e, score in result['exact']],
        'possible_matches': [{'entry': serialize_entry(e), 'match_type': 'POSSIBLE', 'confidence': score} for e, score in result['possible']],
    }
