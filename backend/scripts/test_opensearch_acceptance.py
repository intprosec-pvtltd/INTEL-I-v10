from __future__ import annotations
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from db.database import SessionLocal
from services.openSearchIntelligence import opensearch_intelligence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--query", default="alert watchlist evidence")
    args = ap.parse_args()

    print("HEALTH:", opensearch_intelligence.health())
    if not opensearch_intelligence.enabled:
        raise SystemExit("FAIL: OPENSEARCH_ENABLED is false")
    opensearch_intelligence.ensure_indices()
    db = SessionLocal()
    try:
        print("SYNC:", opensearch_intelligence.sync_user(db, args.user_id))
        print("KNOWLEDGE:", opensearch_intelligence.index_knowledge(ROOT))
        hits = opensearch_intelligence.search(user_id=args.user_id, query=args.query, limit=10, include_knowledge=True)
        for h in hits:
            print(f"HIT type={h.entity_type} id={h.entity_id} score={h.score:.4f} href={h.href}")
        print("PASS: OpenSearch indexing/search acceptance completed")
    finally:
        db.close()

if __name__ == "__main__":
    main()
