# INTEL-I backend production audit

## Verified
- 47 Python source files were syntax-compiled successfully after updates.
- `main.py` now uses user-id subjects consistently during refresh rotation.
- Refresh-token rotation locks the existing token row where supported by the DB.
- Production schema creation is disabled unless explicitly enabled outside prod.
- CORS origin is validated as a single http/https origin.
- Security response headers are added.
- Upload limits are bounded and upload MIME headers are treated as untrusted.
- Stream-cookie names are sanitized and stream types are constrained.
- Startup/shutdown handling is defensive.
- The hardened vehicle tracking, Re-ID, ANPR, correlation, state, and persistence
  modules from the previous audit are included.
- Real `.env` secrets from the uploaded archive are excluded from the deliverable.
- A sanitized `.env.example` is included.

## Critical security action
The uploaded `.env` contained real-looking credentials/secrets. Assume them
compromised and rotate/revoke them before using any deployment.

## Architecture limitation that remains
The FastAPI process still owns the camera-worker executor and GPU model
instances. This is functional for a single-node deployment but for true
horizontal production scale the next architectural step is to move camera
ingestion/AI inference into dedicated worker processes and use Redis/Kafka for
shared runtime state/events.

## Database
Run Alembic migrations before starting the production application. Do not rely
on SQLAlchemy `create_all()` in production.

## Evidence model
The vehicle pipeline keeps detection/tracking/Re-ID/ANPR/GIS/correlation
evidence separate. A plate or a single Re-ID score is not treated as a
permanent global identity by itself.

## Dynamic digital-evidence PDF export (2026-09-08)

Implemented a tenant-scoped incident evidence report pipeline under `backend/evidence/` with Gujarat Police branding, server-controlled classification/watermark policy, dynamic incident/timeline/journey/exhibit sections, source snapshot SHA-256 verification, source-manifest SHA-256, final watermarked PDF SHA-256, bounded image decoding, XML-safe report text, fail-closed export auditing, no-store response headers and Incident Center download integration. The generator does not alter original CCTV/evidence media. Deployment acceptance still requires running the API against the target PostgreSQL/PostGIS instance and validating representative incident types and evidence images.
