# INTEL-I — Phase 0 to Phase 3 Implementation

This package is an updated version of the supplied backend/frontend source for the Phase 0–3 target from the INTEL-I architecture and production-risk documents.

## Implemented

### Phase 0 — Architecture
- Added a canonical camera contract independent of source type.
- Added explicit architecture/ownership contract in `docs/PHASE_0_3_PRODUCTION_CONTRACT.md`.
- Preserved existing AI/media architecture instead of rewriting it.

### Phase 1 — Infrastructure
- Startup readiness now distinguishes process liveness from operational readiness.
- READY requires PostgreSQL + Redis + Kafka to be healthy.
- Added infrastructure Docker Compose for PostgreSQL, Redis and Kafka.
- Added `.env.example`; production secrets are not included.
- Added Alembic to Python requirements.

### Phase 2 — Database
- Added persistent normalized camera stream metadata.
- Canonical camera lifecycle state is persisted as `ONLINE`, `DEGRADED`, `RECONNECTING`, `OFFLINE`, `AUTHENTICATION_FAILED`, `TIMEOUT`, or `AI_DISABLED`.
- Added `camera_health`, `evidence_hashes`, `audit_logs`, `system_events`, and `watchlist_versions` tables.
- Added migration `b7e4c9d1a2f3_complete_phase0_3_camera_contract.py`.
- Added migration `c1d7e9f3a5b2_phase0_3_operational_core.py`.
- Added camera create/start/stop/delete audit records.
- Added camera lifecycle health records.
- Added backup/restore scripts.

### Phase 3 — Camera Integration
- Normalized source types include RTSP, ONVIF, NVR, VMS, recorded media, HTTP/HTTPS/HLS, webcam and vendor connectors.
- NVR/VMS use the existing vendor API connector contract.
- Added persistent direction, stream FPS, dimensions, transport and codec metadata.
- Added `/camera/{camID}` canonical camera details endpoint.
- `/cameras` now returns a non-secret canonical camera contract.
- Backend health inventory uses persistent camera records plus runtime state.
- Frontend camera health uses backend-authoritative lifecycle state and refreshes every 5 seconds.
- Frontend onboarding exposes normalized stream/topology metadata.

## Security
- Plaintext camera source is never returned by the camera APIs.
- Redis runtime state stores only camera identity, state and non-secret metadata.
- Connector credentials remain encrypted in the database.
- The supplied `.env` file is intentionally excluded from this package.
- `node_modules` is intentionally excluded from this package.

## Verification performed
- Python `compileall`: PASS.
- Phase 0–3 contract unit tests: PASS.
- Alembic migration graph: single head `c1d7e9f3a5b2`.
- Infrastructure Compose YAML parsing: PASS.

## Remaining environment-dependent verification
The final database migration and live frontend build require the target machine's installed PostgreSQL credentials and Node dependencies. The build container did not have a complete npm dependency installation available, so the frontend build was not falsely marked as passed.

## Run

Backend:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Fill real values in .env
alembic upgrade head
uvicorn main:app --host 0.0.0.0 --port 3000
```

Frontend:

```powershell
cd frontend
npm ci
npm run build
npm run dev
```

For local Vite development, leave `VITE_BASE_URL` empty so the configured Vite proxy can keep API and stream requests same-origin from the browser.
