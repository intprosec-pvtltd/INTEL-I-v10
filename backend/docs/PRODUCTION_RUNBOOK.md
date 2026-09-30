# INTEL-I production runbook

## Deployment sequence

1. Provision PostgreSQL, Redis, Kafka, object/evidence storage and an NVIDIA GPU
   host with the CUDA runtime expected by the pinned PyTorch build.
2. Copy `.env.example` to the secret manager/deployment environment and replace
   every placeholder. Keep Fernet keys stable across releases or encrypted
   camera/watchlist data becomes unreadable.
3. Mount model weights read-only under `/app/models` and persistent evidence
   storage at the configured snapshot/upload paths.
4. Run `alembic upgrade head` before switching application traffic.
5. Start the API with the provided container entrypoint. Confirm `/health`,
   `/api/system/health`, Redis, Kafka, database and GPU telemetry.
6. Configure the Sentinel/VMS integration in the administration UI. Discover,
   sync and verify that no stream address appears in browser/API responses.
7. Start selected cameras, verify `SOURCE_PTS`/`PTS_ALIGNED` timing in camera
   health, then exercise detection, ANPR, watchlist and evidence capture.
8. Export both PDF and CSV reports and archive the generated report ID plus the
   audit record.

## Release verification

```bash
python -m compileall -q .
python -m pytest -q
alembic upgrade head
```

Frontend:

```bash
npm ci
npm run lint
npm run build
```

Use clean Linux dependencies for the frontend build. A `node_modules` folder
copied from Windows does not contain the Linux native Rolldown binding.

## Rollback

Keep the previous application image and database backup. Stop ingestion, drain
the AI queue, switch traffic to the previous image, then apply the database
downgrade only after confirming that no new integration/timing fields must be
retained. Never rotate encryption keys as part of application rollback.

## Operational alarms

Alert on repeated catalogue sync failures, stale `last_sync_at`, camera states
remaining `RECONNECTING`, decoder failures, missing source PTS, discontinuity
spikes, AI queue drops, GPU memory pressure, evidence write failures and outbox
delivery retries. Treat TLS verification disablement as a security exception.

## Demonstration truth criteria

Run `tools/validate_government_e2e.py` against the deployed environment. A green
readiness result is necessary but still capture the actual authenticated UI and
Government feed in the final recordings. Do not substitute simulated camera
metadata for Government acceptance evidence.
