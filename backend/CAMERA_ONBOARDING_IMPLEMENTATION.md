# INTEL-I Automated Camera Onboarding

## Changed architecture

The existing manual `/cameras` registration, streaming, GIS, alerts, WebSocket, integration catalogue, and distributed camera-worker flows remain intact. The new `/api/camera-onboarding/*` control plane validates and registers cameras with `desired_state=STOPPED` and `processing_enabled=false`; registration therefore never starts GPU inference automatically.

## Database migration

Run:

```bash
alembic upgrade head
```

Migration `f9a1b2c3d4e5` adds encrypted credential profiles, hierarchical groups, AI profiles, durable onboarding jobs/items, fleet health fields, stream metadata, and worker-assignment fields. It preserves all existing camera rows.

## Environment

Copy the values in `CAMERA_ONBOARDING_ENV.example` into the deployment environment. Continue using the existing `CAMERA_SOURCE_ENCRYPTION_KEY`; plaintext passwords and credential-bearing URLs are never returned by an API or written to logs.

For local development, leave `CAMERA_ONBOARDING_INLINE_FALLBACK=true`. For production, set it to `false` and run one or more workers:

```bash
python -m workers.camera_onboarding_worker
```

Worker concurrency is bounded by `CAMERA_ONBOARDING_MAX_CONCURRENCY`. Every decoder is closed in a `finally` block.

## APIs

- `POST /api/camera-onboarding/discover`
- `POST /api/camera-onboarding/import/csv`
- `POST /api/camera-onboarding/import/rtsp`
- `GET /api/camera-onboarding/jobs`
- `GET /api/camera-onboarding/jobs/{job_id}`
- `GET /api/camera-onboarding/jobs/{job_id}/items`
- `GET /api/camera-fleet` (server-side search, filters, pagination)
- `POST /api/cameras/bulk-actions`
- `GET|POST /api/credential-profiles`
- `GET|POST /api/camera-groups`
- `GET|POST /api/camera-ai-profiles`

Existing Sentinel/VMS endpoints under `/api/integrations` remain the catalogue source. The adapter package exposes one stable discovery/metadata/stream/validation/sync contract for Sentinel, generic VMS/NVR, and ONVIF.

## Security and RBAC

Only existing INTEL-I administrator roles can discover/import cameras, manage credentials, create groups/profiles, or run bulk actions. Operators can view the paginated fleet. Discovery rejects public, loopback, link-local, and multicast networks and caps the authorized private range size. Dangerous bulk deletion requires `confirmed=true`.

## WebSocket progress

The existing authenticated `/ws` channel emits `camera_onboarding_progress` with total, completed, successful, failed, duplicate, and processing counts. The UI also reloads durable job history, so reconnecting does not lose status.

## Validation stages

Each RTSP camera is checked through network/RTSP connection, video-track discovery, bounded frame decode, FPS/resolution/codec extraction, and PTS availability. `ONLINE` is assigned only after a frame is decoded. PTS capability is recorded as `VALID` or `UNAVAILABLE`; missing PTS does not block registration. A failure affects only its own job item.

## Local tests

Install the updated requirements, migrate, then run:

```bash
pip install -r requirements.txt
alembic upgrade head
pytest -q tests/test_camera_onboarding.py
```

Frontend:

```bash
npm install
npm run build
```

### One camera

Open Camera Setup → Camera Onboarding → RTSP URLs, paste one URL, start validation, and confirm the job reaches `1 / 1`. The camera must appear as registered in Camera Fleet without starting AI.

### Fifty cameras

Paste 50 URLs (one per line), or upload a 50-row CSV. Confirm progress updates in batches, individual errors do not stop healthy cameras, completed + processing equals 50, and no more than the configured concurrency is opened at once.

## Backward compatibility checklist

- Existing `/cameras` manual registration remains available under Single Camera.
- Existing upload-video tab remains unchanged.
- Existing camera source encryption is reused.
- Existing Sentinel/VMS catalogue and metadata scheduler remain available.
- Existing `/ws` authentication is reused.
- Existing distributed camera assignment stays authoritative for active processing.
- Existing camera rows receive nullable/defaulted columns only.
