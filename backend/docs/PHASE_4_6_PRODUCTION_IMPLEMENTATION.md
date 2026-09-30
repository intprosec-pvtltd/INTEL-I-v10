# INTEL-I Phase 4–6 Production Implementation

## Phase 4 — Camera Health
- Canonical lifecycle states remain the only persisted/exposed states.
- Per-camera health registry tracks frame arrival, FPS, latency, reconnects, read/decoder failures and bounded queue pressure.
- A watchdog marks stale active streams DEGRADED without stopping other cameras.
- Health telemetry is persisted at a controlled interval; per-frame database writes are avoided.
- `/api/cameras/health` and `/api/cameras/{camID}/health` expose authoritative operational telemetry.

## Phase 5 — Time
- Every ingested frame has server receive and ingestion timestamps.
- Source/camera timestamps are accepted when a connector supplies them.
- When source timestamps are unavailable, the system explicitly reports `ESTIMATED` using arrival time; it never falsely claims camera-clock synchronization.
- Per-camera offset, jitter, sample count and sync status are persisted.
- `/api/cameras/{camID}/time-sync` exposes non-secret synchronization status.
- OS clock health is surfaced through the system health endpoint.

## Phase 6 — Stream Manager
- Every active camera has an independent bounded frame buffer.
- Old frames are discarded when the buffer is full so a slow AI path cannot create unbounded memory growth.
- Stream metrics include FPS, latency, decoded/processed/dropped frames, queue depth/capacity and reconnects.
- Existing RTSP reconnect/backoff remains isolated per camera.
- Frame packets carry frame number, receive time and optional source timestamp.

## Security properties
- No credentials, stream URLs or connector secrets are placed in health telemetry.
- Health persistence is rate-limited.
- Camera APIs remain authenticated.
- Camera ownership is enforced before returning telemetry.
- No endpoint accepts a filesystem path or arbitrary command for these features.
- The time-sync implementation never reports `SYNCED` without a real source timestamp sample.
