# Distributed camera migration - implemented changes

## Added

- Persistent `Camera.desired_state` and update timestamp with Alembic migration.
- Redis atomic camera leases with compare-and-renew/release semantics.
- Compatibility `cam_id` lease to protect existing globally keyed runtime state.
- Short-lived camera runtime telemetry and worker heartbeats.
- Independent `python -m workers.camera_worker` process.
- Secure worker-side camera source/connector resolution.
- Redis Pub/Sub bridge for worker-generated realtime alerts/vehicle positions.
- Authenticated API-to-worker MJPEG preview proxy with worker-failover recovery.
- Authenticated distributed worker inventory endpoint.
- Distributed local Docker Compose overlay.
- Kubernetes API/worker baseline, PDB/HPA and conservative 60-camera overlay.
- Updated multi-camera benchmark script and distributed regression tests.

## Preserved intentionally

- Current detection/analytics flow.
- YuNet + SFace FRS behavior and biometric encryption.
- ANPR and OCR integration.
- Vehicle attributes/Re-ID/correlation.
- GIS/camera metadata and alert pipeline.
- Existing upload/recorded-video path.
- Existing live-camera path when distributed mode is disabled.

## Known transition boundary

The worker currently imports the existing analytics implementation from
`main.py` to minimize behavioral change. Live-camera ownership is distributed,
but model imports have not yet been fully isolated from the API module. Treat
this as the safe Phase-1 migration bridge; complete model/service extraction is
Phase 2 after the ownership/failover acceptance gate.
