# INTEL-I implementation: Ollama ANPR, person correlation and GIS

Implemented on the 2026-08-31 supplied source snapshot.

## Delivered

- Bounded asynchronous Ollama vision verification for low-confidence ANPR.
- Strict JSON parsing, host allowlist, timeouts, queue limits, cooldown and no redirects.
- PP-OCR + temporal voting + Ollama evidence fusion.
- Policy enforcement: Ollama-only evidence can never exceed POSSIBLE.
- Tenant-scoped GlobalPersonID, person observations and correlation decisions.
- Appearance histogram, clothing colour, time and camera-topology correlation.
- Persistent observed person journey and GIS route API.
- Calibrated person live position; explicit CAMERA_ANCHOR fallback.
- Authenticated `person_position` WebSocket events and GIS frontend markers.
- Person Journeys frontend with persisted-evidence summaries.
- Ollama summaries run asynchronously only after evidence persistence.
- Database migration `e5a7c9d2f410`.
- Three-camera vehicle/person preflight tools and deployed GIS/WebSocket validator.

## Required target-host acceptance

Source validation cannot substitute for the actual GPU/camera environment. Before release run:

1. Alembic upgrade on staging PostgreSQL and rollback rehearsal from backup.
2. Ollama vision/text model load on the selected L4/L40S and queue-overload test.
3. Real three-camera single-vehicle and single-person journeys with configured topology.
4. Camera homography calibration for any claim of exact live map position.
5. Authenticated Netlify-to-backend HTTP/WebSocket validation.
6. Ollama outage/timeout test proving camera FPS is unaffected.
7. Indian CCTV ANPR test set covering day, night, blur, glare and two-line plates.

## Migration

```bash
alembic upgrade e5a7c9d2f410
```

Do not enable automatic schema creation in production.
