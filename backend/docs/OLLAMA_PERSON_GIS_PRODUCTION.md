# Ollama, person correlation and live GIS production contract

## Safety boundaries

- Ollama is advisory, asynchronous, bounded and disabled by default.
- Ollama-only plate text is capped at POSSIBLE and cannot create a confirmed identity.
- Only cropped vehicle imagery is submitted. Do not expose Ollama publicly.
- Investigation summaries are generated only from persisted, user-owned evidence.
- Summaries must not claim identity, guilt, intent or a complete physical route.

## Deployment

1. Run Ollama on a private host/container reachable only by the backend.
2. Pull one vision model and one text model; do not load multiple large vision models on an L4.
3. Apply Alembic migration `e5a7c9d2f410`.
4. Enable the required `OLLAMA_*` variables after `/api/chat` health is verified.
5. Validate person correlation with `python tools/validate_three_camera_correlation.py` and vehicle correlation with `python tools/validate_three_camera_vehicle_correlation.py`.
6. Validate deployed authenticated GIS/WebSocket using environment-only test credentials and `python tools/validate_live_gis_ws.py`.

## Acceptance

- Camera workers do not block when Ollama is slow/offline.
- Queue exhaustion drops advisory requests without dropping video frames.
- PP-OCR and temporal evidence remain authoritative.
- GlobalPersonID data is tenant-scoped and every observation is persisted.
- Person GIS reports CAMERA_ANCHOR unless camera homography calibration exists.
- Summary generation returns 409 when correlation evidence is absent.
