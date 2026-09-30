# INTEL-I main.py import/runtime wiring audit

This audit is generated against the updated production backend.

## Production wiring corrections

- Camera tamper/frozen health now runs on the original decoded frame before frame enhancement/DarkIR.
- DarkIR remains scene-level and conditional before YOLO.
- EDSR remains ROI-only/task-specific for difficult ANPR crops; original ROI remains available for Re-ID/evidence.
- ANPR canonical service performs deterministic OCR, temporal evidence, character voting, optional bounded Ollama/Gemma verification and source-aware fusion before observation validation/correlation.
- Vehicle attributes now execute before observation validation, so validation/correlation sees the complete observation.
- Observation validation gates cross-camera correlation and watchlist decisions.
- Correlation persists Global Vehicle ID/correlation evidence, updates GIS/journey state, then local vehicle state and bounded observation persistence.
- Person appearance/correlation, behavior/EventFusion, alert decision, evidence hashing, risk/incident correlation, outbox/WebSocket, camera health, time sync, Stream Manager and AI Scheduler remain wired at their existing production stages.
- Identity correction is intentionally operator-controlled through `/api/vehicles/identity/correction`; it is not automatically invoked by inference.

## Imports removed from main.py after reference/ownership analysis

These were not wired in `main.py` because calling them there would duplicate canonical service ownership:

- `OCRCandidate`, `character_vote` — owned by canonical ANPR service.
- `fuse_plate_candidates` — owned by canonical ANPR service.
- `ConfidenceCalibrator` — owned by `AdvancedIntelligenceEngine` instance.
- `build_evidence_manifest` — owned by `AdvancedIntelligenceEngine` evidence path.
- `get_vehicle_embedding` — superseded by richer `get_vehicle_embedding_result` in main pipeline.
- `get_snapshot` — insecure/unscoped duplicate; main uses user-scoped snapshot retrieval.
- unused ORM symbols `Snapshot`, `Watchlist`, `WatchlistEntry` — removed from main imports; their owning DB/services remain intact.

## Static result

Every explicit project-level import remaining in `main.py` has at least one real runtime/reference use. No function is called merely to make an import appear used.
