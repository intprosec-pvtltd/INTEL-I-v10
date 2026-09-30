# INTEL-I production P0 acceptance

A P0 is green only after implementation, component verification, E2E verification and failure/recovery verification.

## Required external acceptance

1. Controlled models: run `scripts/test_model_manifest_acceptance.py`, `scripts/preflight_models.py`, then authenticated `POST /api/models/verify`. Every enabled required model must show configured/model_present/checksum_verified/loaded/inference_verified/ready=true.
2. Ollama ANPR: set `OLLAMA_VISION_ENABLED=true` and `OLLAMA_VISION_REQUIRED=true` when promised. Deployment must show the exact `OLLAMA_VISION_MODEL` in Ollama `/api/tags`; `/ready` must become 503 if it disappears.
3. Face E2E: enroll a reference face, upload a video, verify alert+snapshot+incident+IncidentEvidence+saved-alert Kafka+WebSocket. Repeat on one live camera.
4. Intelligence Assistant: test cross-tenant denial, unavailable facts, prompt injection, RAG grounding references, safe internal links and Ollama outage.
5. Correlation: `vehicleCorrelation.py` is the sole stateful runtime. Verify PostGIS rejects impossible cross-camera travel and accepts a feasible calibrated route.
6. Alert delivery: persisted alert + transactional outbox -> Kafka saved topic -> savedAlertConsumer -> WebSocket/OpenSearch. Frontend reconnect uses DB recovery.
7. Government data: adapters remain NOT CONFIGURED until authorized endpoint/schema/credentials are supplied. Never claim live VAHAN/SARATHI/CCTNS/AFIS/NAFIS connectivity from the abstraction alone.
8. Benchmark: run `scripts/benchmark_multicamera.py` on the target GPU with real/representative RTSP feeds for 5,10,25,50 cameras. Archive CSV plus hardware/software versions.
9. Failure tests: run `scripts/failure_acceptance.py` for camera, DB, Kafka, Ollama, GPU, frontend and WebSocket scenarios and attach logs. Verify no lost durable alerts and successful recovery.

## ANPR difficult-condition set

Build a labeled Indian-plate dataset covering day/night, IR, rain, haze/fog, headlight glare, motion blur, compression, small plates and oblique views. Report plate-detection recall, full-plate exact-match accuracy, character accuracy, false plate rate and p50/p95 latency. Do not tune against the final holdout set. Enhancement may improve visibility but must never be treated as evidence that an unreadable character is known.
