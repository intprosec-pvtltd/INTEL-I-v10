# INTEL-I Distributed Production Migration Report

Date: 2026-09-17

## Implemented in this package

| Phase | Result | Evidence |
|---|---|---|
| 1–2 API/AI separation | Implemented | `main.py` has no YOLO constructor; models are loaded by `runtime/analytics_runtime.py` only in worker roles. Camera start/stop persist desired state for live cameras. |
| 3 latest-frame camera pipeline | Implemented | `services/latestFrameBuffer.py` and `services/cameraPipeline.py`; independent capture/analytics loops and one-frame replacement semantics. |
| 4 worker ownership/execution | Implemented | `workers/camera_worker.py` uses `AnalyticsRuntime`, `CameraPipeline`, PostgreSQL desired state and Redis leases. |
| 5 truthful readiness | Implemented | `services/workerReadiness.py`; required dependency failures produce `NOT_READY`. |
| 6 isolated Awiros | Preserved | Existing `workers/anpr_ocr_worker.py` and `services/anprOcrClient.py` remain isolated. |
| 7–15 isolated FRS | Implemented foundation | ONNX Runtime GPU embedder, quality gate, track fusion, watchlist index, temporal confirmation, internal API and HTTP client added. Existing CPU FRS remains available as fallback. |
| 16 Kafka contracts | Implemented | `events/schemas.py` and `events/publisher.py`; inline frames are rejected. |
| 17–20 separate images | Implemented | API, camera, ANPR and FRS Dockerfiles use isolated dependency sets. |
| 21 distributed Compose | Implemented | `docker-compose.distributed.yml` wires API, camera worker, ANPR and FRS services. Existing infrastructure Compose provides PostgreSQL/PostGIS, Redis, Kafka and MinIO. |
| 25–33 Kubernetes baseline | Implemented where code exists | Namespace, ConfigMap, Secret template, API, camera worker, ANPR, FRS and monitoring manifests added. |

## Validation completed here

- All Python files pass parsing and byte-code compilation.
- Six distributed-runtime contract tests pass.
- Latest-frame stress smoke test confirms stale-frame replacement and sampled analytics.
- All Kubernetes YAML parses successfully.
- Frontend production build passes: 1,878 modules transformed.
- Existing invalid `requirements.txt` concatenation was corrected.

## Requires the target GPU/deployment environment

The following are operational acceptance tests, not source-code tests, and were not falsely marked as passed because this workspace has no Docker daemon, NVIDIA GPU, Kubernetes cluster, model weights, PostgreSQL/Redis/Kafka/MinIO services or live RTSP credentials:

1. Build and start all four images.
2. Confirm FastAPI GPU memory is effectively zero with `nvidia-smi`.
3. Confirm Awiros reports `actual_device=gpu`.
4. Confirm FRS reports `CUDAExecutionProvider` first and performs a real embedding inference.
5. Run one complete camera through capture, detection, ANPR, FRS, alerts and preview.
6. Stop/restart API, ANPR and FRS independently and verify camera continuity.
7. Delete a camera-worker pod and measure Redis-lease failover.
8. Execute controlled 5, 10, 20, 30, 45 and 60-camera tests while recording GPU, VRAM, CPU, RAM, FPS, drops, latencies, Kafka lag and reconnects.

## Important remaining integration boundary

The isolated FRS service and client are production-structured, but the legacy `processFrame` watchlist path still uses `services/personRecognition.py` as the active fallback. Before declaring the new GPU FRS operational, synchronize watchlist embeddings into the FRS index and switch the tracked-face call site to `RemoteFRSClient` in the target environment, then validate its detector/model-specific preprocessing against the chosen ONNX models. This is deliberately reported rather than hidden.
