# INTEL-I Phase 8 — AI Scheduler Production Design

## Scope
Phase 8 connects Phase 6 bounded stream ingestion to the shared AI execution plane and Phase 7 TensorRT runtime without moving camera credentials or raw stream URLs into scheduler state.

## Runtime path

```text
Camera decoder
  -> bounded per-camera stream buffer
  -> AI admission
  -> priority queue
  -> shared AI workers
  -> processFrame / validated inference stack
  -> observation/event pipeline
```

## Controls
- Per-camera queue limit prevents one camera from monopolizing inference.
- Global queue limit prevents unbounded memory growth.
- Priority levels: BACKGROUND, NORMAL, IMPORTANT, WATCHLIST.
- Older/lower-priority work is shed first under pressure.
- Stale normal-priority frames are discarded rather than processed late.
- GPU utilization and VRAM are cached and used for admission protection.
- If GPU telemetry is unknown, only WATCHLIST priority work is admitted when protection is enabled.
- Camera decode threads are separated from AI worker threads.
- A failed AI job does not kill the camera decoder.
- Scheduler shutdown drains/cancels queued work without accepting new jobs.

## Security
- Scheduler jobs contain only camera ID, frame object and non-secret processing metadata.
- No decrypted RTSP URLs, passwords, tokens or connector configuration are stored in the queue.
- Queue status endpoint is authenticated and excludes per-camera queue internals.
- GPU telemetry is diagnostic only and never trusted as an authorization decision.

## Production validation
Run the backend tests, then validate with a real NVIDIA GPU. Measure GPU utilization, VRAM, inference latency, queue depth and dropped frames at increasing camera counts. Do not claim a camera capacity until it is measured.
