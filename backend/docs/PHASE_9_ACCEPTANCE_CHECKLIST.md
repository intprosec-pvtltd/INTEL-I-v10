# Phase 9 Acceptance Checklist

## Detection
- [ ] Validated vehicle TensorRT engine is used when `TENSORRT_ENABLED=true`.
- [ ] `TENSORRT_REQUIRED=true` never falls back to PyTorch.
- [ ] Detector class mapping is verified against engine/model metadata.
- [ ] Detection confidence and geometry are validated.

## Tracking
- [ ] ByteTrack/NvTracker produces stable local track IDs.
- [ ] Tracker state is isolated per camera.
- [ ] Track quality and stability are exposed as evidence.
- [ ] Camera reconnect resets scene/tracker continuity.

## Identity evidence
- [ ] Re-ID runs only on valid vehicle crops and quality-gates bad crops.
- [ ] ANPR remains multi-frame evidence and is never identity truth.
- [ ] Vehicle attributes are evidence with model/version metadata.
- [ ] Observation validation runs before correlation.
- [ ] Rejected observations cannot trigger correlation/watchlist escalation.

## Security
- [ ] No camera credentials appear in logs, metrics, API responses, or job objects.
- [ ] No plaintext plate values appear in ordinary logs.
- [ ] Model/engine provenance and SHA-256 validation pass.
- [ ] Model loading is local/allowlisted and fail-closed when required.
- [ ] Per-stage failures are isolated.

## Performance
- [ ] Detector latency measured on target GPU.
- [ ] Tracking latency measured.
- [ ] Re-ID latency measured.
- [ ] ANPR latency measured.
- [ ] End-to-end latency measured.
- [ ] GPU/VRAM/queue/drop metrics captured.
- [ ] 1/5/10/20/30/50 camera tests completed before claiming capacity.
