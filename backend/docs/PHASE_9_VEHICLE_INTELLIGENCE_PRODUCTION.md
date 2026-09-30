# INTEL-I Phase 9 — Vehicle Detection / Tracking / Identity Production Contract

## Pipeline

```text
Stream Manager
    ↓
AI Scheduler
    ↓
Vehicle Detector (TensorRT when enabled and validated; controlled fallback otherwise)
    ↓
Vehicle Detection
    ↓
ByteTrack / configured tracker
    ↓
Local Vehicle Track
    ↓
Observation Validation
    ↓
┌───────────────┬────────────────┬──────────────────┐
│ Plate/ANPR    │ Vehicle Re-ID  │ Attributes       │
└───────────────┴────────────────┴──────────────────┘
    ↓
Cross-camera Correlation
    ↓
Global Vehicle ID
    ↓
Journey / GIS / Watchlist / Alerts
```

## Production gates

1. Vehicle detector artifact must pass the Phase 7 TensorRT manifest/hash gate when TensorRT is enabled.
2. `TENSORRT_REQUIRED=true` must never silently fall back to PyTorch.
3. Every detection is converted into a bounded local-track observation.
4. Low-confidence, malformed, stale, frozen, or tampered observations cannot strengthen cross-camera identity.
5. Re-ID is quality-gated and throttled; it is evidence, not identity truth.
6. ANPR is multi-frame evidence; OCR confidence is never treated as vehicle identity confidence.
7. Correlation is executed only on accepted observations.
8. Watchlist matching is downstream of stable evidence and never promotes fuzzy matches to automatic positive alerts.
9. No license plate or secret camera credential is written to ordinary application logs.
10. Every pipeline stage is failure-isolated so one model failure does not kill a camera worker or the API process.

## Operational acceptance

Measure at minimum:

- detector latency
- tracking latency
- Re-ID latency
- ANPR latency
- accepted/rejected observation rate
- active tracks
- GPU utilization
- GPU memory
- scheduler queue depth
- dropped frames
- end-to-end alert latency

Run the benchmark at 1, 5, 10, 20, 30 and 50 cameras on the target NVIDIA hardware. Do not claim a camera capacity without measured evidence.
