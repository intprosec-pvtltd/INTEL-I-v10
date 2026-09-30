# INTEL-I Central Inference Implementation

This package introduces a central GPU inference service while preserving local fallback mode. Camera workers retain ingest, timestamping, reconnect, preview, lease ownership and frame sampling responsibilities; the inference worker owns heavy models. Dynamic bounded priority batching prevents unbounded latency. Existing ByteTrack, ANPR, FRS, Re-ID, behavior, correlation, evidence and alert logic remain inside the validated analytics path.

## Run locally
`python -m workers.inference_worker` then `python -m workers.camera_worker`. Set `CENTRAL_INFERENCE_ENABLED=false` to use legacy local inference.

## Capacity validation
Run 10/20/30/40/50-camera stages and collect inference worker metrics with `python scripts/benchmark_inference_capacity.py --stage 10`. Do not claim 50-camera capacity until all stages pass operational SLOs.
