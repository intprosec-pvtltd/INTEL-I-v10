# Production SAHI ANPR integration

This integration applies sliced inference only to the existing configured plate detector. It does not replace or modify the general `yolov8m.pt` detector.

## Behavior

- Uses the same `ANPR_PLATE_MODEL` instance through SAHI, avoiding a duplicate GPU model allocation.
- Runs sliced inference on sufficiently large enhanced vehicle regions every configured interval.
- Uses Greedy NMM with IOS matching to merge detections crossing tile boundaries.
- Falls back to the existing standard plate detector if SAHI is unavailable unless `ANPR_SAHI_REQUIRED=true`.
- Maintains a bounded, expiring crop buffer per camera and ByteTrack vehicle identity.
- Selects the highest-quality crops across frames for the existing EDSR/PP-OCRv5/temporal-fusion path.
- Clears crop evidence when a track ends and during scheduled ANPR cleanup.

## Deployment

1. Install the updated `requirements.txt`.
2. Copy the SAHI variables from `.env.example` to the deployment environment.
3. Keep `ANPR_SAHI_REQUIRED=false` for the first controlled deployment.
4. Restart the backend and check `/api/intelligence/vehicle-production-status` for `sahi_enabled`, `sahi_loaded`, crop-buffer count, and slice settings.
5. Compare the same CCTV clip with `ANPR_SAHI_ENABLED=false` and `true`.

Start with an interval of 3. Increase it to 4 or 5 if GPU latency or the inference queue rises. Do not lower the final watchlist confirmation thresholds merely to increase detections.
