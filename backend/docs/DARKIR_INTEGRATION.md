# INTEL-I DarkIR integration

## Pipeline

The production video path is:

`CCTV/RTSP -> stream processing -> quality assessment -> (low light: DarkIR) -> YOLO -> tracker -> ROI -> EDSR_x4 -> OCR/Re-ID/Pose -> existing INTEL-I intelligence pipeline`

DarkIR is **not** applied to the full frame in normal conditions. It is gated by the existing camera-quality analyzer and runs only for low-light/very-dark frames.

EDSR remains a post-detection ROI enhancement stage. Do not move EDSR back to the full-frame pre-YOLO stage.

## Model

- Model: DarkIR-m
- Checkpoint: `models/DarkIR_384.pt`
- SHA-256: `61eee7d5cfb593d408fba4c716874449b324ed1f49cbff97c2a25ce2fcbe4fde`
- Expected size: `13,397,397` bytes

The architecture is vendored in `services/darkIR.py` and matches the official DarkIR-m configuration.

## Install

From `backend/`:

```powershell
python scripts\download_darkir.py
```

Then configure:

```env
DARKIR_ENABLED=true
DARKIR_MODEL_PATH=./models/DarkIR_384.pt
DARKIR_DEVICE=auto
DARKIR_MAX_WIDTH=1280
DARKIR_MAX_HEIGHT=720
DARKIR_STRICT=false
DARKIR_SHA256=61eee7d5cfb593d408fba4c716874449b324ed1f49cbff97c2a25ce2fcbe4fde
DARKIR_MIN_QUALITY=0.72
```

## Security

- The checkpoint is verified by SHA-256 before installation.
- A temporary file is used and atomically renamed only after verification.
- The application also verifies the configured SHA-256 before loading the checkpoint.
- PyTorch is required for DarkIR.
- The original CCTV frame is not overwritten in evidence storage.
- Do not commit `.env`, private keys, JWT secrets, camera credentials, or model secrets.

## Runtime behavior

If `DARKIR_STRICT=false` and the checkpoint is missing/unloadable, INTEL-I keeps the existing pipeline running and records the failure in logs/status. For evaluation, set `DARKIR_STRICT=true` so a low-light frame cannot silently proceed without DarkIR.

## License

DarkIR source code is used under its upstream MIT license. See `third_party/DarkIR-LICENSE.txt`.
