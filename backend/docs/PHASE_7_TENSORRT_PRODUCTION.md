# INTEL-I Phase 7 — TensorRT Production Implementation

## Scope

Phase 7 establishes a secure, reproducible TensorRT artifact pipeline for the exact INTEL-I models. It does not claim a TensorRT engine is available when the model binary/GPU build environment is absent.

## Model policy

### Vehicle detector

`models/yolov8m.pt` → `vehicle_detector.engine`

### Plate detector

`models/license_plate_yolov8m.pt` → `plate_detector.engine`

### Vehicle Re-ID

`models/osnet_x1_0_vehicle_reid.onnx` is eligible for a later TensorRT conversion only after preprocessing/output compatibility is validated. It is not silently converted here.

### OCR

`en_PP-OCRv5_rec_mobile.onnx` remains ONNX Runtime initially. TensorRT is not forced onto every model.

### Unknown model

`best.pt` must not be assigned to a production TensorRT role until its task/classes are inspected and verified.

## Security controls

1. Engine and source model SHA-256 hashes are stored in `manifest.json`.
2. Engine loading fails closed on missing or changed artifacts.
3. TensorRT imports are lazy; control-plane startup does not require TensorRT unless configured as required.
4. No arbitrary remote model URLs are accepted by the build script.
5. Existing engines are never overwritten unless `--force` is explicitly supplied.
6. Engine files are not trusted merely because they have a `.engine` extension.
7. Production TensorRT mode is separate from the CPU/OpenCV fallback.
8. No secret, RTSP credential, or source URL is written into the engine manifest.
9. Model provenance is retained with source hash, builder version, platform and GPU name.
10. The application never claims measured camera capacity from VRAM alone.

## Runtime states

`DISABLED` → TensorRT is not selected.

`NOT_READY` → requested but manifest/artifacts/runtime are not valid.

`READY` → manifest and engine artifacts are valid and TensorRT is available.

`LOADED` → an engine has been deserialized by the TensorRT runtime.

## Production acceptance

A Phase 7 deployment is accepted only after:

- source model hash verified;
- engine hash verified;
- TensorRT deserialization succeeds;
- exact GPU/driver/DeepStream environment is recorded;
- exact parser/inference integration is validated;
- representative inputs produce valid output tensors;
- latency is measured;
- GPU/VRAM utilization is measured;
- no memory growth occurs during soak testing;
- fallback behavior is tested;
- no unvalidated engine is exposed as production-ready.

The INTEL-I architecture requires actual GPU benchmarking to establish AI FPS/capacity and specifically warns that a YOLO model cannot simply be assumed to work with stock `nvinfer`. The Phase 7 implementation therefore treats engine creation and engine validation as separate gates.
