# INTEL-I DeepStream production video path

This directory defines the production GPU video path for deployments where the
number of RTSP streams exceeds what the Python/OpenCV worker can process with
the required latency.

## Target architecture

```text
RTSP / ONVIF
    -> GStreamer / NVDEC
    -> nvstreammux (batched streams)
    -> nvinfer (TensorRT vehicle detector)
    -> nvtracker (DeepStream tracker)
    -> metadata probe
    -> Kafka: intel-i.vehicle.observation.v1
    -> FastAPI correlation / journey / watchlist services
    -> WebSocket
    -> React GIS
```

The FastAPI application remains the control plane and durable intelligence
layer. DeepStream is the high-throughput media/inference plane.

## Why it is separate

DeepStream is an NVIDIA system dependency, not a normal Python pip package.
Install it on the GPU server using the NVIDIA-supported DeepStream release for
that OS and CUDA/TensorRT stack. Do not add fake `pyds` versions to
`requirements.txt`.

## Required production configuration

Set these environment variables in the DeepStream worker environment:

- `DEEPSTREAM_ENABLED=true`
- `DEEPSTREAM_NVINFER_CONFIG=/opt/intel-i/configs/vehicle_detector_nvinfer.txt`
- `DEEPSTREAM_TRACKER_LL_LIB=/opt/nvidia/deepstream/lib/libnvds_nvmultiobjecttracker.so`
- `DEEPSTREAM_TRACKER_CONFIG=/opt/intel-i/configs/tracker.yml`
- `INTELI_KAFKA_BOOTSTRAP_SERVERS=...`
- `INTELI_KAFKA_VEHICLE_TOPIC=intel-i.vehicle.observation.v1`

The `nvinfer` configuration must point at a validated TensorRT engine for the
actual INTEL-I vehicle model. A YOLO model cannot be assumed to be parseable by
stock `nvinfer`; use the parser/custom inference integration appropriate to the
exact exported model.

## Migration rule

Keep the current Python/OpenCV pipeline as a controlled fallback while the
DeepStream pipeline is being benchmarked. Production cutover is allowed only
after the real camera set meets the latency, dropped-frame, GPU/VRAM and ANPR
acceptance thresholds.


## Phase 7 TensorRT gate

DeepStream production mode is permitted only after `models/tensorrt/manifest.json` validates and the exact engine/parser combination has passed target-machine validation. Use `python tools/validate_tensorrt.py --manifest models/tensorrt/manifest.json --runtime` before enabling production mode.
