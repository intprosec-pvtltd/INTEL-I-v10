# DeepStream deployment acceptance checklist

- [ ] NVIDIA driver/CUDA/TensorRT/DeepStream versions are a tested combination.
- [ ] Vehicle TensorRT engine is versioned and checksum recorded.
- [ ] YOLO parser/custom output parser is validated on the exact engine.
- [ ] `nvstreammux` batch size matches the intended camera batch.
- [ ] `nvtracker` configuration is validated on day/night traffic.
- [ ] RTSP reconnect behavior has been tested.
- [ ] Camera timestamps are normalized to server time/NTP.
- [ ] Metadata only is sent through Kafka; evidence video/images use object storage.
- [ ] 1/5/10/20/50 camera benchmarks are recorded using real RTSP sources.
- [ ] GPU utilization, VRAM, inference latency, queue depth and dropped frames are within the acceptance envelope.
- [ ] One camera failure does not stop the remaining streams.
- [ ] WebSocket disconnect does not lose durable observations/alerts.
