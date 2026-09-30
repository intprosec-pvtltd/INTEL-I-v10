# INTEL-I Architecture Freeze v1.0

Status: FROZEN
Phase: 0
Implementation Strategy: Minimum Change

## 1. API / Control Plane

Owner:
main.py

Responsibilities:
- REST API
- authentication
- camera control
- system control
- frontend communication

Must not own:
- global correlation decisions
- durable event delivery
- frontend intelligence

## 2. Camera Service

Owners:
cameraConfig.py
connectors/
security/cameraSource.py
security/connectorConfig.py

Responsibilities:
- camera configuration
- source validation
- connector management
- camera lifecycle

## 3. Stream Manager

Current owner:
main.py

Functions:
- camera_worker()
- start_camera_worker()
- stop_camera_worker()
- stream_latest_frame()

Responsibilities:
- decoding
- frame acquisition
- reconnect
- frame timing
- stream state

## 4. Adaptive Frame Processing

Owner:
services/adaptivePipeline.py

Responsibilities:
- target FPS
- frame sampling
- adaptive processing policy

## 5. AI Processing

Current owner:
main.py

Responsibilities:
- detection
- pose
- activity processing
- AI inference

Future:
TensorRT / DeepStream

## 6. Tracking

Owner:
tracker.py

Responsibilities:
- local tracking
- track lifecycle
- trajectories

## 7. ANPR

Owners:
vehicleANPR.py
services/anprVoting.py

Responsibilities:
- plate detection
- OCR
- multi-frame OCR aggregation

## 8. Vehicle Re-ID

Owner:
vehicleReid.py

Responsibilities:
- vehicle embedding
- similarity
- Re-ID quality

## 9. Vehicle Attributes

Owner:
services/vehicleAttributes.py

Responsibilities:
- vehicle attributes

## 10. Observations

Owners:
db/intelligence_model.py
services/intelligence_persistence.py

Responsibilities:
- observation persistence
- observation metadata
- evidence references

## 11. Correlation

Owner:
vehicleCorrelation.py

Responsibilities:
- cross-camera matching
- Global Vehicle ID
- correlation confidence
- journey construction

## 12. GIS

Owners:
services/gis_engine.py
routers/gis.py

## 13. Watchlist

Owners:
services/watchlist_service.py
routers/watchlist.py
db/watchlist_model.py

## 14. Alerts

Owners:
alert.py
alertDecision.py

## 15. Event Delivery

Owners:
events/outbox.py
events/outboxWorker.py
events/kafkaProducer.py
events/kafkaConsumer.py

## 16. WebSocket

Current owner:
main.py

Responsibilities:
- live event delivery
- client connection management

Database remains the source of truth.

## 17. Evidence

Owners:
snapshot.py
services/evidenceIntegrity.py

## 18. Health

Owners:
services/systemHealth.py
monitoring/metrics.py

## Architecture Rule

Existing implementations are retained.

No major refactoring is performed during Phase 0.

Future phases may extract components into dedicated workers/processes without changing the contracts defined here.