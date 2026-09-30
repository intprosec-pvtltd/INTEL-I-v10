# INTEL-I Advanced Intelligence Additions

## 1. Behavioral intelligence

The backend now emits behavioral alerts from tracked object counts:

- `TRAFFIC_DENSITY` / `HEAVY_TRAFFIC`
- `TRAFFIC_DENSITY_ESCALATION`
- `CROWD_DENSITY` / `CROWD_BUILDUP`
- `CROWD_DENSITY_ESCALATION`

Default levels:

- INFO: traffic >= 35 or people >= 25
- LOW: traffic >= 55 or people >= 50
- MEDIUM: traffic >= 80 or people >= 90
- HIGH: traffic >= 120 or people >= 150

Tune thresholds in `.env`. These are count thresholds, not a substitute for calibrated road/camera density metrics; production deployment should calibrate per camera/ROI.

## 2. Vehicle attributes

`services/vehicleAttributes.py` supports:

- vehicle type from the existing detector
- conservative color fallback
- optional make/model/color attribute model via `VEHICLE_ATTRIBUTE_MODEL_PATH`

The attribute model is intentionally not downloaded automatically. Put the approved model on the secured host and configure its path. No external cloud inference is used.

## 3. Person watchlist

The person watchlist supports:

- `MISSING`
- `WANTED`
- `OTHER`

Reference images are processed into face embeddings. The database stores an encrypted embedding using a Fernet key; the reference image is not exposed through a public endpoint.

Configure:

```env
FACE_DETECTOR_MODEL_PATH=C:\\secure-models\\face_detection_yunet.onnx
FACE_RECOGNIZER_MODEL_PATH=C:\\secure-models\\face_recognition_sface.onnx
FACE_RECOGNITION_MODEL_VERSION=1.0
PERSON_FACE_MATCH_THRESHOLD=0.45
PERSON_EMBEDDING_KEY=<generated-fernet-key>
```

Generate a Fernet key locally, store it in the secret manager/environment, and do not commit it to Git.

The implementation does not auto-download biometric model weights.

## 4. Incident and evidence

Vehicle exact watchlist matches and person watchlist matches now create an `Incident` plus `IncidentEvidence` record. Evidence is tenant-scoped and can reference the authenticated alert/snapshot records.

Frontend routes:

- `/person-watchlist`
- `/incidents`

## 5. Migration

Run against the actual PostgreSQL database:

```powershell
alembic current
alembic heads
alembic upgrade head
```

The new migration is:

`c4d9f0a1b2c3_add_advanced_intelligence.py`

## 6. Security

- Do not commit `.env`.
- Do not expose camera source credentials to the frontend.
- Do not expose face embeddings through API responses.
- Keep person watchlist APIs authenticated and user-scoped.
- Keep snapshots behind the existing authenticated snapshot endpoint.
- Use TLS/WSS in production.
- Rotate any secret that has previously been exposed.
