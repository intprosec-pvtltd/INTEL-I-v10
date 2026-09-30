# Sentinel and VMS/NVR catalogue integration

## What is implemented

INTEL-I can retrieve a Government Sentinel camera catalogue from the configured
`/api/ingest` endpoint, discover cameras without manual stream entry and keep
camera metadata synchronized. A mapping-driven REST adapter supports
heterogeneous VMS/NVR catalogues. A scheduled worker runs idempotent syncs; each
run is transactionally recorded and can also be triggered manually.

Stream URLs, catalogue credentials and alternate profiles are encrypted with
`CAMERA_SOURCE_ENCRYPTION_KEY`. API responses expose only URL hashes and safe
metadata. Integration actions and report exports are audit logged.

## Sentinel setup

Open **Camera Setup → Sentinel / VMS**, enter the HTTPS origin, catalogue path
(`/api/ingest` by default), authentication method and interval, then save. Use
**Discover** for a non-mutating preview and **Sync now** to onboard/update the
catalogue. Automatic synchronization runs thereafter when enabled.

Equivalent API flow:

```http
POST /api/integrations
Content-Type: application/json

{
  "name": "Government Sentinel",
  "provider_type": "sentinel",
  "enabled": true,
  "auto_sync": true,
  "sync_interval_seconds": 300,
  "config": {
    "base_url": "https://sentinel.example.gov",
    "ingest_path": "/api/ingest",
    "auth_type": "bearer",
    "token": "REDACTED",
    "verify_tls": true
  }
}
```

Supported authentication types are `none`, `bearer`, `api_key` and `basic`.
GET and POST catalogue methods, cursor/page pagination and bounded query/body
objects are supported. The normalizer accepts common camera fields, including
ID/name, live status, codec, FPS, width/height or nested resolution, location,
and RTSP/HLS/HTTPS stream URLs.

## Generic VMS/NVR mapping

Set `provider_type` to `generic_vms` or `generic_nvr`. The `mapping` object uses
dotted JSON paths. Example:

```json
{
  "base_url": "https://vms.example.gov",
  "catalogue_path": "/v2/devices",
  "auth_type": "api_key",
  "api_key": "REDACTED",
  "api_key_header": "X-API-Key",
  "mapping": {
    "items": "data.devices",
    "id": "uuid",
    "name": "label",
    "status": "availability",
    "latitude": "geo.y",
    "longitude": "geo.x",
    "stream_url": "play"
  }
}
```

This adapter covers REST catalogues that ultimately provide RTSP, HLS or HTTPS
media addresses. Vendor SDKs, proprietary binary protocols, multicast routing,
and vendor-specific playback/PTZ controls require a vendor plug-in implementing
`CameraCatalogueConnector`; they are not silently claimed as universal support.

## Synchronization behavior

- `(user, provider, external camera ID)` is the stable identity.
- Existing camera rows are updated in place; unchanged rows are not duplicated.
- Missing cameras are marked `MISSING` and may be deactivated by policy; they
  are not deleted automatically.
- RTSP is preferred over HLS and HTTPS when the catalogue supplies alternatives.
- IDs, live state, codec, resolution, FPS, location and safe metadata are kept.
- API responses never return a source stream URL or credential.
- Sync requests for the same integration are serialized.

## PTS and discontinuity policy

Production defaults to PyAV decoding with `STRICT_PTS_REQUIRED=true`. Analytics
time, tracker duration inputs, evidence timestamps, ANPR observations and
cross-camera correlation use decoder presentation timestamps. A frame with no
PTS is rejected rather than assigned a fixed-FPS clock. Set strict mode to false
only for a documented legacy source that cannot expose timestamps.

Backward PTS jumps, excessive PTS gaps, reconnects and visual hard cuts create a
new segment and reset camera-local person/vehicle trackers, ANPR temporal state,
behavior state, appearance state and scene baselines. Discontinuity counts and
reasons are available from camera health/time-sync APIs.

## Report and demonstration endpoints

- `GET /api/reports/analytics/preview`
- `GET /api/reports/analytics/export?format=pdf|csv`
- `GET /api/reports/government-e2e/readiness?hours=24`

The readiness endpoint is evidence-based: it becomes complete only after a real
Sentinel sync, online stream, source-PTS observations, vehicle detection, ANPR,
watchlist evaluation, alert and snapshot evidence have all occurred.
