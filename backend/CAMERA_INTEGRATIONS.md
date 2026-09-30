# INTEL-I camera integrations

The existing camera worker remains the single AI-ingestion path. ONVIF, Vendor
API and Vendor SDK connectors resolve into the worker's capture-like interface;
YOLO, ANPR, Re-ID, correlation, GIS and alerting are unchanged.

## Supported source types

- `rtsp`
- `http`
- `https`
- `hls`
- `webcam`
- `onvif`
- `vendor_api`
- `vendor_sdk`
- existing `live` / `upload`

## Database migration

Run:

```bash
alembic upgrade head
```

The migration adds connector metadata and an encrypted connector configuration
field without changing existing camera rows.

## ONVIF

Install the backend requirements, then configure:

- `host`
- `port`
- `username`
- `password`
- optional `profile_token`

The UI can discover ONVIF devices and media profiles. The connector resolves
the selected ONVIF media profile to RTSP and the existing FFmpeg/OpenCV worker
opens that stream.

Endpoints:

- `GET /camera/onvif/discover`
- `POST /cameras/onvif/profiles`
- `POST /cameras/connection-test`

## Vendor REST / HTTP API

Configure either:

- a direct `stream_url`, or
- `base_url`, `stream_path`, HTTP method and `stream_json_path`.

Supported API authentication modes are `none`, `basic`, `bearer` and
`api_key`. Connector configuration is encrypted at rest.

The API response must contain a usable RTSP/HTTP/HTTPS stream URL at the
configured JSON path when `stream_url` is not supplied.

## Vendor SDK

A native vendor SDK adapter must be installed separately according to that
vendor's distribution/licensing requirements.

Adapters are loaded only from modules allowlisted by:

`INTEL_I_SDK_MODULE_ALLOWLIST`

Each adapter subclasses:

`connectors.vendor_sdk.adapter.VendorSDKAdapter`

and returns a `ConnectorResult`. A native frame source must expose:

- `read() -> (bool, frame)`
- `release()`

This keeps the AI pipeline independent of the vendor SDK.

## Security

- Connector credentials are encrypted using `CAMERA_SOURCE_ENCRYPTION_KEY`.
- Connector secrets are never returned by `/cameras`.
- Connector secrets are never stored in Redis camera state.
- Resolved stream URLs are kept in worker memory only.
- SDK module loading is allowlist-controlled.
- Vendor API redirects are disabled.
- Connector errors returned to the frontend are intentionally generic.

## Existing functionality

Direct RTSP/HTTP/HTTPS/HLS/webcam/upload behavior continues through the
existing camera worker. The connector layer is an additional integration
boundary, not a second analytics pipeline.
