# External watchlist gateway

The gateway is an admin-only, read-only REST/HTTPS JSON connector. It normalizes records and mirrors supported records into INTEL-I's existing person and vehicle watchlist tables. Camera/AI workers do not receive the remote credential.

## Required configuration

Generate a Fernet key using the backend virtual environment's `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`, then set it as `EXTERNAL_WATCHLIST_ENCRYPTION_KEY`. Back up this key securely; changing it makes already-stored connector credentials unreadable. Never put the key in frontend environment variables or source control.

Optional environment settings:

```dotenv
EXTERNAL_WATCHLIST_ENABLED=true
EXTERNAL_WATCHLIST_CONNECT_TIMEOUT_SECONDS=5
EXTERNAL_WATCHLIST_READ_TIMEOUT_SECONDS=15
EXTERNAL_WATCHLIST_MAX_RESPONSE_BYTES=5242880
EXTERNAL_WATCHLIST_MAX_RETRIES=2
EXTERNAL_WATCHLIST_ALLOWED_HOSTS=
EXTERNAL_WATCHLIST_CA_BUNDLE=
```

TLS verification is enabled on newly created sources. `EXTERNAL_WATCHLIST_CA_BUNDLE` may point to an operator-managed CA bundle when required. The connector rejects non-HTTPS URLs, credentials in URLs, redirects, local/private IP destinations, and responses above the configured byte limit. A specifically trusted intranet endpoint can be allowlisted by exact hostname/IP in the server-only `EXTERNAL_WATCHLIST_ALLOWED_HOSTS` setting; never expose that setting to the browser.

## Mapping

`field_mapping` is a JSON object whose keys are INTEL-I fields and whose values are dot-separated JSON paths in the remote record. Supported keys include `external_id`, `entity_type`, `category`, `active`, `name`, `plate`, `priority`, `case_reference`, `image_data`, `image_content_type`, `make`, `model`, `color`, and `description`. For person FRS matching, the source must provide authorized JPEG/PNG/WEBP reference image bytes as base64 in the mapped `image_data` field. No independent FRS/ANPR model is created. Without an image, the external person is catalogued but cannot be matched by FRS.

A response can be a top-level JSON array or an object containing an array at `records_path`. Optional page-number pagination is configured with `{ "enabled": true, "page_param": "page", "start_page": 1, "page_size": 100, "max_pages": 20 }`.

Run the existing Alembic migration to add connector tables. Source credentials are encrypted at rest. `source_id + external_id` is unique. Disabling a source stops scheduled and manual synchronization; deletion removes only the integration's own synchronized entries.
