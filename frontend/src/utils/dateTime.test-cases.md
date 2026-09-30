# Indian timestamp verification cases

These examples document the expected browser-independent rendering contract:

- `2026-09-10T10:00:00Z` → `10 Sep 2026, 15:30:00`
- `2026-09-10 10:00:00` (legacy naive backend UTC) → `10 Sep 2026, 15:30:00`
- epoch `1789034400` → interpreted as seconds, then displayed in Asia/Kolkata

The backend stores UTC and emits an explicit `Z`; only presentation converts to
Asia/Kolkata. Existing historical rows written as naive local time may require a
one-time database-specific migration after their provenance is confirmed.
