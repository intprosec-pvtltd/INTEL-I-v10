# Canonical vehicle correlation runtime

`vehicleCorrelation.py` is the single stateful vehicle-correlation runtime used by INTEL-I.
`services/vehicleCorrelationPolicy.py` contains stateless policy/test primitives only.
`services/vehicleCorrelationEngine.py` is a compatibility facade and MUST NOT own correlation state.

PostGIS feasibility is applied in the canonical runtime before cross-camera identity acceptance. PostgreSQL/PostGIS remains authoritative for camera geometry and travel feasibility. No second correlation state store is permitted.
