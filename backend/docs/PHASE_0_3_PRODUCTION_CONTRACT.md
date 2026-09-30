# INTEL-I Phase 0-3 Production Contract

## Phase 0 — Architecture
The control plane owns API/auth/configuration and the media plane owns camera ingestion. Downstream intelligence consumes a normalized camera contract and must not depend on source-specific details.

## Phase 1 — Infrastructure
Required runtime dependencies: PostgreSQL, Redis and Kafka/event bus. Startup readiness is separate from process liveness. The application reports READY only when critical infrastructure is healthy.

## Phase 2 — Database
Camera configuration is persistent. Secrets remain encrypted. Runtime state is represented by canonical camera states and health timestamps. Alembic owns production schema changes.

## Phase 3 — Camera Integration
Every camera exposes the same normalized contract: camera identity, source type, resolved source type, location, direction, stream properties, analytics capabilities and canonical lifecycle state. Plaintext sources and connector credentials are never returned to the frontend or stored in Redis runtime state.

### Canonical camera states
- ONLINE
- DEGRADED
- RECONNECTING
- OFFLINE
- AUTHENTICATION_FAILED
- TIMEOUT
- AI_DISABLED

Legacy CONNECTED/DISCONNECTED values are accepted only as migration/input aliases.
