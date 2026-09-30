# INTEL-I OpenSearch + RAG production integration

## Scope

The OpenSearch layer indexes searchable metadata for authorized person watchlists, vehicle watchlists, incident evidence and alerts. It never indexes image bytes, biometric embeddings, source credentials, JWT keys, encryption keys or `.env` values. Every operational document carries `user_id`, and every operational search applies an exact user filter.

The RAG corpus is intentionally restricted to approved INTEL-I Markdown documentation plus authorized operational records. Source code and environment files are excluded by design so the assistant cannot retrieve credentials or internal secrets.

## Indices

- `inteli-person-watchlist-v1`
- `inteli-vehicle-watchlist-v1`
- `inteli-evidence-v1`
- `inteli-alerts-v1`
- `inteli-knowledge-v1`

When `OPENSEARCH_VECTOR_ENABLED=true`, the same indices also contain a `knn_vector` field. INTEL-I uses the Ollama `/api/embed` endpoint and requires the configured vector dimension to exactly match the chosen embedding model.

## Production sequence

1. Deploy OpenSearch with TLS, authentication, a private network and persistent volumes.
2. Create a least-privilege INTEL-I application account. Do not use the admin account from the application.
3. Configure the CA path and enable certificate verification.
4. Install backend requirements.
5. Start INTEL-I and confirm `/api/search/health` reports READY.
6. POST `/api/search/reindex` once for each tenant/operator account that owns data, or run the same operation from an administrative orchestration workflow.
7. Query `/api/search?q=...` and confirm only the authenticated user's records are returned.
8. Test Intelligence Assistant. References returned by the assistant include only approved relative frontend links.

## RAG behavior

Live operational questions continue to use backend-controlled SQLAlchemy retrieval as the authority. OpenSearch RAG is used for retrieval-grounded INTEL-I knowledge and broader search. The LLM receives bounded retrieved context and is explicitly instructed to treat retrieved text as data, not instructions. SQL generation is not permitted.

## Deep links

- alert -> `/alert-history?alert_id=<id>`
- incident/evidence -> `/incidents?incident_id=<id>`
- person watchlist -> `/person-watchlist`
- vehicle watchlist -> `/watchlist`

The frontend validates that assistant-provided references are relative paths before rendering them as links. Alert History and Incident Center read the query parameter and open the matching record after data loads.
