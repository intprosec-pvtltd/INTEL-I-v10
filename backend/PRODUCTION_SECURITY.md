# INTEL-I Production Security Notes

## Critical: rotate exposed credentials

The supplied archive contained real-looking secrets in `.env`, including:
- Telegram bot token
- PostgreSQL credentials
- Redis credentials
- RSA private key
- Prometheus token
- camera-source encryption key

Those values are intentionally NOT included in the production package.

Treat them as compromised and rotate/revoke them before deployment.

## Deployment rules

1. Use a secret manager or environment injection for all credentials.
2. Never commit `.env`, private keys, database passwords, Redis passwords,
   Kafka credentials, or alert tokens.
3. Run Alembic migrations in production; keep `AUTO_CREATE_SCHEMA=false`.
4. Use HTTPS for the frontend/API and secure cookies.
5. Prefer Kafka TLS/SASL in production rather than PLAINTEXT.
6. Restrict PostgreSQL and Redis to private network access.
7. Run FastAPI behind a TLS-terminating reverse proxy/API gateway.
8. Run GPU camera workers separately from the API process when scaling.
9. Keep uploaded video storage outside the application source tree when possible.
10. Encrypt sensitive persisted evidence at rest and apply retention policies.
11. Do not log camera source URLs, credentials, raw plate text, or embeddings.
12. Monitor Redis/Kafka/PostgreSQL health and worker lag.
13. Test correlation thresholds against labeled validation data before operational use.
