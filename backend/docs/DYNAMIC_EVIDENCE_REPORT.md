# INTEL-I Dynamic Digital Evidence PDF

The incident evidence export is generated from tenant-scoped authoritative database records. The frontend never supplies report body text, exhibit hashes, incident ownership, or evidence image bytes.

## Endpoints

- `GET /api/reports/evidence/classifications`
- `GET /api/reports/incidents/{incident_id}/evidence/preview?classification=RESTRICTED`
- `GET /api/reports/incidents/{incident_id}/evidence/export?classification=RESTRICTED`

The export is watermarked on every page, includes Gujarat Police branding, hashes the source-evidence manifest, hashes the final delivered PDF after watermarking, and records the export in `audit_logs`. If the audit write fails, the PDF is not released.

## Dynamic sections

Sections are generated only when their backing evidence exists: incident/subject details, event timeline, analytical assessment, multi-camera journey, `E-001..E-N` exhibits, model/pipeline context, integrity/provenance, limitations, and authorisation.

## Security controls

- Tenant ownership is enforced with `Incident.user_id` and `IncidentEvidence.user_id` filters.
- Report classifications are server allow-listed.
- Snapshot images have byte, pixel and dimension limits and are decoded/verified before embedding.
- Source snapshot SHA-256 is computed before report-safe re-encoding.
- Untrusted DB text is XML-escaped before ReportLab `Paragraph` rendering.
- Sensitive embedding/image/token-like metadata is excluded from exhibit metadata.
- Final PDF bytes are `Cache-Control: no-store` and carry `X-PDF-SHA256` and evidence-manifest hash response headers.
- The audit entry records report ID, export ID, classification, source manifest SHA-256, final PDF SHA-256 and exhibit snapshot hashes.
- Original media is never modified by the PDF generator.
