# Audit of the attached k8s(3).zip

| Original finding | Assessment / correction |
|---|---|
| Namespace `intel-i`, application/service names, port mappings | Usable; preserved |
| Redis, Kafka, MinIO, ANPR and FRS DNS names | Usable; retained and backed by actual Services |
| Camera POD_IP from `status.podIP` | Usable; preserved |
| API `/health/live` and `/health/ready` | Preserved; startup probe added |
| FRS token-authenticated curl probes | Useful baseline; JSON whitespace-dependent grep replaced |
| API replicas 2; camera replicas 2; FRS replicas 2 | Each is now 1 (ANPR was already 1) |
| FRS HPA 2–6 replicas included by Kustomize | Removed from package and default deployment |
| FRS RollingUpdate maxSurge 1 | Recreate avoids a temporary fourth GPU request |
| FRS minAvailable 1 PDB | Removed for this deliberate one-replica test |
| GPU limits across default replicas required five GPU resources | Three worker replicas now consume three time-sliced slots on one GPU |
| No NVIDIA sharing/runtime configuration | Added ConfigMap, runtime class, Helm values and operator/standalone instructions |
| No Postgres, Redis, Kafka or MinIO manifests | Added all four, with Services, probes and persistent volumes |
| ReadOnlyMany FRS models PVC, with no populated storage | Removed; model files must be baked into images |
| Camera had readiness only; ANPR had no probes | Startup, readiness and liveness included for both |
| `grep` required exact compact JSON `"status":"READY"` | Python parser accepts JSON whitespace and rejects missing/unready status |
| `DATABASE_URL` and worker-preview token missing from usable setup | Required secret template/generator and validation included |
| Object storage backend/cache/bucket and Redis enablement incomplete | Added explicit MinIO cache/bucket and Redis settings |
| `latest` application image tags | Image configuration script requires version tags or digests |
| NetworkPolicy protected FRS only | Default-deny ingress plus explicit per-service traffic rules |
| ServiceMonitor selected nonexistent labels and `metrics` port | Optional monitor selects real API labels and `http` service port; verify `/metrics` before enabling |
| README referenced absent `base`, `overlays/60-camera`, and Dockerfile | Replaced with actual paths and staged commands for this package |
| No migration/dependency startup sequence | Added migration and bucket Jobs, then ordered application deployment |

No INTEL-I Python application files were supplied or changed. The health parser and deployment helpers are new operational files only. Upload source resolution and finite-video EOF behavior are outside this change.
