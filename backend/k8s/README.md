# INTEL-I — single AWS L40S Kubernetes package

Start here, then follow `RUNBOOK.md` in order. This folder contains complete manifests and Linux deployment/validation scripts. It does not contain your backend source, Docker images, model weights, Kubernetes installer, or credentials.

Default target: one API, one camera worker (maximum two cameras), one ANPR worker, one FRS worker, and one instance each of PostGIS, Redis, Kafka and MinIO. There is no HPA. The three GPU workers each request one of three time-sliced slots on the same physical L40S. GPU memory and compute are shared, not partitioned or increased.

**Status:** static package checks passed. No AWS cluster, image build, model inference or live camera test was available in this session. Complete the runbook gates before calling this deployment validated.

## Package decisions

- The application commands, DNS names, health paths and token headers follow your attachment. API retains the existing image entrypoint; worker commands are explicit.
- All deployments have one replica and use `Recreate`. Updates intentionally have downtime on this single-node target and cannot temporarily request extra GPU slots.
- Models are baked into each applicable image under `/app/models`; PaddleOCR vendor files live at `/app/vendor/PaddleOCR`. No FRS model PVC is used. The supplied face model paths are preserved; verify they match the actual CUDA-capable FRS implementation. Kubernetes cannot turn a CPU-only face pipeline into GPU inference.
- Local persistent volumes live at `/var/lib/intel-i/{postgres,redis,kafka,minio}` on the labeled EC2 node. Put this path on persistent EBS-backed storage. No EBS CSI driver or ReadOnlyMany volume is required. This is node-bound storage, not multi-node failover. PV deletion policy is Retain; it does not protect against EC2/EBS deletion. Capacity declarations do not enforce directory quotas.
- API and workers use shared MinIO objects; `/app/object-cache` is disposable per-pod cache. Uploaded MP4s must be in MinIO, not just an API pod's filesystem.
- Redis AOF and Kafka data persist. Redis uses noeviction to avoid silently evicting camera coordination keys. Redis and Kafka are unauthenticated on internal ClusterIP endpoints for this private single-node validation. NetworkPolicy blocks unrelated ingress when the CNI enforces it. Egress is unrestricted for camera connections and DNS. No public service or ingress is created.
- Authenticated exec probes use curl. Readiness parses JSON and requires `status` equal to READY, case-insensitively. Startup/liveness check HTTP success. If the application returns non-2xx health while a dependency is unavailable, liveness will eventually restart it; inspect the real endpoint behavior during testing.
- Migration and bucket initialization run as separate Jobs before the API. Migration failures stop deployment; do not use `alembic stamp head` to bypass them.
- Camera onboarding and Prometheus ServiceMonitor are optional and excluded from the default target. No unsupported monitoring endpoint is claimed as working.

## Files

| Path | Purpose |
|---|---|
| `configmaps/` | Non-secret platform settings and health JSON parser |
| `secrets/intel-i-secrets.example.yaml` | Placeholders only; excluded from Kustomize |
| `postgres/`, `redis/`, `kafka/`, `minio/` | One persistent infrastructure instance each |
| `storage/` | Static local PVs and StorageClass |
| `api/`, `camera-worker/`, `anpr-worker/`, `frs-worker/` | Complete deployment/service manifests |
| `gpu/` | NVIDIA runtime class, time-slicing config and Helm values |
| `jobs/` | DB migration and MinIO upload-bucket initialization |
| `network-policy.yaml` | Default-deny ingress plus service allow-lists |
| `scripts/` | Image configuration, secrets, storage, deployment, validation |
| `optional/` | Camera onboarding worker |
| `monitoring/` | Optional ServiceMonitor and scrape network policy |
| `RUNBOOK.md` | Exact AWS/Linux commands and acceptance gates |
| `AUDIT.md` | Findings in the uploaded ZIP and corrections |
| `VALIDATION.md` | Validation performed and remaining checks |

## Before deploying

You must supply your four built ECR images with version tags/digests and real secrets locally. Check model paths in `configmaps/platform.yaml`. The API image must include Python, Alembic, your migrations and the `minio` Python package; worker images must include Python, sh, curl and their GPU dependencies. Use the existing working auth configuration; generated fresh-install secrets cover common names but the exact auth variable names and algorithm cannot be verified without backend source.

Do not merge this folder into the old one: that would leave the old HPA and model PVC files. Back up the old folder and replace it. Replacing files does not delete already-deployed Kubernetes resources; the runbook explicitly removes old scaling objects without deleting model data.

Sources used for the infrastructure configuration:
- [NVIDIA time-slicing configuration and limitations](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
- [NVIDIA device plugin installation and Helm configuration](https://github.com/NVIDIA/k8s-device-plugin)
- [Apache Kafka 3.9.1 official Docker image](https://kafka.apache.org/39/getting-started/docker/)
