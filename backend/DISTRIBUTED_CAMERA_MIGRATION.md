# INTEL-I distributed camera migration runbook

This release is the safe migration bridge from the current in-process camera
runtime to independently owned camera workers. Existing ANPR, YuNet/SFace FRS,
vehicle Re-ID, correlation, GIS, watchlists, suspicious-activity analytics and
alert logic are intentionally reused rather than redesigned in this phase.

## Safety model

- PostgreSQL `Camera.desired_state` is persistent operator intent.
- Redis short-lived leases are the authority for live camera ownership.
- The public FastAPI process no longer starts/stops live camera workers when
  `INTEL_I_DISTRIBUTED_CAMERA_WORKERS=true`.
- Upload/recorded-video processing stays on the existing path for compatibility.
- Worker preview endpoints are internal and token-protected.
- Setting `INTEL_I_DISTRIBUTED_CAMERA_WORKERS=false` retains the existing live
  camera execution path as a controlled rollback switch.

## 0. Before rollout

1. Back up PostgreSQL.
2. Preserve your current working backend build/tag.
3. Keep `.env`, RSA keys, Fernet keys, RTSP credentials and database dumps out
   of source/deployment archives.
4. Generate a shared worker preview token, for example in PowerShell:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

5. Add the variables from `DISTRIBUTED_CAMERA_ENV.example` to your secret
   manager/current environment.

## 1. Apply database migration

From the backend virtual environment:

```powershell
alembic heads
alembic upgrade head
```

Expected head:

```text
b8d4e6f1a2c3
```

The migration deliberately sets every existing camera to
`desired_state=STOPPED`; cameras will not auto-start after deployment.

## 2. First local process test: one worker, one camera

Keep distributed mode enabled in the API and worker environment.

Terminal A:

```powershell
uvicorn main:app --host 0.0.0.0 --port 3000
```

Terminal B:

```powershell
$env:INTEL_I_WORKER_ID="worker-01"
$env:INTEL_I_MAX_CAMERAS_PER_WORKER="1"
$env:INTEL_I_WORKER_PREVIEW_PORT="9101"
$env:INTEL_I_WORKER_PREVIEW_ADVERTISE_URL="http://127.0.0.1:9101"
python -m workers.camera_worker
```

Start one camera through the existing INTEL-I UI/API. Confirm:

- `/camera/{camID}/start` returns `execution=distributed` and `status=starting`.
- Redis has one `camera:lease:<camera_pk>` key.
- `/api/system/workers` shows one worker and one owned camera.
- Existing live preview, alerts, ANPR/FRS/Re-ID/GIS behavior still works.

## 3. Two-worker 10-camera acceptance test

Terminal B:

```powershell
$env:INTEL_I_WORKER_ID="worker-01"
$env:INTEL_I_MAX_CAMERAS_PER_WORKER="5"
$env:INTEL_I_WORKER_PREVIEW_PORT="9101"
$env:INTEL_I_WORKER_PREVIEW_ADVERTISE_URL="http://127.0.0.1:9101"
python -m workers.camera_worker
```

Terminal C (use another preview port on the same host):

```powershell
$env:INTEL_I_WORKER_ID="worker-02"
$env:INTEL_I_MAX_CAMERAS_PER_WORKER="5"
$env:INTEL_I_WORKER_PREVIEW_PORT="9102"
$env:INTEL_I_WORKER_PREVIEW_ADVERTISE_URL="http://127.0.0.1:9102"
python -m workers.camera_worker
```

Start 10 live cameras and confirm that each `camera_pk` has exactly one runtime
owner. The exact distribution can differ; the total per worker may not exceed
its configured capacity.

## 4. Mandatory failure tests

### FastAPI restart

Kill and restart only FastAPI. Worker camera processing must continue. Browser
streams should reconnect to the same owners after API recovery.

### Worker hard failure

Force-kill one worker (do not give it a graceful shutdown). Its leases should
expire after approximately the configured lease TTL. A healthy worker with
free capacity should claim those cameras. Verify there are no duplicate alerts.

### Repeated start

Send the same start request multiple times. Expected result: one persistent
RUNNING intent, one Redis lease, one camera owner and one analytics path.

### Stop

Stop a running camera. Its desired state must become STOPPED, the owner should
release it, and no other worker should reclaim it.

### Redis outage

Temporarily make Redis unavailable. A worker may tolerate a short hiccup, but
once it cannot prove ownership for the full lease TTL it must stop the camera
rather than risk split-brain processing.

## 5. Docker Compose local validation

The distributed compose file layers on the existing infrastructure compose:

```powershell
docker compose -f docker-compose.yml -f docker-compose.distributed.yml up --build api worker-01 worker-02
```

The shared `INTEL_I_WORKER_PREVIEW_TOKEN` must already be set in the host
environment or `.env` used only on the trusted deployment machine.

## 6. Benchmark gates

Do not jump straight to 60 cameras. Validate in this order:

```text
5 -> 10 -> 20 -> 30 -> 45 -> 60
```

Use `scripts/benchmark_multicamera.py` against your API and the Prometheus
server that scrapes **every worker process/pod**. In distributed mode, do not
benchmark from the API pod's `/metrics` alone because that cannot represent the
worker GPUs/camera loops. See `k8s/monitoring/README.md`. Record camera FPS,
queue depth, dropped frames, frame latency, GPU utilization, VRAM, AI queue
depth, ANPR latency, FRS latency, Re-ID latency and alert latency.

The configured `INTEL_I_MAX_CAMERAS_PER_WORKER` is an ownership ceiling, not a
claim about GPU throughput. Increase it only when benchmark results meet your
latency/FPS/error SLA.

## 7. Kubernetes rollout

Build and push `Dockerfile.k8s`, replace the placeholder registry in the YAML,
create the secret from your real secret manager, update `FRONTEND_URL`, then:

```bash
kubectl apply -k k8s/base
```

Start with the base two GPU workers. Only after staged acceptance testing use:

```bash
kubectl apply -k k8s/overlays/60-camera
```

The 60-camera overlay is deliberately conservative: 12 worker replicas x a
5-camera ownership ceiling = 60 assignment slots. It requires 12 schedulable
GPU resources and **does not prove** that 60 streams meet performance targets.
Benchmarking determines the final worker/GPU count.

## 8. What remains for the next phase

This release intentionally reuses the existing `main.py` analytics functions
inside the worker as a compatibility bridge. It moves live-camera ownership and
lifecycle out of FastAPI, but it does not yet complete the later optimization
of extracting every model/runtime import from `main.py` into worker-only service
modules. That second refactor should be done after the distributed ownership
acceptance tests pass.

After this phase is stable, strengthen FRS with person-track-level observations,
pose/illumination quality gates, multi-frame SFace embedding fusion and
watchlist verification. Do not change FRS and the camera ownership architecture
at the same time.
