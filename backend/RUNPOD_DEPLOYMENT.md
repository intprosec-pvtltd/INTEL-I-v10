# INTEL-I backend: RunPod deployment (rebuild-and-redeploy runbook)

This package runs FastAPI, PostgreSQL, and Redis inside one RunPod GPU
container. Uploaded videos, JWT keys, camera-source/person-embedding
encryption keys, and PostgreSQL **backups** persist under the RunPod Network
Volume mounted at `/workspace`. Kafka is intentionally disabled.

## Why the last pod broke after stop/start

RunPod pods without state pinned to a Network Volume are not guaranteed to
restart on the same physical host. When you stop a pod and RunPod cannot
reuse the original machine (capacity, maintenance, etc.), it **migrates**
the pod to different hardware and rebuilds the container from the image —
anything that only lived on the container disk is gone.

PostgreSQL's data directory (`PGDATA`) is deliberately kept on the container
disk, not `/workspace`, because RunPod Network Volumes generally cannot be
`chown`'d to the `postgres` UID that PostgreSQL requires. That is correct and
still true — but it means a migration wipes the database, so the app comes
up with an empty/uninitialized DB and starts throwing errors. That is almost
certainly what happened this morning.

**Fix applied in this repo:** `docker-entrypoint.sh` now backs up the
database to `/workspace/backups` on a timer and on every graceful stop, and
automatically restores the newest backup on boot if it finds an empty
database. A migration now costs you at most `AUTO_BACKUP_INTERVAL_SECONDS`
(default 30 min) of data instead of everything. See `.env.example` for the
two new variables. This does not fix the underlying RunPod behavior — it
makes the app resilient to it.

## Before building

1. Confirm every model file listed in `.env.example` / `scripts/preflight_models.py`
   is present under `models/` (they already are in this checkout).
2. Review `.env.example` and decide real values for the RunPod template —
   don't bake them into the image.
3. Never rename `.env.example` to `.env` inside the image and never commit
   real secrets.

## 1. Build the image

Run this from the `backend/` folder, on a machine with Docker installed
(Docker Desktop on Windows/macOS, or Docker Engine on Linux). A GPU is not
required to build, only to run.

```bash
docker build -t YOUR_DOCKERHUB_USER/intel-i-backend:v4 .
```

Use a **new tag every time** (`v4`, `v5`, ...). Reusing a tag lets RunPod (and
your own machine) serve a stale cached image.

## 2. Push to a registry

```bash
docker login
docker push YOUR_DOCKERHUB_USER/intel-i-backend:v4
```

(Any registry RunPod can pull from works — Docker Hub, GHCR, RunPod's own
registry. Docker Hub is the simplest default.)

## 3. Create the RunPod pod

- **Container image:** `YOUR_DOCKERHUB_USER/intel-i-backend:v4`
- **GPU:** any CUDA 12.4-compatible GPU (e.g. RTX 4090, A5000, A40). Pick a
  GPU type/region combination with good availability to reduce how often
  RunPod needs to migrate you.
- **Container disk:** 40 GB or more (holds the OS, venv, models baked into
  the image, and PostgreSQL's live data directory).
- **Network Volume:** 50 GB or more, mounted at `/workspace`. **This is
  required** — it is where backups, uploads, and encryption keys live.
- **Exposed HTTP port:** `8000`
- **Do not expose** ports `5432` (Postgres) or `6379` (Redis) publicly.
- **Environment variables:** copy every variable from `.env.example` into the
  RunPod template with real values. At minimum set real values for
  `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, and `FRONTEND_URL` (the entrypoint
  refuses to start without these three).

## 4. First boot verification

Open the pod's web terminal (or `runpodctl`/SSH) and check:

```bash
nvidia-smi
curl -f http://127.0.0.1:8000/health
pg_isready -h 127.0.0.1 -p 5432 -U "$POSTGRES_USER" -d "$POSTGRES_DB"
redis-cli -a "$REDIS_PASSWORD" --no-auth-warning ping
python -c "import onnxruntime as o; print(o.get_available_providers())"
ls -la /workspace/backups
```

`CUDAExecutionProvider` must appear in the ONNX Runtime provider list.
`VEHICLE_ATTRIBUTE_DEVICE=cpu` is intentional — the packaged PaddlePaddle
runtime is CPU-only. YOLO, ONNX OCR, and DarkIR still use GPU.

On a brand-new deployment `/workspace/backups` will be empty until the first
auto-backup runs (or you take one manually — see below). That is expected
and is logged by the entrypoint as "starting fresh".

## 5. Stopping and restarting safely

- Prefer **Stop** over **Terminate** in the RunPod dashboard — the entrypoint
  traps the stop signal and takes a final backup before exiting.
- Before a stop you know is risky (e.g. RunPod is showing capacity warnings,
  or you're about to change instance type), take a manual backup first so
  you don't depend on the timer:

  ```bash
  bash /app/scripts/db_backup.sh
  ```

- On restart, watch the entrypoint log. If it says "Empty database detected
  ... restoring latest backup", the pod migrated and the auto-restore
  handled it. If it says "Empty database and no backup found", something is
  wrong — check `/workspace/backups` and restore manually:

  ```bash
  ls -t /workspace/backups/*.dump
  bash /app/scripts/db_restore.sh /workspace/backups/<the file you want>.dump
  ```

## 6. Manual backup / restore reference

```bash
# Backup now
bash /app/scripts/db_backup.sh

# Restore a specific dump
bash /app/scripts/db_restore.sh /workspace/backups/intel_i_backup_YYYYMMDD_HHMMSS.dump
```

Both scripts require `DATABASE_URL`, which the entrypoint already exports
inside the running container.

## Troubleshooting

- **App fails immediately with "POSTGRES_PASSWORD must be configured"** —
  the RunPod template is missing an env var. Fix the template, not the image.
- **App starts but no camera/model data** — you're on a fresh Network Volume
  with no backups yet; this is expected on first deploy only.
- **`CUDAExecutionProvider` missing** — the GPU RunPod assigned doesn't have
  a matching driver for CUDA 12.4, or you migrated onto a non-GPU/older host.
  Stop and restart the pod requesting the same GPU type explicitly.
- **Backups directory has old dumps but restore didn't run** — the database
  had at least one table already (a partial/corrupt init) so the entrypoint
  assumed it was intentional and refused to overwrite it automatically.
  Restore manually with step 6 above.
