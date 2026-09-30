# INTEL-I uploaded-source implementation report

## Actual architecture found

The uploaded backend already has a distributed camera worker, central inference worker, isolated ANPR OCR worker, isolated FRS worker, a Triton Python backend, a Triton gRPC client, dynamic batching, conditional ROI processing, track caches, ANPR temporal voting, OpenSearch retrieval and an Ollama-backed assistant. Kubernetes and Compose definitions exist in multiple generations. The main pipeline still contains substantial state and model ownership in `main.py`; the Triton pipeline embeds the existing inference worker, so its Python backend is **not proof of TensorRT execution**. The `intel_i_frs` and `awiros_anpr_ocr` Triton models are also Python backends. The repository has build scripts and model repository configuration, but no model weights or L40S execution results.

Camera frames move through camera-worker submission to the inference service or Triton compatibility backend. The latter calls the existing analytics runtime, which applies detector, tracking, conditional ROI branches, ANPR, FRS, Re-ID, correlation and evidence logic. The isolated OCR worker uses Awiros/Paddle. FRS uses the configured ONNX embedder and a synchronized tenant-scoped watchlist. The assistant retrieves tenant-filtered OpenSearch hits and uses Ollama when enabled; operational intent also uses direct database queries.

## Implemented and statically validated in this revision

- RAG operational hits are checked against the current tenant-owned database rows before their indexed text is sent to Ollama. Deleted or reassigned rows are omitted. Knowledge documents remain separate.
- The remote FRS path now requires independent source frames of the same enhanced-only identity before returning an alert candidate. It no longer scans an entire camera frame when tracking data is absent. Existing original-image matching retains priority. DarkIR remains conditional on verified face crops, camera policy, low light, cooldown and the shared AdvancedIntelligenceEngine restorer; no new model owner was added. A regression test covers the three-frame decision, but could not execute in this environment.
- The ANPR OCR endpoint reads request bodies with an incremental byte limit and rejects oversized decoded images. FRS rejects oversized decoded face images.
- A sanitized production environment example records the model mounts and relevant feature switches.

## Preserved existing

ANPR DarkIR uses `main.py`'s `advanced_intelligence.restore_low_light` on an eligible vehicle ROI. FRS DarkIR uses `services/faceEnhancement.py` and the same `advanced_intelligence` object through `_face_darkir_gate()` in `main.py`. The remote bridge calls that gate after original recognition fails. The gate rejects tiny, severely blurred, already good and out-of-policy crops; it limits attempts per track and declines late or invalid restoration. It is an elapsed-time fallback, not a hard cancellation of a synchronous GPU call. The original source frame stays canonical for evidence. The current `main.py` local FRS path already applies an enhanced-only independent-frame gate. No model binary was included or reconstructed.

## Validation run here

- Backend `compileall` and explicit AST parsing of changed Python files: passed.
- Frontend `npm ci` and `npm run build`: passed; bundle size warning.
- Frontend `npm run lint`: passed with 56 existing warnings and zero errors.
- Backend pytest suite: not run because this runtime lacks pytest, OpenCV, SQLAlchemy, FastAPI, httpx and model dependencies.
- No actual OCR/FRS accuracy, latency, load, 60-camera capacity, L40S memory or TensorRT engine compilation was measured.

## Remaining production work and target validation

- Build and integrity-check the actual model weights on the L40S. Run the existing `tools/build_triton_tensorrt.py`, `tools/validate_tensorrt.py` and `tools/benchmark_inference_capacity.py` against the real deployment and inspect every model's actual execution provider. A Triton Python backend is not a native TensorRT engine.
- Choose one mutually exclusive deployment generation and GPU ownership map. `docker-compose.distributed.yml` gives GPU access to camera, OCR and FRS workers; `docker-compose.triton.yml` uses a separate Triton process. Do not combine these files or simultaneously deploy standalone FRS/inference owners and their Triton equivalents without measuring memory and state consistency.
- Run Alembic migrations, internal worker readiness and authorized end-to-end tests with real models, database, RTSP feeds and alert delivery. Test daylight/dark face non-match and match, tiny/blurred faces, vehicle and bike plates, worker failure/recovery and cross-camera correlation.
- Run 60 representative RTSP streams under sustained load on the target L40S. Measure dropped/stale frames, latency percentiles, GPU memory, alert correctness and OCR/FRS error rates. There is no basis yet for a 60-camera guarantee.
- Archived `.env.before-*` files in the uploaded backend contained credential values. They are excluded from the deliverable. Rotate any secrets actually used by the deployed system, including worker tokens and third-party credentials.

## Deployment notes

The final package deliberately excludes `.env*` backups, models, vendor runtimes, venv, node_modules, build output, caches and generated engine files. Supply model and vendor mounts before startup. Use one runtime mode in `INTEL_I_INFERENCE_MODE`; keep tokens in Secrets, not the source package. Start databases and migrations before internal workers, then API and frontend. Verify `/health/ready` and worker-specific internal health with authorized tokens. The existing `k8s/README.md` and `k8s/RUNBOOK.md` contain deployment commands; inspect and adapt them to the chosen generation before applying manifests.
