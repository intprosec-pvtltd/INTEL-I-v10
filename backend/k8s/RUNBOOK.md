# AWS/Linux execution order

Run from the parent of the new `k8s/` folder on your EC2 host, with kubectl pointed at the intended cluster. Keep the working cluster/driver installation. Commands assume Bash, Python 3 + PyYAML, OpenSSL, Helm and kubectl. They do not create EC2 resources or publish a public endpoint.

## 0. Replace folder and verify prerequisites

Keep an archival copy of your previous folder, then extract this ZIP so that there is exactly one new `k8s/`. Do not merge with old manifests.

```bash
sudo apt-get update
sudo apt-get install -y python3-yaml openssl curl
kubectl config current-context
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods -A
nvidia-smi
helm version
python3 k8s/scripts/validate-package.py
```

Pass: your intended single AWS node is Ready, nvidia-smi shows one L40S, and package validation passes. If kubectl cannot reach a cluster, stop: this package assumes Kubernetes already exists. Docker GPU support alone does not configure Kubernetes/containerd GPU support.

Select the exact node name from the output (this deliberately does not guess):

```bash
read -r -p 'Exact Kubernetes node name: ' GPU_NODE
export GPU_NODE
kubectl label node "$GPU_NODE" intel-i/node=single-l40s --overwrite
kubectl get nodes -l intel-i/node=single-l40s
kubectl describe node "$GPU_NODE"
```

Pass: exactly one labeled node. Inspect taints, free CPU/RAM and allocatable resources. The package requests about 3.35 CPU cores and 10.25 GiB RAM before Kubernetes overhead, temporary jobs and caches; workload usage can be much higher. No CPU limits are imposed; memory limits are per-container, not a promise that every limit fits simultaneously. Do not remove arbitrary taints. On a dedicated kubeadm one-node test cluster only, a control-plane NoSchedule taint must be deliberately removed or tolerated before application scheduling.

## 1. Configure the Kubernetes GPU runtime

First inspect existing installation:

```bash
kubectl get runtimeclass
kubectl get daemonsets -A
helm list -A
```

If a working NVIDIA runtime and plugin already exist, keep them and use the corresponding sharing configuration below. Never install a second device plugin beside an existing one.

For a standard containerd host with NVIDIA Container Toolkit already installed, and no GPU Operator managing it:

```bash
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
kubectl apply -f k8s/gpu/runtime-class.yaml
kubectl wait --for=condition=Ready "node/$GPU_NODE" --timeout=180s
```

This restart affects container runtime availability; run before deploying the stack. For K3s, do not apply the standard containerd restart/configuration blindly: K3s owns its generated containerd config. Have NVIDIA Container Toolkit installed and visible in the K3s service PATH, then restart `k3s` and verify `kubectl get runtimeclass nvidia`. For an existing GPU Operator, let it manage the runtime. Continue only when the `nvidia` runtime handler is actually installed; creating RuntimeClass YAML alone does not install it.

## 2. Configure three GPU sharing slots

### Path A — standalone plugin, only when no plugin/operator is installed

```bash
kubectl create namespace nvidia-device-plugin --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/gpu/time-slicing-config.yaml
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin --version 0.17.1 \
  -f k8s/gpu/device-plugin-values.yaml --wait --timeout 5m
kubectl -n nvidia-device-plugin get pods -o wide
```

If a standalone plugin already exists, update its existing Helm release/namespace using these same config contents and the release's compatible chart version; do not run the new installation alongside it. Change the ConfigMap namespace to that existing namespace.

### Path B — existing GPU Operator only

Read its namespace from `kubectl get daemonsets -A`. The following assumes it is `gpu-operator` and the ClusterPolicy is `cluster-policy`; substitute the actual names if different.

```bash
sed 's/namespace: nvidia-device-plugin/namespace: gpu-operator/' \
  k8s/gpu/time-slicing-config.yaml | kubectl apply -f -
kubectl patch clusterpolicy cluster-policy --type merge \
  -p '{"spec":{"devicePlugin":{"config":{"name":"intel-i-time-slicing","default":"single-l40s"}}}}'
kubectl -n gpu-operator rollout restart daemonset/nvidia-device-plugin-daemonset
kubectl -n gpu-operator rollout status daemonset/nvidia-device-plugin-daemonset --timeout=300s
```

Do not run both paths. Now verify:

```bash
kubectl get node "$GPU_NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
nvidia-smi --query-gpu=name,uuid --format=csv
```

Pass: Kubernetes advertises `3`; the host still reports ONE physical GPU. These slots share all GPU memory and compute. They are not isolated 16 GB partitions, and three slots do not triple performance. See [NVIDIA's explanation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html).

## 3. Prepare disk directories, namespace and local secrets

Run on the actual labeled node:

```bash
bash k8s/scripts/prepare-storage.sh
kubectl apply -f k8s/namespace.yaml
```

Pass: `/var/lib/intel-i` is on a persistent filesystem with adequate space. The four PV capacity declarations sum to 95 GiB; reserve additional space for images, models, caches and OS. If using a dedicated EBS volume, mount it at `/var/lib/intel-i` and make the mount persistent BEFORE this command. The script never formats or mounts disks. Kafka directory ownership is set to UID/GID 1000 used by the selected image. PostgreSQL and Redis entrypoints handle their own data ownership.

For a genuinely fresh database and new camera/watchlist records:

```bash
python3 k8s/scripts/create-secrets.py --fresh-database
# Review locally only; do not paste the contents into chat.
chmod 600 k8s/secrets/intel-i-secrets.local.yaml
kubectl apply -f k8s/secrets/intel-i-secrets.local.yaml
kubectl -n intel-i get secret intel-i-secrets
```

Pass: the Secret exists. The generator refuses to overwrite a local file or an existing cluster Secret. It generates RSA JWT keys, shared symmetric auth key aliases, random service tokens and Fernet-form encryption keys. Check the auth variable names, key format and JWT algorithm against your existing backend before first launch.

**If reusing/restoring your existing database, do not generate new encryption keys.** Instead:

```bash
cp k8s/secrets/intel-i-secrets.example.yaml k8s/secrets/intel-i-secrets.local.yaml
chmod 600 k8s/secrets/intel-i-secrets.local.yaml
nano k8s/secrets/intel-i-secrets.local.yaml
kubectl apply -f k8s/secrets/intel-i-secrets.local.yaml
```

Fill every placeholder locally. Preserve the exact existing CAMERA_SOURCE_KEY and PERSON_EMBEDDING_KEY, auth keys and appropriate service tokens. DATABASE_URL must use host `postgres`, port 5432, database `cctv_db`, and match POSTGRES_PASSWORD; percent-encode special characters in the URI password. Multiline RSA keys require YAML `|` blocks. If your auth code uses a different environment variable, add it. PostgreSQL's initialization password does not rotate an already-initialized DB password. Back up the secret file privately and never include it in shared ZIPs.

## 4. Supply and verify application images

Use your existing built application images. This package does not fabricate a Dockerfile without your dependencies/source. Required image contract:

| Image | Contract |
|---|---|
| API | Existing entrypoint binds 0.0.0.0:3000; Python; Alembic/migration config; `minio` SDK; `/health/live` and `/health/ready` |
| Camera | `python -m workers.camera_worker`; preview binds 0.0.0.0:9101; Python, sh, curl; CUDA inference dependencies; relevant detector weights |
| ANPR | `python -m workers.anpr_ocr_worker`; Awiros weights, dictionary, vendor/PaddleOCR and GPU-enabled Paddle stack; Python, sh, curl |
| FRS | `python -m workers.frs_worker`; actual model files named in ConfigMap; CUDA-capable inference implementation; Python, sh, curl |

All application images must have their working directory set to the backend root. The mount `/app/object-cache` must be writable for the image's user. No GPU runtime is requested for the API, so keep it in control-plane mode. The inherited API entrypoint must not start in-process camera analytics or require CUDA.

Provide your actual ECR URI and immutable tag/digest for each role:

```bash
read -r -p 'API ECR image URI with tag/digest: ' API_IMAGE
read -r -p 'Camera worker ECR image URI with tag/digest: ' CAMERA_IMAGE
read -r -p 'ANPR ECR image URI with tag/digest: ' ANPR_IMAGE
read -r -p 'FRS ECR image URI with tag/digest: ' FRS_IMAGE
python3 k8s/scripts/configure-images.py \
  --api "$API_IMAGE" --camera-worker "$CAMERA_IMAGE" \
  --anpr-worker "$ANPR_IMAGE" --frs-worker "$FRS_IMAGE"
nano k8s/configmaps/platform.yaml
```

Expected URI shape: `<aws-account>.dkr.ecr.<region>.amazonaws.com/intel-i-api:<immutable-tag>`. No AWS account is invented. Review face/Awiros paths and copy other required NON-SECRET settings from your working backend. Add the actual frontend origin using the environment variable your application reads; no CORS variable name is guessed here. The original FRS model filenames are retained, not assumed to be AdaFace/SCRFD or GPU-compatible.

For ECR use an existing working kubelet credential provider/IAM configuration if available. `docker login` by itself does not authenticate Kubernetes pulls. For a short test without a credential provider, create an image pull Secret through stdin (AWS CLI required):

```bash
read -r -p 'AWS region: ' AWS_REGION
read -r -p 'ECR registry hostname (no scheme/repository): ' ECR_REGISTRY
export AWS_REGION ECR_REGISTRY
aws ecr get-login-password --region "$AWS_REGION" | python3 -c '
import base64,json,os,sys
p=sys.stdin.read().strip()
assert p, "No ECR password returned"
a=base64.b64encode(("AWS:"+p).encode()).decode()
d=json.dumps({"auths":{os.environ["ECR_REGISTRY"]:{"auth":a}}})
print(json.dumps({"apiVersion":"v1","kind":"Secret","metadata":{"name":"ecr-pull","namespace":"intel-i"},"type":"kubernetes.io/dockerconfigjson","data":{".dockerconfigjson":base64.b64encode(d.encode()).decode()}}))
' | kubectl apply -f -
kubectl -n intel-i patch serviceaccount default --type merge \
  -p '{"imagePullSecrets":[{"name":"ecr-pull"}]}'
```

Refresh that temporary pull credential before later image pulls when it expires; use the kubelet provider for ongoing operation. Do not run with shell tracing (`set -x`).

## 5. Remove old autoscaling and render

If the old package has already been applied, remove only its scaling/disruption objects:

```bash
kubectl -n intel-i delete hpa frs-worker --ignore-not-found
kubectl -n intel-i delete pdb frs-worker --ignore-not-found
# Stop old GPU allocations before installing the new three-slot layout.
kubectl -n intel-i get deployments
```

If camera-worker, anpr-worker and frs-worker already exist, scale these existing deployments to zero now:

```bash
kubectl -n intel-i scale deployment/camera-worker deployment/anpr-worker deployment/frs-worker --replicas=0
kubectl -n intel-i wait --for=delete pod -l 'app in (camera-worker,anpr-worker,frs-worker)' --timeout=180s
```

Skip those two commands on a fresh installation. Do not delete the old FRS model PVC/data as part of this change. If infrastructure already exists under these names, do not overwrite it blindly: reconcile existing claims, passwords and data first.

```bash
python3 k8s/scripts/validate-package.py
kubectl kustomize k8s > /tmp/intel-i-rendered.yaml
kubectl apply --dry-run=server -f /tmp/intel-i-rendered.yaml
```

Pass: all objects accepted. This dry run includes infrastructure and apps but not the example secrets, NVIDIA installer, migration jobs or optional components. Initial deployment must use the staged script, not a simultaneous `kubectl apply -k`.

## 6. Deploy in order with validation at every stage

```bash
bash k8s/scripts/deploy.sh
```

The script stops on failure and runs these gates:

| Stage | Commands executed / pass condition |
|---|---|
| Node and secrets | Node Ready; exactly three shared slots; required secret values present, not printed |
| Config/storage/policy | Apply ConfigMaps, static PVs and ingress policies |
| Postgres | Apply PVC/Service/Deployment; rollout available; PVC Bound |
| Redis | Same; rollout available and PVC Bound |
| Kafka | Same; broker API probe succeeds and PVC Bound |
| MinIO | Same; HTTP health succeeds and PVC Bound |
| Internal DNS/TCP | Diagnostic pod connects to postgres:5432, redis:6379, kafka:9092, minio:9000; four TCP OK messages |
| MinIO bucket | API-image Job verifies/creates intel-i-uploaded-videos; Job Complete |
| Database schema | API-image Job runs python -m alembic upgrade head; Job Complete |
| API | Deployment submitted first; readiness may wait for remote workers |
| ANPR | Startup/readiness pass; Deployment available |
| FRS | Startup/readiness pass; Deployment available |
| Camera | Startup/readiness pass; Deployment available |
| API final | /health/ready succeeds; Deployment available |
| GPU and services | nvidia-smi in each GPU pod; worker HTTP health; API ready; Postgres/Redis/Kafka checks |

After the script, the separate repeatable validation command is:

```bash
bash k8s/scripts/validate-live.sh
kubectl -n intel-i get events --sort-by=.lastTimestamp
kubectl -n intel-i get hpa
```

Pass: eight Deployments at `1/1`, four PVCs Bound, no HPA, no unexpected restart loops. Three workers show the same physical L40S UUID. CUDA visibility alone does NOT prove models run on CUDA. Inspect worker initialization logs locally:

```bash
kubectl -n intel-i logs deployment/anpr-worker --tail=150
kubectl -n intel-i logs deployment/frs-worker --tail=150
kubectl -n intel-i logs deployment/camera-worker --tail=150
```

Pass: strict ANPR/FRS initialization succeeds and logs/health identify the real GPU backend; models loaded; no CPU fallback; no OOM. Check GPU utilization while processing actual input:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 2
```

Ctrl+C stops the watch. Health returning READY is not sufficient evidence of recognition quality.

## 7. Connect your existing frontend and test one uploaded video

On EC2, keep this command running:

```bash
kubectl -n intel-i port-forward --address 127.0.0.1 service/intel-i-api 3000:3000
```

From Windows in a separate PowerShell terminal, forward a free local port:

```powershell
ssh -i "C:\path\your-key.pem" -L 3000:127.0.0.1:3000 ubuntu@YOUR_EC2_PUBLIC_HOST
```

Your local frontend can then use `VITE_BASE_URL=http://localhost:3000` and `VITE_WEBSOCKET_URL=ws://localhost:3000/ws`. Restart Vite after changing its environment. Stop any old local backend occupying port 3000 first. Configure the actual frontend origin in backend CORS settings. This is an authenticated SSH test path; no worker/infrastructure public ports are needed.

Upload one small MP4 through the existing UI. Pass: upload READY, object in MinIO, camera worker claims its cam_id, materializes the object in `/app/object-cache`, preview displays while processing, and expected analytics/alerts are produced. Check:

```bash
kubectl -n intel-i logs deployment/intel-i-api --since=10m
kubectl -n intel-i logs deployment/camera-worker --since=10m
```

Do not print credential-bearing environment variables. A 404 preview after a finite upload reaches EOF is not automatically a Kubernetes failure: correlate worker ownership/lease and pipeline exit timestamps first.

## 8. Test one Sentinel RTSP stream

Add ONE known working RTSP source through the UI using your credentials privately. Keep capacity at two (at most one upload plus one live stream). Pass: RTSP TCP connects, frames progress, preview loads through the API, ANPR/FRS receive requests, and evidence timestamps are valid. DNS/egress is open in these policies; AWS routing/security groups and source-server permissions must still allow the feed. Do not paste RTSP URLs containing credentials.

## 9. Verify replacement and lease reclaim

Use the continuously running RTSP source, not an already-finished MP4:

```bash
OLD_POD=$(kubectl -n intel-i get pod -l app=camera-worker -o jsonpath='{.items[0].metadata.name}')
kubectl -n intel-i delete pod "$OLD_POD"
kubectl -n intel-i rollout status deployment/camera-worker --timeout=900s
kubectl -n intel-i get pods -l app=camera-worker -o wide
kubectl -n intel-i logs deployment/camera-worker --since=10m
```

Pass: replacement pod has a new name/IP, starts successfully, claims the same running camera after release/lease expiry, and API preview resumes using its new advertised POD_IP. Only one owner processes the camera. Measure actual recovery time; it depends on the application's lease/heartbeat TTL. With one replica this is replacement/reclaim with downtime, not uninterrupted failover to an already-running second worker. Do not increase replicas until this and both video tests pass.

## 10. Optional components

Onboarding worker (after the core eight services pass):

```bash
kubectl apply -f k8s/optional/onboarding.yaml
kubectl -n intel-i rollout status deployment/camera-onboarding-worker --timeout=300s
kubectl -n intel-i logs deployment/camera-onboarding-worker --tail=100
```

This is a ninth non-GPU service. No unsupported HTTP health endpoint was invented for it; inspect its real loop/heartbeat behavior.

Monitoring is optional: first verify API `/metrics` exists and check that the Prometheus Operator CRD is installed:

```bash
kubectl get crd servicemonitors.monitoring.coreos.com
kubectl -n intel-i exec deployment/intel-i-api -- python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:3000/metrics",timeout=5).status)'
```

Then adapt the ServiceMonitor metadata to your Prometheus `serviceMonitorSelector` and namespace selector, and the network policy to your actual Prometheus namespace/pod labels:

```bash
kubectl apply -f k8s/monitoring/service-monitor.yaml -f k8s/monitoring/network-policy.yaml
```

Pass: target appears UP in Prometheus. If `/metrics` is absent, leave the monitor disabled; it requires an application endpoint.

## Troubleshooting: collect actual output

```bash
kubectl -n intel-i get pods -o wide
kubectl -n intel-i get events --sort-by=.lastTimestamp
kubectl -n intel-i describe pod POD_NAME
kubectl -n intel-i logs POD_NAME --tail=100
kubectl -n intel-i logs POD_NAME --previous --tail=100
```

Review/redact logs locally before sharing; application errors may contain credentials or camera URLs. Never send Secret YAML, complete environment dumps or raw connection URLs.

| Symptom | Next concrete check |
|---|---|
| Pending / insufficient nvidia.com/gpu | Node must advertise 3; no old workers or other GPU consumers; only one device plugin |
| No runtime handler nvidia | RuntimeClass exists but actual containerd/K3s handler is absent |
| PVC Pending | Labeled node matches PV affinity, directories exist on that node; consumer pod can schedule |
| Kafka permission denied | `/var/lib/intel-i/kafka` ownership and underlying mount permissions |
| ImagePullBackOff | Exact ECR image URI, repository access, kubelet credentials or refreshed pull Secret |
| Exec probe curl not found | Rebuild that image with curl; do not remove auth or disable probes |
| 401/403 internal health | Matching Secret token values loaded in server/client; restart pods after secret changes |
| HTTP 200 but not Ready | Actual JSON status, model startup, Redis/DB dependencies; parser expects READY |
| FRS/ANPR strict CUDA failure | Correct GPU framework/provider and actual model code; scheduling alone is insufficient |
| API readiness waits | Remote workers, DB schema, Redis/Kafka and MinIO health; inspect actual body/logs |
| Migration fails | Job logs locally; correct schema/migration issue, never stamp over it |
| Preview 404 | Current cam_id owner and lease, POD_IP, live pipeline vs finite-video EOF |
| Worker OOM / GPU memory exhausted | Model memory/VRAM contention; keep two-camera cap, measure workload |
| NetworkPolicy seems ineffective | Check CNI supports/enforces policy; manifests alone cannot enable enforcement |

After ConfigMap/Secret changes, restart affected application deployments in order (API, ANPR, FRS, camera), using one-at-a-time restarts and checking each rollout. Do not run simultaneous GPU rollout restarts. Local PV data and Secrets must survive ordinary pod replacements. Deleting the namespace is not a reset procedure: it deletes Secrets and claims and can strand retained PVs.
