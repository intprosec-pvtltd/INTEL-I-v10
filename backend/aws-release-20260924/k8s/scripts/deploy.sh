#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/validate-package.py
if grep -Rq 'image:.*REPLACE' api camera-worker anpr-worker frs-worker jobs; then
  echo 'Configure your application images first.' >&2; exit 1
fi
kubectl get nodes -l intel-i/node=single-l40s -o json | python3 -c '
import json,sys
nodes=json.load(sys.stdin)["items"]
assert len(nodes)==1, "Label exactly one node"
n=nodes[0];a=n["status"]["allocatable"]
assert int(a.get("nvidia.com/gpu",0))==3, "Expected three shared GPU slots"
assert any(c["type"]=="Ready" and c["status"]=="True" for c in n["status"]["conditions"])
print("Node Ready; three shared GPU slots advertised")'
kubectl apply -f namespace.yaml
kubectl -n intel-i get secret intel-i-secrets -o json | python3 -c '
import json,sys,base64
s=json.load(sys.stdin)["data"]
required="POSTGRES_PASSWORD DATABASE_URL CAMERA_SOURCE_KEY PERSON_EMBEDDING_KEY MINIO_ACCESS_KEY MINIO_SECRET_KEY ANPR_OCR_INTERNAL_TOKEN FRS_INTERNAL_TOKEN INTEL_I_WORKER_PREVIEW_TOKEN".split()
for k in required:
 v=base64.b64decode(s.get(k,"")).decode()
 assert v and "REPLACE" not in v, "Missing/unconfigured secret key: "+k
print("Required secret fields present; values hidden")'
kubectl apply -f configmaps/
kubectl apply -f storage/
kubectl apply -f network-policy.yaml
for name in postgres redis kafka minio; do
  kubectl apply -f "$name/pvc.yaml" -f "$name/service.yaml" -f "$name/deployment.yaml"
  kubectl -n intel-i rollout status "deployment/$name" --timeout=600s
  kubectl -n intel-i get pvc "$name-data"
done
# A short-lived pod using your API image tests service DNS/TCP from the application namespace.
kubectl -n intel-i delete pod intel-i-diagnostics --ignore-not-found --wait=true
API_IMAGE=$(python3 -c 'import yaml; print(next(yaml.safe_load_all(open("api/deployment.yaml")))["spec"]["template"]["spec"]["containers"][0]["image"])')
kubectl -n intel-i run intel-i-diagnostics --image="$API_IMAGE" --restart=Never --labels=app=intel-i-diagnostics --command -- python -c 'import socket; [(socket.create_connection((h,p),10).close(),print(h,"TCP OK",flush=True)) for h,p in [("postgres",5432),("redis",6379),("kafka",9092),("minio",9000)]]'
kubectl -n intel-i wait --for=jsonpath='{.status.phase}'=Succeeded pod/intel-i-diagnostics --timeout=180s
kubectl -n intel-i logs intel-i-diagnostics
kubectl -n intel-i delete pod intel-i-diagnostics
for job in minio-init migrate; do
  kubectl -n intel-i delete job "intel-i-$job" --ignore-not-found --wait=true
  kubectl apply -f "jobs/$job.yaml"
  if ! kubectl -n intel-i wait --for=condition=complete "job/intel-i-$job" --timeout=600s; then
    echo "Job failed. Inspect locally: kubectl -n intel-i logs job/intel-i-$job" >&2
    exit 1
  fi
done
kubectl apply -f api/deployment.yaml
# API /health/ready may depend on the remote workers, so wait after they start.
for name in anpr-worker frs-worker camera-worker; do
  kubectl apply -f "$name/deployment.yaml"
  kubectl -n intel-i rollout status "deployment/$name" --timeout=900s
done
kubectl -n intel-i rollout status deployment/intel-i-api --timeout=600s
bash scripts/validate-live.sh
