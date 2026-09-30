#!/usr/bin/env bash
set -euo pipefail
kubectl -n intel-i get deployments,pods,services,pvc -o wide
for name in intel-i-api anpr-worker frs-worker camera-worker; do
 kubectl -n intel-i rollout status "deployment/$name" --timeout=120s
done
for name in anpr-worker frs-worker camera-worker; do
 echo "GPU visibility: $name"
 kubectl -n intel-i exec "deployment/$name" -- nvidia-smi --query-gpu=name,uuid,memory.used,memory.total --format=csv
done
kubectl -n intel-i exec deployment/camera-worker -- sh -ec 'curl -fsS --max-time 5 -H "X-Intel-I-Worker-Token: $INTEL_I_WORKER_PREVIEW_TOKEN" http://127.0.0.1:9101/internal/health'
kubectl -n intel-i exec deployment/anpr-worker -- sh -ec 'curl -fsS --max-time 5 -H "X-Intel-I-OCR-Token: $ANPR_OCR_INTERNAL_TOKEN" http://127.0.0.1:9201/internal/health'
kubectl -n intel-i exec deployment/frs-worker -- sh -ec 'curl -fsS --max-time 5 -H "X-Intel-I-FRS-Token: $FRS_INTERNAL_TOKEN" http://127.0.0.1:9202/internal/health'
kubectl -n intel-i exec deployment/intel-i-api -- python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:3000/health/ready",timeout=10).status)'
kubectl -n intel-i exec deployment/postgres -- sh -ec 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
kubectl -n intel-i exec deployment/redis -- redis-cli ping
kubectl -n intel-i exec deployment/kafka -- /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --list
