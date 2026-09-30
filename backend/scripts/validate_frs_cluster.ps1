param(
    [string]$Namespace = "intel-i",
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"

kubectl version --client | Out-Null
kubectl cluster-info | Out-Null
kubectl apply -k k8s
kubectl -n $Namespace rollout status deployment/frs-worker --timeout="${TimeoutSeconds}s"
kubectl -n $Namespace rollout status deployment/camera-worker --timeout="${TimeoutSeconds}s"

$frsPods = kubectl -n $Namespace get pods -l app=frs-worker -o json | ConvertFrom-Json
if ($frsPods.items.Count -lt 2) { throw "Expected at least two FRS pods" }
foreach ($pod in $frsPods.items) {
    $ready = $pod.status.containerStatuses[0].ready
    if (-not $ready) { throw "FRS pod $($pod.metadata.name) is not ready" }
}

$gpu = kubectl -n $Namespace exec deployment/frs-worker -- python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
if ($gpu -notmatch "CUDAExecutionProvider") { throw "CUDAExecutionProvider is not available in the FRS pod" }

Write-Host "FRS Kubernetes validation passed: replicas ready, rollout healthy, CUDA provider available."
