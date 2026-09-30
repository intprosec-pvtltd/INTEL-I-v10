# Validation record

Performed while preparing this package:

- Parsed all 47 YAML documents (including optional manifests and Helm values).
- Built the default package successfully with `kubectl kustomize` v1.34.1: 37 Kubernetes resources.
- Confirmed eight default Deployments, each with exactly one replica and Recreate strategy.
- Confirmed three GPU workers, each requesting/limiting one GPU slot and selecting RuntimeClass nvidia.
- Confirmed no default HPA, no FRS model PVC and no example secret in the default Kustomization.
- Checked every Kustomization resource path, deployment selector/pod-label match and default container probe set.
- Checked non-secret ConfigMap values are strings.
- Parsed Python helper syntax and checked Bash script syntax.
- Exercised readiness parser with compact/spaced/lowercase READY JSON, non-ready state, missing state, malformed JSON and non-object JSON.
- Ran image configuration on a temporary copy and verified replacement in applications, jobs and optional onboarding manifest.

Not performed here: Kubernetes API-server schema/admission dry run, Docker image pulls/builds, model/dependency validation, CUDA execution, database migrations, network-policy enforcement, actual service health, camera tests, lease recovery, disk persistence across host lifecycle or performance measurements. The supplied runbook contains those gates. There was no connection to your AWS account or running cluster.

Application image references and secret placeholders intentionally require local configuration. No actual credentials are included. The default app ConfigMap preserves original face model filenames; GPU recognition compatibility must be demonstrated by your actual implementation.
