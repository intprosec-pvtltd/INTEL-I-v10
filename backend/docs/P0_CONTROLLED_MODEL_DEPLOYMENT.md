# P0 controlled model deployment acceptance

## Required controls

A model is production-ready only when it is enabled, configured, present, checksum-pinned, checksum-valid and loadable by its production wrapper. `scripts/provision_models.py` downloads only over HTTPS, requires an allowlisted source host, bounds download size, verifies SHA-256 and expected size when available, and atomically replaces the destination only after validation.

Production sets `MODEL_REQUIRE_PINNED_SHA256=true`, which makes an enabled model with no pinned digest a startup/preflight failure.

## Windows acceptance commands

```powershell
# 1. Pin the exact EDSR artifact used by this deployment.
(Get-FileHash .\models\EDSR_x4.pb -Algorithm SHA256).Hash.ToLower()
# Copy the result to EDSR_MODEL_SHA256 in your local secret-backed environment.

# 2. Verify all controlled/core model artifacts.
python .\scripts\preflight_models.py

# 3. Inspect model readiness through the authenticated API after startup.
# GET /api/models/readiness

# 4. Provision a missing approved artifact (only after URL + digest are configured).
python .\scripts\provision_models.py --model darkir_384
python .\scripts\provision_models.py --model edsr_x4
python .\scripts\provision_models.py --model yunet_2023mar
python .\scripts\provision_models.py --model sface_2021dec
```

## Negative tests that must fail

1. Rename an enabled model temporarily: preflight must fail.
2. Copy the model, alter one byte, point the path to the copy: preflight must fail SHA-256.
3. Remove the EDSR SHA in production with `MODEL_REQUIRE_PINNED_SHA256=true`: preflight must fail.
4. Configure an HTTP model URL: provisioner must reject it.
5. Configure a non-allowlisted HTTPS host: provisioner must reject it.
6. Download content larger than `max_bytes`: provisioner must abort without replacing the valid file.

## Device/performance status

Artifact provisioning and performance are separate gates. EDSR can pass controlled deployment while still being unsuitable for the live hot path. Current OpenCV EDSR wrapper reports the actual loaded device separately from the requested device. DarkIR/EDSR performance should be benchmarked on deployment hardware before enabling them at full camera rate.
