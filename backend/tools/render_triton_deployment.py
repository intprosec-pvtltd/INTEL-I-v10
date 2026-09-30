"""Derive a single-GPU deployment from the actual existing Kubernetes base."""
import argparse
import math
from pathlib import Path
import yaml


def render(cameras, capacity, image):
    if cameras < 1 or capacity < 1:
        raise ValueError("cameras and capacity must be positive")
    root = Path(__file__).resolve().parents[1] / "k8s"
    base = yaml.safe_load((root / "kustomization.yaml").read_text(encoding="utf-8-sig"))
    docs = []
    for path in base["resources"]:
        docs.extend(d for d in yaml.safe_load_all((root / path).read_text(encoding="utf-8-sig")) if d)
    removed = {"frs-worker", "anpr-worker"}
    docs = [d for d in docs if not (d["kind"] in {"Deployment", "Service"} and d["metadata"]["name"] in removed)]
    for d in docs:
        name = d["metadata"]["name"]
        if d["kind"] == "ConfigMap" and name == "intel-i-config":
            d["data"].update({
                "INTEL_I_INFERENCE_MODE": "triton", "INTEL_I_TRITON_GRPC_URL": "triton:8001",
                "INTEL_I_MAX_CAMERAS_PER_WORKER": str(capacity), "INFERENCE_FRAME_TTL_MS": "500",
                "INFERENCE_BATCH_FALLBACK_ENABLED": "false", "PRIMARY_DETECTOR_TENSORRT_ENGINE": "",
                "TENSORRT_ENABLED": "false", "VIDEO_HW_DECODE_ENABLED": "false",
                "GPU_PREPROCESSING_ENABLED": "false", "INFERENCE_GPU_REQUIRED": "false",
            })
        if d["kind"] == "Deployment" and name in {"camera-worker", "inference-worker", "intel-i-api"}:
            spec = d["spec"]["template"]["spec"]
            spec.pop("runtimeClassName", None)
            for c in spec["containers"]:
                c["image"] = image
                c.setdefault("env", []).extend([
                    {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
                    {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
                    {"name": "INFERENCE_GPU_REQUIRED", "value": "false"},
                ])
                # Deduplicate env keys, preserving last override.
                c["env"] = list({x["name"]: x for x in c["env"]}.values())
                for section in ("requests", "limits"):
                    c.get("resources", {}).get(section, {}).pop("nvidia.com/gpu", None)
            if name == "camera-worker":
                d["spec"]["replicas"] = math.ceil(cameras / capacity)
        if d["kind"] == "NetworkPolicy":
            for rule in d["spec"].get("ingress", []):
                sources = rule.get("from", [])
                if any(s.get("podSelector", {}).get("matchLabels", {}).get("app") == "inference-worker" for s in sources):
                    sources.append({"podSelector": {"matchLabels": {"app": "triton"}}})
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", type=int, default=60)
    p.add_argument("--capacity", type=int, default=10)
    p.add_argument("--image", default="intel-i-runtime:triton-v1")
    p.add_argument("--output", type=Path, default=Path("k8s/triton/platform.yaml"))
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(yaml.safe_dump_all(render(a.cameras, a.capacity, a.image), sort_keys=False))
    print(f"Wrote {a.output}; {math.ceil(a.cameras / a.capacity)} camera workers; apply with triton/server.yaml")


if __name__ == "__main__":
    main()
