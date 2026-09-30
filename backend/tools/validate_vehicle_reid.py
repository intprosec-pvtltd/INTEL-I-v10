#!/usr/bin/env python3
from __future__ import annotations

"""Validate vehicle Re-ID on a labelled local dataset.

Dataset:
    dataset_root/<vehicle_identity>/*.jpg|jpeg|png|webp

For every image, the script extracts an embedding using the same INTEL-I
Re-ID model and reports:
    - same-identity similarity distribution
    - different-identity similarity distribution
    - rank-1 identification accuracy
    - threshold sweep (precision/recall/F1)

This is an evaluation tool, not a claim that the model is production-ready
until the reported metrics meet the deployment acceptance criteria.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vehicleReid import get_vehicle_embedding_result, cosine_similarity  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float32), p)) if values else 0.0


def summary(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "p01": round(percentile(values, 1), 6),
        "p05": round(percentile(values, 5), 6),
        "p50": round(percentile(values, 50), 6),
        "p95": round(percentile(values, 95), 6),
        "p99": round(percentile(values, 99), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reid_validation_report.json"))
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.98)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    identities = {}
    for identity_dir in sorted(args.dataset.iterdir()):
        if not identity_dir.is_dir():
            continue
        images = [p for p in identity_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            identities[identity_dir.name] = images

    if len(identities) < 2:
        raise SystemExit("Need at least two vehicle identity folders.")

    embeddings = {}
    quality = {}
    failures = []
    for identity, images in identities.items():
        embeddings[identity] = []
        for image_path in images:
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append(str(image_path))
                continue
            result = get_vehicle_embedding_result(image)
            if not result.get("valid") or not result.get("embedding"):
                failures.append(str(image_path))
                continue
            embeddings[identity].append(np.asarray(result["embedding"], dtype=np.float32))
            quality[str(image_path)] = float(result.get("crop_quality", 0.0))

    identities = {k: v for k, v in embeddings.items() if v}
    names = sorted(identities)

    same = []
    different = []
    rank1_hits = 0
    rank1_total = 0

    # Image-level nearest identity evaluation.
    for query_identity in names:
        for query_index, query_embedding in enumerate(identities[query_identity]):
            best_identity = None
            best_score = -1.0
            for gallery_identity in names:
                gallery = identities[gallery_identity]
                for gallery_index, gallery_embedding in enumerate(gallery):
                    if gallery_identity == query_identity and gallery_index == query_index:
                        continue
                    score = float(cosine_similarity(query_embedding, gallery_embedding))
                    if gallery_identity == query_identity:
                        same.append(score)
                    else:
                        different.append(score)
                    if score > best_score:
                        best_score = score
                        best_identity = gallery_identity
            if best_identity is not None:
                rank1_total += 1
                rank1_hits += int(best_identity == query_identity)

    thresholds = []
    threshold = args.threshold_min
    while threshold <= args.threshold_max + 1e-9:
        tp = sum(score >= threshold for score in same)
        fn = sum(score < threshold for score in same)
        fp = sum(score >= threshold for score in different)
        tn = sum(score < threshold for score in different)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        thresholds.append({
            "threshold": round(threshold, 4),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "false_positive_rate": round(fp / max(1, fp + tn), 6),
        })
        threshold += args.threshold_step

    best = max(thresholds, key=lambda item: item["f1"]) if thresholds else None
    report = {
        "model": {
            "path": str(__import__("vehicleReid").MODEL_PATH),
            "embedding_dimension": int(__import__("vehicleReid").EMBEDDING_DIM),
        },
        "dataset": {
            "identities": len(names),
            "images_with_embeddings": sum(len(v) for v in identities.values()),
            "failed_images": len(failures),
            "quality_mean": round(statistics.fmean(quality.values()), 6) if quality else 0.0,
        },
        "same_identity_similarity": summary(same),
        "different_identity_similarity": summary(different),
        "rank1_accuracy": round(rank1_hits / max(1, rank1_total), 6),
        "best_threshold_by_f1": best,
        "thresholds": thresholds,
        "failed_images": failures[:200],
        "deployment_note": "Validate day/night/rain/compression/angle/occlusion cohorts separately before setting a production threshold.",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
