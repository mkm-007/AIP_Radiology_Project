#!/usr/bin/env python3
"""Offline region-guided stub — no MIMIC / no PHI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REGIONS = ["heart", "mediastinum", "left_lung", "right_lung"]
CONFIG = ROOT / "configs" / "region_config_v1.json"


def load_regions():
    if CONFIG.exists():
        data = json.loads(CONFIG.read_text())
        names = data.get("canonical_regions") or data.get("regions") or REGIONS
        return list(names)[:4] or REGIONS
    return REGIONS


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def bleu1(hyp: list[str], ref: list[str]) -> float:
    if not hyp:
        return 0.0
    ref_counts = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    hit = 0
    for t in hyp:
        if ref_counts.get(t, 0) > 0:
            hit += 1
            ref_counts[t] -= 1
    return hit / len(hyp)


def main():
    rng = np.random.default_rng(0)
    regions = load_regions()
    # fake region embeddings + keyword bags
    gallery = []
    for i, r in enumerate(regions):
        vec = rng.normal(size=16)
        vec = vec / np.linalg.norm(vec)
        tokens = {r.replace("_", ""), "unremarkable", "normal"}
        gallery.append((r, vec, tokens))

    print("Region-guided offline demo")
    print(f"Canonical regions: {', '.join(regions)}")
    print("Report metrics live in README (enrolled CV/test). Demo shows the loop only.\n")

    for r, q, true_tokens in gallery:
        # retrieval: nearest region embedding in gallery (leave-one-in toy)
        scores = [float(q @ g[1]) for g in gallery]
        j = int(np.argmax(scores))
        retrieved = gallery[j][2]
        gen_tokens = list(true_tokens)  # stub "generation" copies region prior
        print(
            f"{r:12} retrieval_jaccard={jaccard(true_tokens, retrieved):.3f} "
            f"stub_bleu1={bleu1(gen_tokens, list(true_tokens)):.3f}"
        )

    print(
        "\nEnrolled reference — gen test BLEU-1 0.306 / ROUGE-L 0.381; "
        "retrieval test Jaccard 0.114."
    )


if __name__ == "__main__":
    main()
