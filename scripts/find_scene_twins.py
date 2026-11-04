#!/usr/bin/env python3
"""Find images that show the same field of view, even if shifted, re-framed or flipped.

Pixel differences and perceptual hashes miss re-framed copies of a field, which is what
leaks between train and test in cell datasets. This compares every image pair with SIFT
keypoints and requires the matches to agree on one geometric transform (RANSAC).

Calibration, from reviewing each score band by eye (see reports/dataset_report.html):
- Among same-source, non-mirrored pairs the scores split into two populations with a gap
  between them: look-alike nuclei stop at 10 inliers, and genuine twins start at 12. Twins
  may differ in zoom (up to about 9%) and colour cast, so scale is not a useful filter.
- Matches across sources, or between mirrored images, are noise. They also stop at 10 or
  11 inliers, and no pair checked by eye was a genuine twin.
- Loose rules are dangerous: at 8 inliers, false matches chain unrelated images into a
  single group of 149, which would discard good training data for no reason.

Writes reports/dataset/scene_twins.json, consumed by prepare_dataset.py and the reports.
The slow all-pairs scores are cached in build/, so the rule can be revised cheaply.

Usage: python scripts/find_scene_twins.py [--data data/TXL-PBC] [--out reports/dataset]
                                          [--reuse-scores]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
MAX_SIDE = 512
N_FEATURES = 800
RATIO = 0.75
MIN_MATCHES = 6
# Decision rule for "same field of view", calibrated as described above.
MIN_INLIERS = 12
# Pairs outside this scale are degenerate (matches collapsing onto one keypoint).
SCALE_RANGE = (0.5, 2.0)
SCORE_CACHE = Path("build/scene_pair_scores.json")
# 360x363 and 366x369 images share stain and optics, so they count as one source family.
SOURCE_FAMILY = {(360, 363): "360x363", (366, 369): "360x363"}


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    scale = MAX_SIDE / max(img.shape)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def root_sift(desc: np.ndarray) -> np.ndarray:
    desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc).astype(np.float32)


def extract(path: Path):
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    img = load_gray(path)
    out = {}
    for name, view in (("orig", img), ("mirror", cv2.flip(img, 1))):
        kps, desc = sift.detectAndCompute(view, None)
        if desc is None or len(kps) < MIN_MATCHES:
            out[name] = (np.empty((0, 2), np.float32), np.empty((0, 128), np.float32))
        else:
            out[name] = (np.float32([k.pt for k in kps]), root_sift(desc))
    return out


def verify(a, b, matcher):
    """Return (inliers, inlier_fraction, scale) for the best similarity transform a -> b."""
    pa, da = a
    pb, db = b
    if len(da) < MIN_MATCHES or len(db) < MIN_MATCHES:
        return 0, 0.0, 0.0
    good = [
        pair[0]
        for pair in matcher.knnMatch(da, db, k=2)
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance
    ]
    if len(good) < MIN_MATCHES:
        return 0, 0.0, 0.0
    src = pa[[m.queryIdx for m in good]]
    dst = pb[[m.trainIdx for m in good]]
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000)
    if M is None or mask is None:
        return 0, 0.0, 0.0
    inliers = int(mask.sum())
    return inliers, inliers / len(good), float(np.hypot(M[0, 0], M[1, 0]))


_FEATS = None


def _init(feats):
    global _FEATS
    _FEATS = feats
    cv2.setNumThreads(1)


def _compare_row(i):
    """Compare image i (both orientations) with every later image; keep scored pairs."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for j in range(i + 1, len(_FEATS)):
        best = (0, 0.0, 0.0, "orig")
        for view in ("orig", "mirror"):
            inl, frac, scale = verify(_FEATS[i][view], _FEATS[j]["orig"], matcher)
            if inl > best[0]:
                best = (inl, frac, scale, view)
        if best[0] >= MIN_MATCHES:
            rows.append((i, j, *best))
    return rows


def score_all_pairs(items, workers):
    print(f"Extracting SIFT from {len(items)} images...", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        feats = list(ex.map(extract, [it["path"] for it in items], chunksize=16))
    n_pairs = len(items) * (len(items) - 1) // 2
    print(f"Comparing all {n_pairs:,} image pairs...", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(feats,)) as ex:
        results = [row for rows in ex.map(_compare_row, range(len(items)), chunksize=8) for row in rows]
    scored = [
        {
            "a": items[i]["key"],
            "b": items[j]["key"],
            "inliers": inl,
            "inlier_fraction": round(frac, 3),
            "scale": round(scale, 3),
            "mirrored": view == "mirror",
        }
        for i, j, inl, frac, scale, view in results
        if SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]
    ]
    SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_CACHE.write_text(json.dumps({"pairs_compared": n_pairs, "scored": scored}))
    return n_pairs, scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/TXL-PBC"))
    ap.add_argument("--out", type=Path, default=Path("reports/dataset"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse-scores", action="store_true", help=f"reuse {SCORE_CACHE} instead of recomputing")
    args = ap.parse_args()

    items = []
    for split in SPLITS:
        for p in sorted((args.data / "images" / split).glob("*.png")):
            with Image.open(p) as im:
                size = im.size
            family = SOURCE_FAMILY.get(size, f"{size[0]}x{size[1]}")
            items.append({"key": f"{split}/{p.stem}", "split": split, "path": p, "family": family})
    family_of = {it["key"]: it["family"] for it in items}

    if args.reuse_scores and SCORE_CACHE.exists():
        cache = json.loads(SCORE_CACHE.read_text())
        n_pairs, scored = cache["pairs_compared"], cache["scored"]
        print(f"Reusing {len(scored):,} scored pairs from {SCORE_CACHE}", flush=True)
    else:
        n_pairs, scored = score_all_pairs(items, args.workers)

    pairs, rejected = [], []
    calibration = defaultdict(Counter)
    for row in scored:
        same_source = family_of[row["a"]] == family_of[row["b"]]
        if not same_source:
            population = "different_source"
        elif row["mirrored"]:
            population = "same_source_mirrored"
        else:
            population = "same_source_direct"
        calibration[population][min(row["inliers"], 40)] += 1
        row = {**row, "source": family_of[row["a"]] if same_source else None}
        if population == "same_source_direct" and row["inliers"] >= MIN_INLIERS:
            pairs.append(row)
        elif row["inliers"] >= MIN_MATCHES + 2:
            rejected.append(row)
    pairs.sort(key=lambda p: -p["inliers"])
    rejected.sort(key=lambda p: -p["inliers"])

    parent = {it["key"]: it["key"] for it in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        parent[find(p["a"])] = find(p["b"])
    groups = defaultdict(list)
    for it in items:
        groups[find(it["key"])].append(it["key"])
    twin_groups = [sorted(v) for v in groups.values() if len(v) > 1]
    spanning = [g for g in twin_groups if len({k.split("/")[0] for k in g}) > 1]

    def split_of(k):
        return k.split("/")[0]

    cross_pairs = [p for p in pairs if split_of(p["a"]) != split_of(p["b"])]
    report = {
        "method": "RootSIFT + exhaustive pairwise ratio-test matching + RANSAC similarity (mirror-aware)",
        "rule": f"same source, not mirrored, at least {MIN_INLIERS} RANSAC inliers",
        "params": {
            "min_inliers": MIN_INLIERS,
            "scale_range": SCALE_RANGE,
            "ratio": RATIO,
            "max_side": MAX_SIDE,
        },
        "pairs_compared": n_pairs,
        "calibration_inliers_by_population": {
            pop: {str(k): v for k, v in sorted(hist.items())} for pop, hist in sorted(calibration.items())
        },
        "pairs_total": len(pairs),
        "pairs_by_source": dict(Counter(p["source"] for p in pairs)),
        "pairs_across_splits": len(cross_pairs),
        "pairs_across_splits_by_split": dict(
            Counter("-".join(sorted((split_of(p["a"]), split_of(p["b"])))) for p in cross_pairs)
        ),
        "groups_total": len(twin_groups),
        "groups_spanning_splits": len(spanning),
        "images_in_any_group": sum(len(g) for g in twin_groups),
        "images_in_spanning_groups_by_split": dict(Counter(split_of(k) for g in spanning for k in g)),
        "largest_group": max((len(g) for g in twin_groups), default=0),
        "pairs": pairs,
        "groups": twin_groups,
        "spanning_groups": spanning,
        "rejected_examples": rejected[:200],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scene_twins.json").write_text(json.dumps(report, indent=1))
    keep = {k: v for k, v in report.items() if k not in ("pairs", "groups", "spanning_groups", "rejected_examples")}
    print(json.dumps(keep, indent=1))
    print(f"Rejected at plausible scale: {len(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
