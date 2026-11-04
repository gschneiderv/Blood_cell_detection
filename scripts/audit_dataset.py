#!/usr/bin/env python3
"""Audit the TXL-PBC detection dataset before any training.

Verifies what the dataset README claims, then looks for the problems that quietly
invalidate a detection result: broken labels, alpha channels hiding pixels, and
images shared between the train, val and test splits.

Writes reports/dataset/audit.json, which the dataset report and dashboard read.

Usage: python scripts/audit_dataset.py [--data data/TXL-PBC] [--out reports/dataset]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

SPLITS = ("train", "val", "test")
README_COUNTS = {
    "train": {"images": 882, "cells": 12510},
    "val": {"images": 252, "cells": 3752},
    "test": {"images": 126, "cells": 1881},
}
# Hamming distance on a 64-bit perceptual hash. Low values mean visually the same
# image; pairs at this distance are re-checked at pixel level before being counted.
TRAIN_SIZE = 640
PHASH_NEAR_DUP_DISTANCE = 4
PIXEL_MATCH_MAX_MAE = 6.0


def parse_label_file(path: Path, n_classes: int):
    """Return (boxes, problems) for one YOLO label file."""
    boxes, problems = [], []
    text = path.read_text()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append({"kind": "wrong_column_count", "line": lineno, "text": line[:80]})
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            problems.append({"kind": "non_numeric", "line": lineno, "text": line[:80]})
            continue
        if not 0 <= cls < n_classes:
            problems.append({"kind": "class_out_of_range", "line": lineno, "text": line[:80]})
            continue
        if w <= 0 or h <= 0:
            problems.append({"kind": "non_positive_size", "line": lineno, "text": line[:80]})
            continue
        # Allow a tiny tolerance: labelling tools round coordinates at the image edge.
        eps = 1e-3
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        if min(xc, yc, w, h) < 0 or max(xc, yc, w, h) > 1 or x1 < -eps or y1 < -eps or x2 > 1 + eps or y2 > 1 + eps:
            problems.append({"kind": "box_outside_image", "line": lineno, "text": line[:80]})
        boxes.append((cls, xc, yc, w, h))
    return boxes, problems


def pixel_mae(a: Path, b: Path, size: int = 128) -> float:
    """Mean absolute pixel difference after resizing both images to the same size."""
    with Image.open(a) as ia, Image.open(b) as ib:
        pa = np.asarray(ia.convert("RGB").resize((size, size)), dtype=np.float32)
        pb = np.asarray(ib.convert("RGB").resize((size, size)), dtype=np.float32)
    return float(np.abs(pa - pb).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/TXL-PBC"))
    ap.add_argument("--out", type=Path, default=Path("reports/dataset"))
    args = ap.parse_args()

    cfg = yaml.safe_load((args.data / "data.yaml").read_text())
    names = list(cfg["names"])
    records = []
    label_problems = []
    orphan_images, orphan_labels, unreadable = [], [], []

    for split in SPLITS:
        img_dir, lbl_dir = args.data / "images" / split, args.data / "labels" / split
        images = {p.stem: p for p in img_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
        labels = {p.stem: p for p in lbl_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
        orphan_images += [f"{split}/{s}" for s in sorted(images.keys() - labels.keys())]
        orphan_labels += [f"{split}/{s}" for s in sorted(labels.keys() - images.keys())]

        for stem, img_path in sorted(images.items()):
            rec = {"split": split, "stem": stem, "path": str(img_path)}
            try:
                raw = img_path.read_bytes()
                rec["md5"] = hashlib.md5(raw).hexdigest()
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    im.load()
                    rec["width"], rec["height"] = im.size
                    rec["mode"] = im.mode
                    rec["format"] = im.format
                    if im.mode in ("RGBA", "LA", "PA"):
                        alpha = np.asarray(im.getchannel("A"))
                        rec["alpha_min"] = int(alpha.min())
                        rec["alpha_transparent_frac"] = float((alpha < 255).mean())
                    rgb = im.convert("RGB")
                    rec["phash"] = str(imagehash.phash(rgb, hash_size=8))
                    arr = np.asarray(rgb, dtype=np.float32)
                    rec["mean_rgb"] = [round(float(v), 2) for v in arr.mean(axis=(0, 1))]
                    rec["brightness_std"] = round(float(arr.mean(axis=2).std()), 2)
            except Exception as exc:  # corrupt files must be reported, not crash the audit
                unreadable.append({"file": str(img_path), "error": str(exc)})
                continue

            boxes = []
            if stem in labels:
                boxes, problems = parse_label_file(labels[stem], len(names))
                for p in problems:
                    label_problems.append({"split": split, "stem": stem, **p})
                rec["label_md5"] = hashlib.md5(labels[stem].read_bytes()).hexdigest()
            rec["boxes"] = boxes
            records.append(rec)

    # --- counts versus the README ---
    per_split = {}
    for split in SPLITS:
        recs = [r for r in records if r["split"] == split]
        cls = Counter(b[0] for r in recs for b in r["boxes"])
        per_split[split] = {
            "images": len(recs),
            "cells": sum(len(r["boxes"]) for r in recs),
            "per_class": {names[i]: cls.get(i, 0) for i in range(len(names))},
            "images_with_class": {
                names[i]: sum(any(b[0] == i for b in r["boxes"]) for r in recs) for i in range(len(names))
            },
            "empty_label_images": sum(1 for r in recs if not r["boxes"]),
            "readme_images": README_COUNTS[split]["images"],
            "readme_cells": README_COUNTS[split]["cells"],
        }

    # --- exact duplicates (same file bytes) ---
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]
    exact_cross = [g for g in exact_groups if len({r["split"] for r in g}) > 1]

    # --- near duplicates: pHash pre-filter, confirmed at pixel level ---
    hashes = [imagehash.hex_to_hash(r["phash"]) for r in records]
    bits = np.array([np.asarray(h.hash, dtype=bool).ravel() for h in hashes])
    near_pairs = []
    for i in range(len(records)):
        dists = (bits[i + 1 :] != bits[i]).sum(axis=1)
        for j_off in np.nonzero(dists <= PHASH_NEAR_DUP_DISTANCE)[0]:
            j = i + 1 + int(j_off)
            a, b = records[i], records[j]
            if a["md5"] == b["md5"]:
                continue
            mae = pixel_mae(Path(a["path"]), Path(b["path"]))
            if mae <= PIXEL_MATCH_MAX_MAE:
                near_pairs.append(
                    {
                        "a": f"{a['split']}/{a['stem']}",
                        "b": f"{b['split']}/{b['stem']}",
                        "split_a": a["split"],
                        "split_b": b["split"],
                        "phash_distance": int(dists[j_off]),
                        "pixel_mae": round(mae, 2),
                    }
                )
    near_cross = [p for p in near_pairs if p["split_a"] != p["split_b"]]
    cross_counter = Counter(tuple(sorted((p["split_a"], p["split_b"]))) for p in near_cross)

    # --- connected groups, so splits can be compared by unique scene ---
    parent = {f"{r['split']}/{r['stem']}": f"{r['split']}/{r['stem']}" for r in records}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in exact_groups:
        keys = [f"{r['split']}/{r['stem']}" for r in g]
        for k in keys[1:]:
            parent[find(k)] = find(keys[0])
    for p in near_pairs:
        parent[find(p["a"])] = find(p["b"])
    groups = defaultdict(list)
    for k in parent:
        groups[find(k)].append(k)
    dup_groups = [sorted(v) for v in groups.values() if len(v) > 1]
    leaky_groups = [g for g in dup_groups if len({k.split("/")[0] for k in g}) > 1]
    affected = Counter(k.split("/")[0] for g in leaky_groups for k in g)

    # --- geometry / visual statistics ---
    sizes = Counter((r["width"], r["height"]) for r in records)
    modes = Counter(r["mode"] for r in records)
    alpha_recs = [r for r in records if "alpha_min" in r]
    alpha_non_opaque = [
        {"split": r["split"], "stem": r["stem"], "frac": round(r["alpha_transparent_frac"], 4)}
        for r in alpha_recs
        if r["alpha_transparent_frac"] > 0
    ]

    # Ultralytics scales an image so its long side fits the input size, keeping the aspect
    # ratio. Box sizes in that frame are what decide how hard a cell is to detect.
    box_rows = []
    for r in records:
        scale = TRAIN_SIZE / max(r["width"], r["height"])
        for cls, _, _, w, h in r["boxes"]:
            w_px, h_px = w * r["width"], h * r["height"]
            box_rows.append(
                {
                    "cls": names[cls],
                    "side_px": float(np.sqrt(w_px * h_px)),
                    "side_train": float(np.sqrt(w_px * h_px)) * scale,
                }
            )
    size_stats = {}
    size_hist = {}
    bin_edges = np.arange(0, 420, 10)
    for name in names:
        rows = [b for b in box_rows if b["cls"] == name]
        native = np.array([b["side_px"] for b in rows])
        train = np.array([b["side_train"] for b in rows])
        size_stats[name] = {
            "n": len(rows),
            "side_px_median": round(float(np.median(native)), 1),
            "side_at_640_p05": round(float(np.percentile(train, 5)), 1),
            "side_at_640_median": round(float(np.median(train)), 1),
            "side_at_640_p95": round(float(np.percentile(train, 95)), 1),
            "frac_under_32px_at_640": round(float((train < 32).mean()), 4),
            "frac_under_16px_at_640": round(float((train < 16).mean()), 4),
        }
        counts, _ = np.histogram(train, bins=bin_edges)
        size_hist[name] = [round(float(v), 5) for v in counts / max(1, len(train))]

    cells_per_image = [len(r["boxes"]) for r in records]

    # exact duplicate boxes inside a file are a labelling artefact worth surfacing
    dup_box_files = 0
    dup_box_total = 0
    for r in records:
        c = Counter(tuple(round(v, 4) for v in b) for b in r["boxes"])
        extra = sum(v - 1 for v in c.values() if v > 1)
        if extra:
            dup_box_files += 1
            dup_box_total += extra

    audit = {
        "dataset": str(args.data),
        "names": names,
        "yaml": cfg,
        "totals": {
            "images": len(records),
            "cells": sum(cells_per_image),
            "per_class": {
                n: sum(s["per_class"][n] for s in per_split.values()) for n in names
            },
        },
        "per_split": per_split,
        "integrity": {
            "unreadable_images": unreadable,
            "images_without_label": orphan_images,
            "labels_without_image": orphan_labels,
            "label_problems_count": len(label_problems),
            "label_problems_by_kind": dict(Counter(p["kind"] for p in label_problems)),
            "label_problems_sample": label_problems[:50],
            "duplicate_box_files": dup_box_files,
            "duplicate_boxes_total": dup_box_total,
        },
        "image_stats": {
            "modes": dict(modes),
            "distinct_sizes": len(sizes),
            "top_sizes": [{"width": w, "height": h, "count": c} for (w, h), c in sizes.most_common(12)],
            "alpha_images": len(alpha_recs),
            "alpha_non_opaque": alpha_non_opaque[:50],
            "alpha_non_opaque_count": len(alpha_non_opaque),
            "cells_per_image": {
                "min": int(np.min(cells_per_image)),
                "median": float(np.median(cells_per_image)),
                "mean": round(float(np.mean(cells_per_image)), 2),
                "p95": float(np.percentile(cells_per_image, 95)),
                "max": int(np.max(cells_per_image)),
            },
        },
        "box_size_stats": size_stats,
        "box_size_hist": {"bin_centres": [float(v) for v in (bin_edges[:-1] + 5)], "series": size_hist},
        "leakage": {
            "phash_distance_threshold": PHASH_NEAR_DUP_DISTANCE,
            "pixel_mae_threshold": PIXEL_MATCH_MAX_MAE,
            "exact_duplicate_groups": len(exact_groups),
            "exact_duplicate_groups_across_splits": len(exact_cross),
            "exact_cross_examples": [
                [f"{r['split']}/{r['stem']}" for r in g] for g in exact_cross[:20]
            ],
            "near_duplicate_pairs": len(near_pairs),
            "near_duplicate_pairs_across_splits": len(near_cross),
            "near_cross_by_split_pair": {"-".join(k): v for k, v in cross_counter.items()},
            "near_cross_examples": near_cross[:40],
            "duplicate_groups": len(dup_groups),
            "groups_spanning_splits": len(leaky_groups),
            "images_in_spanning_groups_by_split": dict(affected),
            "leaky_group_members": leaky_groups,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "audit.json"
    out.write_text(json.dumps(audit, indent=2))
    # per-image table used by the report and by the leakage-free evaluation
    slim = [
        {k: r[k] for k in ("split", "stem", "width", "height", "mode", "mean_rgb", "phash") if k in r}
        | {"n_boxes": len(r["boxes"]), "boxes": [[b[0], *[round(v, 5) for v in b[1:]]] for b in r["boxes"]]}
        for r in records
    ]
    (args.out / "images.json").write_text(json.dumps(slim))

    # readable console summary
    print(f"Images {audit['totals']['images']}  Cells {audit['totals']['cells']}  {audit['totals']['per_class']}")
    for s, v in per_split.items():
        flag = "OK" if v["images"] == v["readme_images"] and v["cells"] == v["readme_cells"] else "MISMATCH"
        print(f"  {s:5s} images {v['images']:4d}  cells {v['cells']:5d}  {v['per_class']}  empty={v['empty_label_images']}  README:{flag}")
    print("Integrity:", {k: (len(v) if isinstance(v, list) else v) for k, v in audit["integrity"].items() if k != "label_problems_sample"})
    print("Modes:", dict(modes), " alpha non-opaque images:", len(alpha_non_opaque))
    print("Sizes:", len(sizes), "distinct; top:", sizes.most_common(6))
    print("Cells/image:", audit["image_stats"]["cells_per_image"])
    print("Box sizes:", json.dumps(size_stats, indent=1))
    lk = audit["leakage"]
    print(
        "Leakage: exact groups", lk["exact_duplicate_groups"], "(across splits", lk["exact_duplicate_groups_across_splits"], ")",
        "| near pairs", lk["near_duplicate_pairs"], "(across splits", lk["near_duplicate_pairs_across_splits"], ")",
        "| groups spanning splits", lk["groups_spanning_splits"], "images", lk["images_in_spanning_groups_by_split"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
