#!/usr/bin/env python3
"""Build the training dataset that gets uploaded to Kaggle, leaving data/ untouched.

Two corrections are applied on top of the TXL-PBC splits as provided:

1. Leakage: a train image showing the same field of view as a val or test image is left
   out of training, and a val image sharing a field with a test image is left out of
   validation. The test split is kept exactly as provided.
2. Labels: exact duplicate rows are removed. Ultralytics would drop these silently at
   load time, so doing it here keeps the reported instance counts honest.

Writes build/kaggle_dataset/ (images, labels, data.yaml, manifest.json) and a copy of the
manifest in reports/dataset/, where the reports read it.

Usage: python scripts/prepare_dataset.py [--data data/TXL-PBC] [--owner KAGGLE_USER]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")
DATASET_SLUG = "txl-pbc-blood-cell-detection"


def exclusions_from_twins(groups: list[list[str]]) -> dict[str, dict]:
    """Decide which images to drop so that no field of view spans two splits."""
    excluded = {}
    for group in groups:
        by_split = {s: [k for k in group if k.startswith(f"{s}/")] for s in SPLITS}
        if len([s for s in SPLITS if by_split[s]]) < 2:
            continue
        # Priority test > val > train: the highest-priority split present keeps its images.
        keeper = "test" if by_split["test"] else "val"
        for split in SPLITS:
            if split != keeper:
                for key in by_split[split]:
                    excluded[key] = {"kept_split": keeper, "kept": by_split[keeper]}
    return excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/TXL-PBC"))
    ap.add_argument("--twins", type=Path, default=Path("reports/dataset/scene_twins.json"))
    ap.add_argument("--out", type=Path, default=Path("build/kaggle_dataset"))
    ap.add_argument("--owner", default="gabrielaschneider")
    args = ap.parse_args()

    twins = json.loads(args.twins.read_text())
    excluded = exclusions_from_twins(twins["groups"])

    if args.out.exists():
        shutil.rmtree(args.out)
    per_split = {}
    deduped_rows = Counter()
    for split in SPLITS:
        img_out = args.out / "images" / split
        lbl_out = args.out / "labels" / split
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        kept = dropped = cells = 0
        per_class = Counter()
        for img in sorted((args.data / "images" / split).glob("*.png")):
            key = f"{split}/{img.stem}"
            if key in excluded:
                dropped += 1
                continue
            shutil.copy2(img, img_out / img.name)
            rows, seen = [], set()
            for line in (args.data / "labels" / split / f"{img.stem}.txt").read_text().splitlines():
                row = line.strip()
                if not row:
                    continue
                if row in seen:
                    deduped_rows[split] += 1
                    continue
                seen.add(row)
                rows.append(row)
                per_class[int(row.split()[0])] += 1
            (lbl_out / f"{img.stem}.txt").write_text("\n".join(rows) + "\n")
            kept += 1
            cells += len(rows)
        per_split[split] = {
            "images": kept,
            "dropped": dropped,
            "cells": cells,
            "per_class": {str(k): per_class[k] for k in sorted(per_class)},
        }

    names = ["WBC", "RBC", "Platelets"]
    (args.out / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnc: 3\nnames: ['WBC', 'RBC', 'Platelets']\n"
    )
    manifest = {
        "source": "TXL-PBC (data/TXL-PBC)",
        "names": names,
        "per_split": per_split,
        "duplicate_label_rows_removed": dict(deduped_rows),
        "excluded_images": excluded,
        "exclusion_rule": "A field of view may only appear in one split; priority test > val > train.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    Path("reports/dataset").mkdir(parents=True, exist_ok=True)
    Path("reports/dataset/manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "TXL-PBC Blood Cell Detection",
                "id": f"{args.owner}/{DATASET_SLUG}",
                "licenses": [{"name": "unknown"}],
            },
            indent=1,
        )
    )

    # Prove the correction worked: no retained field of view may span two splits.
    for group in twins["groups"]:
        present = {k.split("/")[0] for k in group if k not in excluded}
        if len(present) > 1:
            raise SystemExit(f"Leak remains after exclusions: {group}")

    print(json.dumps({k: v for k, v in manifest.items() if k != "excluded_images"}, indent=1))
    print(f"Excluded {len(excluded)} images: {dict(Counter(k.split('/')[0] for k in excluded))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
