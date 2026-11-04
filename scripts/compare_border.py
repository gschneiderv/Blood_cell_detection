#!/usr/bin/env python3
"""Measure what the gray inference border is worth, with and without it, on val and test.

The decision to use the border rests on validation. Test is included only to confirm it.

Writes results/border_comparison.json, which the performance report reads.

Usage: python scripts/compare_border.py [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E  # noqa: E402
import predict as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "build" / "kaggle_dataset"


def predictions(model: YOLO, split: str, border: bool, device: str) -> list[dict]:
    rows = []
    for path in sorted((DATASET / "images" / split).glob("*.png")):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        if border:
            cells = P.detect(model, image, 0.001, device)
            boxes = [c["box"] for c in cells]
            conf = [c["confidence"] for c in cells]
            cls = [E.NAMES.index(c["class"]) for c in cells]
        else:
            r = model.predict(image, conf=0.001, iou=0.7, imgsz=P.IMGSZ, device=device, verbose=False)[0]
            boxes, conf, cls = r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), [int(c) for c in r.boxes.cls.tolist()]
        rows.append({"stem": path.stem, "shape": [h, w], "boxes": boxes, "conf": conf, "cls": cls})
    return rows


def summarise(images: list[dict], threshold: float) -> dict:
    s = E.summarise_ap(E.compute_ap(images)["ap"])
    counts = E.counts_at_threshold(images, threshold)
    return {
        "map50": round(s["map50"], 4),
        "map50_95": round(s["map50_95"], 4),
        "threshold": threshold,
        "macro_f1": round(float(counts["f1"].mean()), 4),
        "rbc_fp": int(counts["fp"][1]),
        "rbc_fn": int(counts["fn"][1]),
        "platelet_fp": int(counts["fp"][2]),
        "platelet_fn": int(counts["fn"][2]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = YOLO(str(P.DEFAULT_WEIGHTS))
    out: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        loaded = {}
        for border in (False, True):
            for split in ("val", "test"):
                (folder / f"predictions_{split}.json").write_text(json.dumps({"predictions": predictions(model, split, border, args.device)}))
                loaded[(split, border)] = E.load_split(folder, DATASET / "labels", split)[0]
            # Each input style gets its own threshold, always chosen on validation.
            threshold, _ = E.choose_threshold(loaded[("val", border)])
            key = "with_border" if border else "without_border"
            out[key] = {split: summarise(loaded[(split, border)], threshold) for split in ("val", "test")}
    target = ROOT / "results" / "border_comparison.json"
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
