#!/usr/bin/env python3
"""Detect and classify blood cells in smear images with the trained model.

Use this rather than calling model.predict() directly. Ultralytics' validation pads each
image with a gray border and predict() does not, and without the border, cells cut off by
the image edge are detected much worse (see results/border_comparison.json).

Usage: python scripts/predict.py IMAGE_OR_FOLDER [--out annotated/] [--conf 0.47]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "yolo26_blood_cells.pt"
PERFORMANCE = ROOT / "results" / "performance.json"
IMGSZ = 640
BORDER_PX = 16
BORDER_GRAY = 114
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
BOX_BGR = {"WBC": (214, 120, 42), "RBC": (52, 104, 235), "Platelets": (122, 175, 27)}


def detect(model: YOLO, image: np.ndarray, conf: float, device: str | None = None) -> list[dict]:
    """Return the cells in one BGR image, as dicts with class name, confidence and pixel box."""
    h, w = image.shape[:2]
    border = round(BORDER_PX * max(h, w) / IMGSZ)
    gray = (BORDER_GRAY, BORDER_GRAY, BORDER_GRAY)
    padded = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=gray)
    result = model.predict(padded, conf=conf, iou=0.7, imgsz=IMGSZ + 2 * BORDER_PX, device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy() - border
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    cells = []
    for box, score, cls in zip(boxes, result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        if box[2] > box[0] and box[3] > box[1]:  # skip boxes wholly inside the border
            cells.append({"class": result.names[int(cls)], "confidence": round(float(score), 3), "box": [round(float(v), 1) for v in box]})
    return cells


def default_conf() -> float:
    return json.loads(PERFORMANCE.read_text())["operating"]["conf"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="an image, or a folder of images")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, help="confidence threshold; defaults to the one chosen on validation")
    ap.add_argument("--device", help="for example cpu, mps or 0; chosen automatically by default")
    ap.add_argument("--out", type=Path, help="folder for annotated copies of the images")
    args = ap.parse_args()

    conf = args.conf if args.conf is not None else default_conf()
    paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if args.source.is_dir() else [args.source]
    if not paths:
        raise SystemExit(f"No images found in {args.source}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path.name}: could not be read as an image", file=sys.stderr)
            continue
        cells = detect(model, image, conf, args.device)
        counts = Counter(cell["class"] for cell in cells)
        print(f"{path.name}: " + ", ".join(f"{name} {counts[name]}" for name in BOX_BGR))
        if args.out:
            for cell in cells:
                x1, y1, x2, y2 = (int(round(v)) for v in cell["box"])
                cv2.rectangle(image, (x1, y1), (x2, y2), BOX_BGR[cell["class"]], 2)
            cv2.imwrite(str(args.out / f"{path.stem}.jpg"), image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
