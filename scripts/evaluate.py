#!/usr/bin/env python3
"""Compute the final model metrics from the raw predictions saved by the Kaggle kernel.

Runs locally without a GPU. Validation predictions choose the confidence threshold. The
test predictions then measure performance, with bootstrap confidence intervals, plus the
breakdowns Ultralytics does not report: per image source, and counting accuracy.

As a guard against analysis bugs, the mAP computed here is compared with what Ultralytics
reported on the same model, and the script stops if they disagree.

Writes results/performance.json and results/gallery_items.json.

Usage: python scripts/evaluate.py [--artifacts results/raw/<kernel>/artifacts]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class

NAMES = ["WBC", "RBC", "Platelets"]
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
BOOTSTRAP_SAMPLES = 2000
# Predictions use the same gray border as Ultralytics' validation, so the two should agree
# closely. A larger gap means a bug in this analysis, not a difference in input.
MAX_AGREEMENT_GAP = 0.003
MIN_CLASS_INSTANCES = 5
SOURCE_LABEL = {
    (575, 575): "575×575",
    (360, 363): "360×363",
    (366, 369): "360×363",
    (640, 480): "640×480",
    (256, 256): "256×256",
}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between Nx4 and Mx4 xyxy arrays."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_all_thresholds(pred_cls, gt_cls, iou) -> np.ndarray:
    """Same greedy matching as Ultralytics: returns a detections x thresholds bool array.

    Detections must already be sorted by descending confidence.
    """
    correct = np.zeros((len(pred_cls), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(pred_cls) or not len(gt_cls):
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls)  # rows are ground truth, columns detections
    matched = np.zeros((len(gt_cls), len(IOU_THRESHOLDS)), dtype=bool)
    for j in np.flatnonzero((iou >= IOU_THRESHOLDS.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, range(len(IOU_THRESHOLDS))] >= IOU_THRESHOLDS
        matched[k, range(len(IOU_THRESHOLDS))] |= correct[j]
    return correct


def load_split(artifacts: Path, labels_dir: Path, split: str) -> tuple[list[dict], dict]:
    payload = json.loads((artifacts / f"predictions_{split}.json").read_text())
    images = []
    for row in payload["predictions"]:
        h, w = row["shape"]
        order = np.argsort(-np.array(row["conf"], dtype=float), kind="stable")
        conf = np.array(row["conf"], dtype=float)[order]
        cls = np.array(row["cls"], dtype=int)[order]
        boxes = np.array(row["boxes"], dtype=float).reshape(-1, 4)[order]
        lines = (labels_dir / split / f"{row['stem']}.txt").read_text().split("\n")
        gt = np.array([[float(v) for v in ln.split()] for ln in lines if ln.strip()]).reshape(-1, 5)
        gt_cls = gt[:, 0].astype(int)
        gt_boxes = np.stack(
            [
                (gt[:, 1] - gt[:, 3] / 2) * w,
                (gt[:, 2] - gt[:, 4] / 2) * h,
                (gt[:, 1] + gt[:, 3] / 2) * w,
                (gt[:, 2] + gt[:, 4] / 2) * h,
            ],
            axis=1,
        )
        iou = box_iou(gt_boxes, boxes)
        images.append(
            {
                "stem": row["stem"],
                "split": split,
                "w": w,
                "h": h,
                "source": SOURCE_LABEL.get((w, h), f"{w}×{h}"),
                "conf": conf,
                "cls": cls,
                "boxes": boxes,
                "gt_cls": gt_cls,
                "gt_boxes": gt_boxes,
                "iou": iou,
                "correct": match_all_thresholds(cls, gt_cls, iou),
            }
        )
    return images, payload


def compute_ap(images: list[dict]) -> dict:
    """Per-class AP at each IoU threshold, using the Ultralytics implementation."""
    tp = np.concatenate([im["correct"] for im in images])
    conf = np.concatenate([im["conf"] for im in images])
    pred_cls = np.concatenate([im["cls"] for im in images])
    target_cls = np.concatenate([im["gt_cls"] for im in images])
    out = {"ap": np.full((len(NAMES), len(IOU_THRESHOLDS)), np.nan)}
    if not len(target_cls):
        return out
    result = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names=dict(enumerate(NAMES)))
    ap, classes = result[5], result[6]
    for row, c in enumerate(classes.astype(int)):
        out["ap"][c] = ap[row]
    out["pr_x"] = result[10]
    out["pr_curves"] = {int(c): result[11][i, :] if result[11].ndim == 2 else None for i, c in enumerate(classes.astype(int))}
    return out


def summarise_ap(ap: np.ndarray, classes: list[int] | None = None) -> dict:
    """Per-class and mean AP. A class with no labels has NaN AP and is left out of the mean."""
    ap50 = ap[:, 0]
    ap50_95 = ap.mean(axis=1)  # NaN stays NaN for a class with no labels
    keep = list(range(len(NAMES))) if classes is None else classes
    return {
        "ap50": ap50,
        "ap50_95": ap50_95,
        "map50": float(np.nanmean(ap50[keep])),
        "map50_95": float(np.nanmean(ap50_95[keep])),
    }


def detections_at(im: dict, threshold: float) -> dict:
    keep = im["conf"] >= threshold
    return {"cls": im["cls"][keep], "conf": im["conf"][keep], "iou": im["iou"][:, keep], "index": np.flatnonzero(keep)}


def match_at_threshold(im: dict, threshold: float, iou_cut: float = 0.5) -> dict:
    """One-to-one matching at a fixed confidence.

    Same-class matches are made first, highest confidence first. Whatever remains is then
    paired across classes, which separates real class confusion from missed or spurious
    cells. The confusion matrix diagonal therefore equals the per-class true positives.
    """
    det = detections_at(im, threshold)
    n_gt, n_det = len(im["gt_cls"]), len(det["cls"])
    gt_used = np.zeros(n_gt, dtype=bool)
    det_used = np.zeros(n_det, dtype=bool)
    det_match = np.full(n_det, -1)
    for j in range(n_det):
        best, best_iou = -1, iou_cut
        for i in range(n_gt):
            if not gt_used[i] and im["gt_cls"][i] == det["cls"][j] and det["iou"][i, j] >= best_iou:
                best, best_iou = i, det["iou"][i, j]
        if best >= 0:
            gt_used[best] = det_used[j] = True
            det_match[j] = best
    confused = []
    for j in range(n_det):
        if det_used[j]:
            continue
        best, best_iou = -1, iou_cut
        for i in range(n_gt):
            if not gt_used[i] and det["iou"][i, j] >= best_iou:
                best, best_iou = i, det["iou"][i, j]
        if best >= 0:
            gt_used[best] = det_used[j] = True
            confused.append((best, j))
    return {"det": det, "gt_used": gt_used, "det_used": det_used, "det_match": det_match, "confused": confused}


def counts_at_threshold(images: list[dict], threshold: float, iou_cut: float = 0.5) -> dict:
    tp, fp, fn = np.zeros(3, int), np.zeros(3, int), np.zeros(3, int)
    matrix = np.zeros((4, 4), dtype=int)  # rows true, columns predicted; index 3 = background
    for im in images:
        m = match_at_threshold(im, threshold, iou_cut)
        for j, i in enumerate(m["det_match"]):
            c = m["det"]["cls"][j]
            if i >= 0:
                tp[c] += 1
                matrix[c, c] += 1
            else:
                fp[c] += 1
        for i, j in m["confused"]:
            matrix[im["gt_cls"][i], m["det"]["cls"][j]] += 1
        confused_gt = {i for i, _ in m["confused"]}
        confused_det = {j for _, j in m["confused"]}
        for i, c in enumerate(im["gt_cls"]):
            if not (m["gt_used"][i] and i not in confused_gt):
                fn[c] += 1
            if not m["gt_used"][i]:
                matrix[c, 3] += 1
        for j, c in enumerate(m["det"]["cls"]):
            if not m["det_used"][j] and j not in confused_det:
                matrix[3, c] += 1
    # Every label and every detection must be counted exactly once.
    n_true = np.array([sum((im["gt_cls"] == c).sum() for im in images) for c in range(3)])
    n_det = np.array([sum(((im["conf"] >= threshold) & (im["cls"] == c)).sum() for im in images) for c in range(3)])
    assert (matrix[:3].sum(axis=1) == n_true).all() and (tp + fn == n_true).all(), "labels miscounted"
    assert (matrix[:, :3].sum(axis=0) == n_det).all() and (tp + fp == n_det).all(), "detections miscounted"
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "matrix": matrix}


def choose_threshold(val_images: list[dict]) -> tuple[float, list[list[float]]]:
    """Confidence that maximises macro F1 on validation."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = [[float(t), float(counts_at_threshold(val_images, t)["f1"].mean())] for t in grid]
    best = max(curve, key=lambda row: row[1])
    return best[0], curve


def bootstrap(images: list[dict], samples: int, seed: int = 42) -> dict:
    """Image-level bootstrap. Cells in one image are correlated, so images are the unit."""
    rng = np.random.default_rng(seed)
    n = len(images)
    collected = {"map50": [], "map50_95": [], "ap50": [], "ap50_95": []}
    for _ in range(samples):
        sample = [images[i] for i in rng.integers(0, n, n)]
        s = summarise_ap(compute_ap(sample)["ap"])
        collected["map50"].append(s["map50"])
        collected["map50_95"].append(s["map50_95"])
        collected["ap50"].append(s["ap50"])
        collected["ap50_95"].append(s["ap50_95"])

    def interval(values):
        return [round(float(v), 4) for v in np.nanpercentile(values, [2.5, 97.5], axis=0)]

    ap50 = np.nanpercentile(np.array(collected["ap50"]), [2.5, 97.5], axis=0)
    ap50_95 = np.nanpercentile(np.array(collected["ap50_95"]), [2.5, 97.5], axis=0)
    return {
        "map50": interval(collected["map50"]),
        "map50_95": interval(collected["map50_95"]),
        "ap50": ap50.round(4).T.tolist(),
        "ap50_95": ap50_95.round(4).T.tolist(),
    }


def touches_edge(box: np.ndarray, w: int, h: int, margin: float = 0.01) -> bool:
    """True if a box reaches the image border, meaning its cell is probably cut off."""
    return bool(box[0] <= margin * w or box[1] <= margin * h or box[2] >= (1 - margin) * w or box[3] >= (1 - margin) * h)


def error_breakdown(images: list[dict], threshold: float) -> dict:
    """Separate plausible model errors from cases where the labelling policy is in question."""
    out = {name: {"fp": 0, "fp_near_label": 0, "fp_at_edge": 0, "fp_elsewhere": 0, "fn": 0, "fn_at_edge": 0, "fn_elsewhere": 0} for name in NAMES}
    for im in images:
        m = match_at_threshold(im, threshold)
        det = m["det"]
        confused_det = {j for _, j in m["confused"]}
        confused_gt = {i for i, _ in m["confused"]}
        for j, c in enumerate(det["cls"]):
            if m["det_match"][j] >= 0:
                continue
            row = out[NAMES[c]]
            row["fp"] += 1
            same_class = det["iou"][im["gt_cls"] == c, j]
            if j in confused_det or (same_class.size and same_class.max() >= 0.1):
                row["fp_near_label"] += 1
            elif touches_edge(im["boxes"][det["index"][j]], im["w"], im["h"]):
                row["fp_at_edge"] += 1
            else:
                row["fp_elsewhere"] += 1
        for i, c in enumerate(im["gt_cls"]):
            if m["gt_used"][i] and i not in confused_gt:
                continue
            row = out[NAMES[c]]
            row["fn"] += 1
            if touches_edge(im["gt_boxes"][i], im["w"], im["h"]):
                row["fn_at_edge"] += 1
            else:
                row["fn_elsewhere"] += 1
    return out


def count_accuracy(images: list[dict], threshold: float) -> dict:
    out = {}
    for c, name in enumerate(NAMES):
        true = np.array([(im["gt_cls"] == c).sum() for im in images])
        pred = np.array([((im["conf"] >= threshold) & (im["cls"] == c)).sum() for im in images])
        out[name] = {
            "mean_true": round(float(true.mean()), 3),
            "mean_pred": round(float(pred.mean()), 3),
            "mae": round(float(np.abs(pred - true).mean()), 3),
            "bias": round(float((pred - true).mean()), 3),
            "exact": round(float((pred == true).mean()), 3),
        }
    return out


def f1_vs_conf(images: list[dict]) -> dict:
    grid = np.round(np.arange(0.02, 0.99, 0.02), 2)
    rows = [counts_at_threshold(images, t)["f1"] for t in grid]
    return {NAMES[c]: [[float(t), round(float(r[c]), 4)] for t, r in zip(grid, rows)] for c in range(3)}, [float(t) for t in grid]


def pr_curve(ap_result: dict) -> tuple[dict, list[float]]:
    x = ap_result["pr_x"]
    keep = np.linspace(0, len(x) - 1, 101).astype(int)
    recall = [round(float(v), 3) for v in x[keep]]
    curves = {}
    for c, name in enumerate(NAMES):
        y = ap_result["pr_curves"].get(c)
        curves[name] = [[recall[i], round(float(y[k]), 4)] for i, k in enumerate(keep)] if y is not None else []
    return curves, recall


def per_source(images: list[dict], threshold: float) -> list[dict]:
    rows = []
    for source in sorted({im["source"] for im in images}, key=lambda s: -sum(i["source"] == s for i in images)):
        subset = [im for im in images if im["source"] == source]
        instances = {c: int(sum((im["gt_cls"] == c).sum() for im in subset)) for c in range(3)}
        # A class with a handful of labels here has an AP dominated by chance, and would
        # swing the mean far more than it deserves, so means only use well-populated classes.
        present = [c for c in range(3) if instances[c] >= MIN_CLASS_INSTANCES]
        s = summarise_ap(compute_ap(subset)["ap"], present)
        counts = counts_at_threshold(subset, threshold)
        rows.append(
            {
                "label": source,
                "images": len(subset),
                "classes_in_mean": [NAMES[c] for c in present],
                "instances": {NAMES[c]: instances[c] for c in range(3)},
                "map50": round(s["map50"], 4),
                "map50_95": round(s["map50_95"], 4),
                "f1_macro": round(float(np.mean([counts["f1"][c] for c in present])), 4),
                "per_class": {
                    NAMES[c]: {
                        "ap50": None if np.isnan(s["ap50"][c]) else round(float(s["ap50"][c]), 4),
                        "ap50_95": None if np.isnan(s["ap50_95"][c]) else round(float(s["ap50_95"][c]), 4),
                        "precision": round(float(counts["precision"][c]), 4),
                        "recall": round(float(counts["recall"][c]), 4),
                        "fp": int(counts["fp"][c]),
                        "fn": int(counts["fn"][c]),
                    }
                    for c in range(3)
                },
            }
        )
    return rows


def gallery_records(images: list[dict], threshold: float) -> list[dict]:
    """Per-image outcome used to pick and draw example images in build_reports.py."""
    rows = []
    for im in images:
        m = match_at_threshold(im, threshold)
        det = m["det"]
        confused_det = {j for _, j in m["confused"]}
        records = []
        for j in range(len(det["cls"])):
            kind = "tp" if m["det_match"][j] >= 0 else ("confused" if j in confused_det else "fp")
            records.append({"box": [round(float(v), 1) for v in im["boxes"][det["index"][j]]], "cls": int(det["cls"][j]), "conf": round(float(det["conf"][j]), 3), "kind": kind})
        confused_gt = {i for i, _ in m["confused"]}
        missed = [
            {"box": [round(float(v), 1) for v in im["gt_boxes"][i]], "cls": int(im["gt_cls"][i])}
            for i in range(len(im["gt_cls"]))
            if not m["gt_used"][i] or i in confused_gt
        ]
        tp = sum(r["kind"] == "tp" for r in records)
        fp = len(records) - tp
        rows.append(
            {
                "stem": im["stem"],
                "split": im["split"],
                "source": im["source"],
                "w": im["w"],
                "h": im["h"],
                "n_true": len(im["gt_cls"]),
                "tp": tp,
                "fp": fp,
                "fn": len(missed),
                "f1": round(2 * tp / max(1, 2 * tp + fp + len(missed)), 4),
                "detections": records,
                "missed": missed,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("results/raw/yolo26-blood-cell-detection/artifacts"))
    ap.add_argument("--labels", type=Path, default=Path("build/kaggle_dataset/labels"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    args = ap.parse_args()

    val_images, _ = load_split(args.artifacts, args.labels, "val")
    test_images, test_payload = load_split(args.artifacts, args.labels, "test")

    threshold, val_curve = choose_threshold(val_images)
    val_at_threshold = counts_at_threshold(val_images, threshold)
    print(f"Chosen confidence on validation: {threshold:.2f} (macro F1 {val_at_threshold['f1'].mean():.4f})")

    test_ap = compute_ap(test_images)
    s = summarise_ap(test_ap["ap"])
    print(f"Test mAP50 {s['map50']:.4f}  mAP50-95 {s['map50_95']:.4f}")

    # Guard: my numbers must agree with Ultralytics on the same model and data.
    reference = json.loads((args.artifacts / "ultralytics_test_metrics.json").read_text())
    gaps = {
        "map50": abs(s["map50"] - reference["map50"]),
        "map50_95": abs(s["map50_95"] - reference["map50_95"]),
    }
    print(f"Agreement with Ultralytics: {gaps} (Ultralytics: {reference['map50']:.4f} / {reference['map50_95']:.4f})")
    if max(gaps.values()) > MAX_AGREEMENT_GAP:
        raise SystemExit(f"Analysis disagrees with Ultralytics by more than {MAX_AGREEMENT_GAP}: {gaps}")

    print(f"Bootstrapping {args.bootstrap} resamples...", flush=True)
    intervals = bootstrap(test_images, args.bootstrap)

    op = counts_at_threshold(test_images, threshold)
    f1_curves, conf_grid = f1_vs_conf(test_images)
    pr, recall_grid = pr_curve(test_ap)
    environment = json.loads((args.artifacts / "environment.json").read_text())
    performance = {
        "images": len(test_images),
        "bootstrap_samples": args.bootstrap,
        "head": reference["head"],
        "winner": reference["winner"],
        "overall": {
            "map50": round(s["map50"], 4),
            "map50_95": round(s["map50_95"], 4),
            "map50_ci": intervals["map50"],
            "map50_95_ci": intervals["map50_95"],
        },
        "ultralytics_reference": {
            "map50": reference["map50"],
            "map50_95": reference["map50_95"],
            "map75": reference["map75"],
            "precision": reference["precision"],
            "recall": reference["recall"],
            "agreement_gap": {k: round(v, 5) for k, v in gaps.items()},
        },
        "operating": {
            "conf": threshold,
            "chosen_on": "validation",
            "val_macro_f1": round(float(val_at_threshold["f1"].mean()), 4),
            "macro_f1": round(float(op["f1"].mean()), 4),
            "macro_precision": round(float(op["precision"].mean()), 4),
            "macro_recall": round(float(op["recall"].mean()), 4),
            "val_curve": val_curve,
        },
        "per_class": {
            NAMES[c]: {
                "instances": int(op["tp"][c] + op["fn"][c]),
                "ap50": round(float(s["ap50"][c]), 4),
                "ap50_95": round(float(s["ap50_95"][c]), 4),
                "ap50_ci": intervals["ap50"][c],
                "ap50_95_ci": intervals["ap50_95"][c],
                "precision": round(float(op["precision"][c]), 4),
                "recall": round(float(op["recall"][c]), 4),
                "f1": round(float(op["f1"][c]), 4),
                "tp": int(op["tp"][c]),
                "fp": int(op["fp"][c]),
                "fn": int(op["fn"][c]),
            }
            for c in range(3)
        },
        "confusion_matrix": op["matrix"].tolist(),
        "curves": {
            "f1_conf": f1_curves,
            "conf_table": conf_grid,
            "pr": pr,
            "recall_table": recall_grid,
            "ap_by_iou": {NAMES[c]: [[round(float(t), 2), round(float(test_ap["ap"][c, k]), 4)] for k, t in enumerate(IOU_THRESHOLDS)] for c in range(3)},
        },
        "error_breakdown": error_breakdown(test_images, threshold),
        "by_source": per_source(test_images, threshold),
        "counts": count_accuracy(test_images, threshold),
        "speed": {
            "inference_ms": reference["speed_ms"]["inference"],
            "preprocess_ms": reference["speed_ms"]["preprocess"],
            "postprocess_ms": reference["speed_ms"]["postprocess"],
            "device": environment.get("gpu_name", "CPU"),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "performance.json").write_text(json.dumps(performance, indent=1))
    records = {
        "threshold": threshold,
        "test": gallery_records(test_images, threshold),
        "val": gallery_records(val_images, threshold),
    }
    (args.out / "gallery_items.json").write_text(json.dumps(records))

    for name, m in performance["per_class"].items():
        print(f"  {name:10s} n={m['instances']:5d} AP50 {m['ap50']:.4f} {m['ap50_ci']} AP50-95 {m['ap50_95']:.4f} {m['ap50_95_ci']} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} TP {m['tp']} FP {m['fp']} FN {m['fn']}")
    print("Confusion matrix (rows true, columns predicted):", performance["confusion_matrix"])
    for row in performance["by_source"]:
        print(f"  {row['label']:8s} n={row['images']:3d} mAP50 {row['map50']:.4f} mAP50-95 {row['map50_95']:.4f} macroF1 {row['f1_macro']:.3f}")
    print("Counts:", performance["counts"])
    print("Error breakdown:", performance["error_breakdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
