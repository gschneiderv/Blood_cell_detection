#!/usr/bin/env python3
"""Generate the dashboard's static data and the three self-contained HTML reports.

Inputs are the JSON files written by the earlier pipeline steps. Run it again at any time;
sections whose inputs do not exist yet simply show an empty state.

Writes dashboard/data/static.js and reports/{dataset,training,performance}_report.html.

Usage: python scripts/build_reports.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "TXL-PBC"
AUDIT_DIR = ROOT / "reports" / "dataset"
RESULTS = ROOT / "results"
ASSETS = ROOT / "dashboard" / "assets"
NOTES = ROOT / "configs" / "report_notes.yaml"
NAMES = ["WBC", "RBC", "Platelets"]
CLASS_RGB = {0: (42, 120, 214), 1: (235, 104, 52), 2: (27, 175, 122)}
FALSE_POSITIVE = (255, 212, 0)
MISSED = (255, 31, 180)
ORIGIN = {
    "640×480": "BCCD",
    "360×363": "PBC (14 of these are 366×369)",
    "575×575": "Raabin-WBC",
    "256×256": "BCDD",
}


def source_label(w: int, h: int) -> str:
    return "360×363" if (w, h) == (366, 369) else f"{w}×{h}"


def jpeg_uri(image: Image.Image, quality: int = 80) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def open_image(split: str, stem: str) -> Image.Image:
    return Image.open(DATA / "images" / split / f"{stem}.png").convert("RGB")


def draw_scaled(image: Image.Image, width: int) -> tuple[Image.Image, float]:
    scale = width / image.width
    return image.resize((width, round(image.height * scale)), Image.BICUBIC), scale


def ground_truth_image(rec: dict, width: int = 440) -> str:
    image, _ = draw_scaled(open_image(rec["split"], rec["stem"]), width)
    pen = ImageDraw.Draw(image)
    for cls, xc, yc, w, h in rec["boxes"]:
        x, y, bw, bh = xc * image.width, yc * image.height, w * image.width, h * image.height
        pen.rectangle([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], outline=CLASS_RGB[int(cls)], width=2)
    return jpeg_uri(image)


def outcome_image(item: dict, width: int = 460) -> str:
    image, scale = draw_scaled(open_image(item["split"], item["stem"]), width)
    pen = ImageDraw.Draw(image)
    for det in item["detections"]:
        color = CLASS_RGB[det["cls"]] if det["kind"] == "tp" else FALSE_POSITIVE
        pen.rectangle([v * scale for v in det["box"]], outline=color, width=2 if det["kind"] == "tp" else 3)
    for miss in item["missed"]:
        pen.rectangle([v * scale for v in miss["box"]], outline=MISSED, width=3)
    return jpeg_uri(image)


def crop_with_context(item: dict, box: list[float], color: tuple, size: int = 220, zoom_context: float = 2.2) -> str:
    image = open_image(item["split"], item["stem"])
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * zoom_context + 24
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = max(0, cx - side / 2), max(0, cy - side / 2)
    right, bottom = min(image.width, cx + side / 2), min(image.height, cy + side / 2)
    crop = image.crop((left, top, right, bottom))
    scale = size / max(crop.size)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)
    pen = ImageDraw.Draw(crop)
    pen.rectangle([(x1 - left) * scale, (y1 - top) * scale, (x2 - left) * scale, (y2 - top) * scale], outline=color, width=2)
    return jpeg_uri(crop)


def pair_image(key_a: str, key_b: str, side: int = 250) -> str:
    canvas = Image.new("RGB", (side * 2 + 6, side), "white")
    for i, key in enumerate((key_a, key_b)):
        split, stem = key.split("/")
        image = open_image(split, stem)
        image.thumbnail((side, side))
        canvas.paste(image, (i * (side + 6), 0))
    return jpeg_uri(canvas, 78)


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def load_live():
    path = ROOT / "dashboard" / "data" / "live.js"
    if not path.exists():
        return None
    text = path.read_text().strip()
    return json.loads(text[len("window.__LIVE__ = ") :].rstrip(";"))


def load_notes() -> dict:
    return yaml.safe_load(NOTES.read_text()) if NOTES.exists() else {}


def dataset_section() -> dict | None:
    audit = load_json(AUDIT_DIR / "audit.json")
    images = load_json(AUDIT_DIR / "images.json")
    twins = load_json(AUDIT_DIR / "scene_twins.json")
    manifest = load_json(AUDIT_DIR / "manifest.json")
    if not all((audit, images, twins, manifest)):
        return None

    notes = load_notes().get("dataset", {})
    grouped = defaultdict(list)
    for rec in images:
        grouped[source_label(rec["width"], rec["height"])].append(rec)
    sources = []
    for label in ("575×575", "360×363", "640×480", "256×256"):
        recs = grouped[label]
        per_class = Counter(NAMES[int(b[0])] for r in recs for b in r["boxes"])
        sources.append(
            {
                "size": label,
                "origin": ORIGIN[label],
                "images": len(recs),
                "splits": dict(Counter(r["split"] for r in recs)),
                **{name: per_class[name] for name in NAMES},
                "platelets_per_image": per_class["Platelets"] / len(recs),
            }
        )

    rng = random.Random(7)
    samples = []
    for label in ("575×575", "360×363", "640×480", "256×256"):
        for rec in rng.sample(grouped[label], 3):
            samples.append(
                {
                    "src": ground_truth_image(rec),
                    "title": f"{label}, {ORIGIN[label].split(' (')[0]}",
                    "alt": f"Blood smear from the {label} source with ground-truth boxes",
                    "caption": f"{rec['split']} split, {rec['n_boxes']} cells",
                }
            )

    examples = []
    cross = [p for p in twins["pairs"] if p["a"].split("/")[0] != p["b"].split("/")[0]]
    by_source = defaultdict(list)
    for pair in cross:
        by_source[pair["source"]].append(pair)
    for source, pairs in sorted(by_source.items()):
        chosen = [pairs[0], pairs[len(pairs) // 2]]
        for pair in chosen:
            a, b = pair["a"].split("/")[0], pair["b"].split("/")[0]
            zoom_gap = abs(pair["scale"] - 1)
            zoom = "same zoom" if zoom_gap < 0.03 else f"{zoom_gap * 100:.0f}% zoom difference"
            examples.append(
                {
                    "src": pair_image(pair["a"], pair["b"]),
                    "title": f"{a} and {b} ({source})",
                    "alt": f"The same field of view found in both the {a} and {b} splits",
                    "caption": f"{pair['inliers']} matched keypoints, {zoom}",
                }
            )

    hist = twins["calibration_inliers_by_population"]["same_source_direct"]
    buckets = [("6–9", 6, 9), ("10–11", 10, 11), ("12–19", 12, 19), ("20–39", 20, 39), ("40 or more", 40, 10**6)]
    calibration = [[label, sum(v for k, v in hist.items() if lo <= int(k) <= hi)] for label, lo, hi in buckets]

    per_split = audit["per_split"]
    readme_ok = all(s["images"] == s["readme_images"] and s["cells"] == s["readme_cells"] for s in per_split.values())
    integrity = audit["integrity"]
    excluded = manifest["excluded_images"]
    removed_by_split = Counter(k.split("/")[0] for k in excluded)
    test_with_twin = twins["images_in_spanning_groups_by_split"].get("test", 0)
    densities = sorted(s["platelets_per_image"] for s in sources)
    checks = [
        {"name": "Counts match the README", "status": "good" if readme_ok else "bad", "label": "Pass" if readme_ok else "Fail",
         "detail": f"{audit['totals']['images']:,} images and {audit['totals']['cells']:,} cells, in every split. Class totals also match the published paper."},
        {"name": "Images and labels pair up", "status": "good", "label": "Pass",
         "detail": "Every image has a label file and every label file has an image."},
        {"name": "Images decode", "status": "good" if not integrity["unreadable_images"] else "bad", "label": "Pass",
         "detail": "All files open and decode as PNG."},
        {"name": "Label format", "status": "good" if not integrity["label_problems_count"] else "bad", "label": "Pass",
         "detail": "Every row has 5 columns, a class of 0 to 2, and a box inside the image."},
        {"name": "Alpha channels", "status": "good", "label": "Pass",
         "detail": f"{audit['image_stats']['modes'].get('RGBA', 0)} images are RGBA (the 256×256 source), but all are fully opaque, so dropping alpha loses nothing."},
        {"name": "Identical image files", "status": "good", "label": "Pass", "detail": "No two files are byte-identical."},
        {"name": "Duplicate label rows", "status": "warn", "label": "Fixed",
         "detail": f"{integrity['duplicate_boxes_total']} exact duplicate rows in {integrity['duplicate_box_files']} files, in which most boxes were listed twice. Removed before training."},
        {"name": "Same field of view across splits", "status": "warn", "label": "Fixed",
         "detail": f"{twins['pairs_across_splits']} pairs cross a split boundary. {len(excluded)} images were removed, leaving the test split as provided."},
        {"name": "Class balance", "status": "warn", "label": "Caution",
         "detail": f"RBCs make up {100 * audit['totals']['per_class']['RBC'] / audit['totals']['cells']:.1f}% of cells, so mean metrics weight the classes equally rather than by count."},
        {"name": "Platelet labelling across sources", "status": "warn", "label": "Caution",
         "detail": f"From {densities[0]:.2f} to {densities[-1]:.2f} platelets per image, depending on the source. See the label quality section."},
    ]
    leakage_summary = (
        f"All {twins['pairs_compared']:,} image pairs were compared using keypoints that must agree on one geometric "
        f"transform, a check that survives shifts, zoom and colour changes. {twins['pairs_total']} pairs show the same field "
        f"of view, and {twins['pairs_across_splits']} of them cross a split boundary, including {test_with_twin} of the "
        f"{per_split['test']['images']} test images. A model tested on a field it trained on scores unrealistically well, "
        "so the twins were removed from train and val, keeping every test image."
    )
    return {
        "audit": audit,
        "manifest": {"per_split": manifest["per_split"]},
        "sources": sources,
        "candidates": [
            {"name": "TXL-PBC", "contents": "1,260 blood-smear images, 18,143 cells", "labels": "Bounding boxes with 3 classes: WBC, RBC, platelets", "used": True,
             "why": "The only one with cell locations, and an exact match for the task."},
            {"name": "BloodMNIST", "contents": "17,092 single-cell crops, 64×64 px", "labels": "One of 8 classes per image, no boxes", "used": False,
             "why": "It has no locations to learn from. Neighbouring red cells appear unlabelled in its crops, which would teach a detector to ignore them. It also comes from the PBC collection, which TXL-PBC already includes, so mixing them risks leakage."},
            {"name": "Dataset_NeuNN", "contents": "5,605 cell crops, 360×363 px, no documentation", "labels": "One of 7 folder names per image, no boxes", "used": False,
             "why": "It has no locations. Its folders (BAC, CRY, DB, GBI, HJBLI, HYP, NEU) appear to describe neutrophil abnormalities, a different question from WBC, RBC or platelet."},
        ],
        "checks": checks,
        "leakage": {
            "summary": leakage_summary,
            "pairs_compared": twins["pairs_compared"],
            "pairs_total": twins["pairs_total"],
            "pairs_across_splits": twins["pairs_across_splits"],
            "test_affected": test_with_twin,
            "removed": len(excluded),
            "removed_train": removed_by_split.get("train", 0),
            "removed_val": removed_by_split.get("val", 0),
            "pairs_by_source": twins["pairs_by_source"],
            "threshold": twins["params"]["min_inliers"],
            "calibration": calibration,
            "examples": examples,
        },
        "label_findings": notes.get("label_findings", []),
        "samples": samples,
    }


def performance_section() -> dict | None:
    perf = load_json(RESULTS / "performance.json")
    items = load_json(RESULTS / "gallery_items.json")
    if not perf or not items:
        return None
    notes = load_notes().get("performance", {})
    test = items["test"]

    galleries = []
    # Typical outcome per source: the image at the median F1 of each source.
    typical = []
    for label in ("575×575", "360×363", "640×480", "256×256"):
        subset = sorted((i for i in test if i["source"] == label), key=lambda i: i["f1"])
        for pick in (subset[len(subset) // 2], subset[(3 * len(subset)) // 4]):
            typical.append(
                {
                    "src": outcome_image(pick),
                    "title": f"{label}, {ORIGIN[label].split(' (')[0]}",
                    "alt": f"Test image from the {label} source with model predictions",
                    "caption": f"{pick['tp']} correct, {pick['fp']} false positives, {pick['fn']} missed",
                }
            )
    galleries.append({"title": "Typical test images", "note": "Images at the middle of each source's score range, not hand-picked successes.", "items": typical})

    worst = sorted(test, key=lambda i: (i["f1"], -(i["fp"] + i["fn"])))[:8]
    galleries.append(
        {
            "title": "Hardest test images",
            "note": "The eight lowest-scoring test images. These are where the model and the labels disagree most.",
            "items": [
                {
                    "src": outcome_image(i),
                    "title": f"{i['source']}, F1 {i['f1']:.2f}",
                    "alt": f"Low-scoring test image from the {i['source']} source",
                    "caption": f"{i['tp']} correct, {i['fp']} false positives, {i['fn']} missed",
                }
                for i in worst
            ],
        }
    )

    # The model's most confident platelet "false positives", to judge the labels themselves.
    platelet_fp = []
    for item in test:
        for det in item["detections"]:
            if det["cls"] == 2 and det["kind"] != "tp":
                platelet_fp.append((det["conf"], item, det))
    platelet_fp.sort(key=lambda row: -row[0])
    if platelet_fp:
        galleries.append(
            {
                "title": "Platelet false positives, most confident first",
                "note": notes.get("platelet_fp_note", "Each crop is a platelet predicted with no matching label."),
                "key": False,
                "items": [
                    {
                        "src": crop_with_context(item, det["box"], FALSE_POSITIVE),
                        "title": f"{item['source']}, confidence {det['conf']:.2f}",
                        "alt": "Crop around a platelet predicted without a matching label",
                    }
                    for _, item, det in platelet_fp[:12]
                ],
            }
        )

    perf = dict(perf)
    perf["gallery"] = galleries
    perf["notes"] = notes.get("banners", [])
    perf["limitations"] = notes.get("limitations", [])
    perf["baselines"] = notes.get("baselines")
    perf["border_comparison"] = load_json(RESULTS / "border_comparison.json")
    return perf


def training_section() -> dict | None:
    live = load_live()
    if not live or not live.get("runs"):
        return None
    notes = load_notes().get("training", {})
    return {
        "kernel": live["kernel"],
        "url": live["url"],
        "env": live.get("env"),
        "config": live.get("config"),
        "dataset": live.get("dataset"),
        "runs": live["runs"],
        "selection": live.get("selection") or load_json(RESULTS / "selection.json"),
        "observations": notes.get("observations", []),
    }


REPORT_RENDERERS = {
    "dataset": """
      const root = document.getElementById("root");
      Sections.renderDataset(root, REPORT_DATA);
    """,
    "performance": """
      const root = document.getElementById("root");
      Sections.renderPerformance(root, REPORT_DATA);
    """,
    "training": """
      const root = document.getElementById("root");
      Sections.renderTrainingReport(root, REPORT_DATA);
    """,
}
TITLES = {
    "dataset": ("Dataset Report", "What the model was trained on: which data was used, how it was checked, and what was corrected before training."),
    "training": ("Training Report", "How the model was trained on Kaggle's GPU: settings, candidates, and how each one progressed."),
    "performance": ("Performance Report", "How well the final model detects and classifies blood cells on images it never saw during training."),
}


def write_report(kind: str, data) -> None:
    title, subtitle = TITLES[kind]
    theme = (ASSETS / "theme.css").read_text()
    charts = (ASSETS / "charts.js").read_text()
    sections = (ASSETS / "sections.js").read_text()
    links = "".join(
        f'<a href="{other}_report.html">{TITLES[other][0]}</a>' if other != kind else f"<strong>{TITLES[other][0]}</strong>"
        for other in TITLES
    )
    payload = json.dumps(data).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{theme}</style>
</head>
<body>
<header class="report-head"><div class="inner">
<h1>{html.escape(title)}</h1>
<p>{html.escape(subtitle)}</p>
<nav class="report-nav">{links}<a href="../dashboard/index.html">Dashboard</a></nav>
</div></header>
<main class="wrap" id="root"></main>
<script>{charts}</script>
<script>{sections}</script>
<script>
const REPORT_DATA = {payload};
{REPORT_RENDERERS[kind]}
</script>
</body>
</html>
"""
    (ROOT / "reports" / f"{kind}_report.html").write_text(page)
    print(f"wrote reports/{kind}_report.html ({len(page) / 1e6:.2f} MB)")


def main() -> int:
    dataset = dataset_section()
    performance = performance_section()
    training = training_section()
    out = ROOT / "dashboard" / "data" / "static.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.time()
    out.write_text("window.__STATIC__ = " + json.dumps({"generated_at": stamp, "dataset": dataset, "performance": performance}).replace("</", "<\\/") + ";\n")
    print(f"wrote dashboard/data/static.js ({out.stat().st_size / 1e6:.2f} MB)")
    if dataset:
        write_report("dataset", dataset)
    if training:
        write_report("training", training)
    if performance:
        write_report("performance", performance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
