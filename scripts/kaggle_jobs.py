#!/usr/bin/env python3
"""Upload the dataset to Kaggle and push the training kernel. Both stay private.

Commands:
  upload-dataset    create or update the private Kaggle dataset from build/kaggle_dataset
  upload-artifacts  store the artifacts of a training run as a private dataset, so that
                    the final evaluation can be repeated without training again
  dataset-status    show the processing status of the training dataset
  render            write the kernel to build/kernel without pushing
  push              render and push the kernel, which starts a GPU run

--smoke trains the smallest and a large candidate for two epochs, which validates the whole
pipeline cheaply, including GPU memory across candidates.
--evaluate-only repeats the final evaluation from the artifacts uploaded earlier.
The Kaggle token is read from .env and is never written into the kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OWNER = "gabrielaschneider"
DATASET = f"{OWNER}/txl-pbc-blood-cell-detection"
ARTIFACTS_DATASET = f"{OWNER}/yolo26-blood-cell-artifacts"
KERNEL_SLUG = "yolo26-blood-cell-detection"
ACCELERATOR = "NvidiaTeslaT4"
BUILD = ROOT / "build" / "kernel"
KAGGLE = str(Path(sys.executable).parent / "kaggle")


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run([KAGGLE, *args], cwd=ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode:
        raise SystemExit(result.returncode)
    return result


def load_config(smoke: bool, evaluate_only: bool) -> dict:
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    if smoke:
        config["candidates"] = [
            {"name": "smoke_yolo26n", "weights": "yolo26n.pt"},
            {"name": "smoke_yolo26m", "weights": "yolo26m.pt"},
        ]
        config["train_args"].update(epochs=2, close_mosaic=1, cache=False)
    if evaluate_only:
        config["evaluate_only"] = True
    return config


def upload_artifacts(source: Path) -> None:
    """Publish a training run's artifacts as a private dataset for evaluate-only runs."""
    if not (source / "selection.json").exists() or not (source / "best.pt").exists():
        raise SystemExit(f"{source} must contain selection.json and best.pt")
    (source / "dataset-metadata.json").write_text(
        json.dumps({"title": "YOLO26 Blood Cell Artifacts", "id": ARTIFACTS_DATASET, "licenses": [{"name": "unknown"}]}, indent=1)
    )
    exists = kaggle("datasets", "status", ARTIFACTS_DATASET, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(source), "-m", "Update artifacts", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(source), "-r", "zip")


def render(smoke: bool, evaluate_only: bool) -> dict:
    BUILD.mkdir(parents=True, exist_ok=True)
    config = load_config(smoke, evaluate_only)
    code = (ROOT / "kaggle" / "train_kernel.py").read_text().replace("@@CONFIG@@", json.dumps(config))
    (BUILD / "train_kernel.py").write_text(code)
    slug = KERNEL_SLUG + ("-smoke" if smoke else "") + ("-eval" if evaluate_only else "")
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": slug.replace("-", " ").title().replace("Yolo26", "YOLO26"),
        "code_file": "train_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": ACCELERATOR,
        "dataset_sources": [DATASET, ARTIFACTS_DATASET] if evaluate_only else [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (BUILD / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1))
    return metadata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("upload-dataset", "upload-artifacts", "dataset-status", "render", "push"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--artifacts", type=Path, help="artifacts folder for upload-artifacts")
    ap.add_argument("--message", default="Update dataset")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN missing from .env")

    if args.command == "upload-dataset":
        folder = ROOT / "build" / "kaggle_dataset"
        exists = kaggle("datasets", "status", DATASET, check=False).returncode == 0
        if exists:
            kaggle("datasets", "version", "-p", str(folder), "-m", args.message, "-r", "zip")
        else:
            kaggle("datasets", "create", "-p", str(folder), "-r", "zip")
    elif args.command == "dataset-status":
        kaggle("datasets", "status", DATASET)
    elif args.command == "upload-artifacts":
        upload_artifacts(args.artifacts)
    elif args.command == "render":
        print(json.dumps(render(args.smoke, args.evaluate_only), indent=1))
    elif args.command == "push":
        metadata = render(args.smoke, args.evaluate_only)
        kaggle("kernels", "push", "-p", str(BUILD), "--accelerator", ACCELERATOR)
        record = {"kernel": metadata["id"], "pushed_at": time.time(), "smoke": args.smoke}
        (ROOT / "build" / "last_push.json").write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
