#!/usr/bin/env python3
"""Fine-tune YOLO26 for blood-cell detection on a Kaggle GPU.

Runs inside a Kaggle kernel. scripts/kaggle_jobs.py pushes this file with CONFIG filled in
from configs/experiment.yaml, so each kernel version records exactly what it ran.

Every candidate starts from COCO-pretrained weights and is compared on the validation
split. Only the winner ever touches the test split. Progress is printed as "##EVENT##"
JSON lines, which scripts/monitor.py reads from the live Kaggle log stream to drive the
dashboard. No credentials are needed inside the kernel.

With "evaluate_only" in CONFIG, the kernel skips training and repeats the final evaluation
from saved artifacts attached as a dataset, so an evaluation problem never costs a retrain.
"""
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

CONFIG = json.loads(r"""@@CONFIG@@""")

IS_KAGGLE = Path("/kaggle/input").exists()
INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path(os.environ.get("BLOOD_INPUT", "build/kaggle_dataset"))
WORK = Path("/kaggle/working") if IS_KAGGLE else Path(os.environ.get("BLOOD_WORK", "build/local_run"))
SCRATCH = Path("/tmp/txl-pbc") if IS_KAGGLE else WORK / "scratch"
OUT = WORK / "artifacts"
BORDER_PX = 16  # at the model's input scale, matching Ultralytics' validation padding
BORDER_GRAY = 114
if not IS_KAGGLE:  # local dry runs have no CUDA device 0
    CONFIG["train_args"]["device"] = os.environ.get("BLOOD_DEVICE", "cpu")


def emit(kind, **payload):
    """Print one machine-readable progress event to stdout."""
    print("##EVENT## " + json.dumps({"type": kind, "t": round(time.time(), 2), **payload}), flush=True)


def never_raises(fn):
    """A monitoring bug must never be able to kill a long training run."""

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print(f"[monitoring] {fn.__name__} failed: {exc!r}", flush=True)

    return wrapper


def ensure_ultralytics():
    emit("stage", stage="install", message=f"Installing ultralytics=={CONFIG['ultralytics_version']}")
    try:
        import ultralytics

        if ultralytics.__version__ == CONFIG["ultralytics_version"]:
            return
    except ImportError:
        pass
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", f"ultralytics=={CONFIG['ultralytics_version']}"],
        check=True,
    )


def locate_dataset():
    """Find the dataset wherever Kaggle mounted it, then copy it to writable scratch space."""
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    candidates = sorted(INPUT_ROOT.rglob("data.yaml"))
    if candidates:
        source = candidates[0].parent
        for name in ("images", "labels"):
            folder = source / name
            if folder.exists():
                shutil.copytree(folder, SCRATCH / name)
            else:  # Kaggle may keep a directory uploaded in zip mode as an archive
                with zipfile.ZipFile(source / f"{name}.zip") as z:
                    z.extractall(SCRATCH / name)
        shutil.copy2(candidates[0], SCRATCH / "data.yaml")
        if (source / "manifest.json").exists():
            shutil.copy2(source / "manifest.json", OUT / "dataset_manifest.json")
    else:
        raise FileNotFoundError(f"data.yaml not found under {INPUT_ROOT}: {[str(p) for p in INPUT_ROOT.rglob('*')][:30]}")

    text = (SCRATCH / "data.yaml").read_text()
    (SCRATCH / "data.yaml").write_text(f"path: {SCRATCH}\n{text}")
    counts = {s: len(list((SCRATCH / "images" / s).glob("*.png"))) for s in ("train", "val", "test")}
    if min(counts.values()) == 0:
        raise RuntimeError(f"Empty split after locating dataset: {counts}")
    emit("dataset", counts=counts, source=str(candidates[0].parent))
    return SCRATCH / "data.yaml"


def release_gpu(context):
    """Free what a finished model held. Trainers keep reference cycles, so without this
    GPU memory piles up from one candidate to the next."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        emit(
            "gpu_memory",
            context=context,
            allocated_gb=round(torch.cuda.memory_allocated() / 1e9, 3),
            reserved_gb=round(torch.cuda.memory_reserved() / 1e9, 3),
        )


def environment_info():
    import torch
    import ultralytics

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    emit("env", **info)
    return info


def class_metrics(box, names):
    """Per-class P, R, AP50 and AP50-95 from an Ultralytics Metric object."""
    out = {}
    for i, c in enumerate(box.ap_class_index):
        c = int(c)
        out[names[c]] = {
            "precision": round(float(box.p[i]), 5),
            "recall": round(float(box.r[i]), 5),
            "ap50": round(float(box.ap50[i]), 5),
            "ap50_95": round(float(box.ap[i]), 5),
        }
    return out


def make_callbacks(run_name, names):
    state = {"last_batch_emit": 0.0, "batch": 0, "epoch_open": False}

    @never_raises
    def on_train_start(trainer):
        emit("run_start", run=run_name, epochs=trainer.epochs, batches_per_epoch=len(trainer.train_loader))

    @never_raises
    def on_train_epoch_start(trainer):
        state["batch"] = 0
        state["epoch_open"] = True

    @never_raises
    def on_train_batch_end(trainer):
        state["batch"] += 1
        now = time.time()
        if now - state["last_batch_emit"] < 5:
            return
        state["last_batch_emit"] = now
        emit("batch", run=run_name, epoch=trainer.epoch + 1, batch=state["batch"], of=len(trainer.train_loader))

    @never_raises
    def on_fit_epoch_end(trainer):
        import torch

        # Ultralytics fires this again when it re-validates best.pt after the last epoch.
        # That is not a training epoch, and it must not appear in the training curves.
        if not state["epoch_open"]:
            return
        state["epoch_open"] = False
        row = {"epoch": trainer.epoch + 1, "epoch_time": round(float(trainer.epoch_time or 0), 2)}
        for key, value in {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
            **trainer.lr,
        }.items():
            row[key] = round(float(value), 6)
        row["fitness"] = round(float(trainer.fitness or 0), 6)
        row["best_fitness"] = round(float(trainer.best_fitness or 0), 6)
        row["per_class"] = class_metrics(trainer.validator.metrics.box, names)
        if torch.cuda.is_available():
            row["gpu_mem_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        emit("epoch", run=run_name, **row)

    return {
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_batch_end": on_train_batch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
    }


def train_candidate(candidate, data_yaml, names):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    run_name = candidate["name"]
    emit("stage", stage="train", message=f"Training {run_name}", run=run_name)
    model = YOLO(candidate["weights"])
    for event, fn in make_callbacks(run_name, names).items():
        model.add_callback(event, fn)
    args = {**CONFIG["train_args"], **candidate.get("overrides", {})}
    started = time.time()
    model.train(data=str(data_yaml), project=str(WORK / "runs"), name=run_name, exist_ok=True, **args)
    run_dir = WORK / "runs" / run_name
    best_path = run_dir / "weights" / "best.pt"

    # Score both YOLO26 heads on validation: one-to-many + NMS (default), and NMS-free.
    scores = {}
    for head, nms_arg in (("nms", None), ("nms_free", False)):
        evaluator = YOLO(str(best_path))
        metrics = evaluator.val(
            data=str(data_yaml),
            split="val",
            imgsz=args["imgsz"],
            batch=args["batch"],
            device=args["device"],
            nms=nms_arg,
            plots=False,
            verbose=False,
            project=str(WORK / "runs"),
            name=f"{run_name}_val_{head}",
            exist_ok=True,
        )
        scores[head] = {
            "map50": round(float(metrics.box.map50), 5),
            "map50_95": round(float(metrics.box.map), 5),
            "precision": round(float(metrics.box.mp), 5),
            "recall": round(float(metrics.box.mr), 5),
            "per_class": class_metrics(metrics.box, names),
        }
    best_model = YOLO(str(best_path)).model
    result = {
        "name": run_name,
        "weights": candidate["weights"],
        "overrides": candidate.get("overrides", {}),
        "train_seconds": round(time.time() - started, 1),
        "params": sum(p.numel() for p in best_model.parameters()),
        "gflops": round(float(get_flops(best_model, args["imgsz"])), 2),
        "val": scores,
        "best_head": max(scores, key=lambda h: scores[h]["map50_95"]),
    }
    result["val_map50_95"] = scores[result["best_head"]]["map50_95"]
    emit("run_end", run=run_name, result=result)

    dest = OUT / "runs" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("results.csv", "args.yaml"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dest / name)
    del model, evaluator, best_model
    release_gpu(run_name)
    return result, best_path


def select_winner(results):
    """Pick the smallest candidate within the margin of the best validation mAP50-95.

    Differences smaller than the margin are within the noise of a validation set this
    size, so they should not justify a larger, slower model.
    """
    best_score = max(r["val_map50_95"] for r in results)
    eligible = [r for r in results if r["val_map50_95"] >= best_score - CONFIG["selection_margin"]]
    return min(eligible, key=lambda r: (r["params"], -r["val_map50_95"])), best_score


def predict_with_border(model, image_path, conf, imgsz, device, nms_arg):
    """Predict with the gray border that Ultralytics' validation adds and predict() does not.

    Without it, cells cut off by the image edge are detected much worse, so this is the
    official way to run the model. Images go one at a time, so results cannot depend on
    batch order. scripts/predict.py is the same procedure for everyday use.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    border = round(BORDER_PX * max(h, w) / imgsz)
    gray = (BORDER_GRAY, BORDER_GRAY, BORDER_GRAY)
    padded = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=gray)
    result = model.predict(
        padded, conf=conf, iou=0.7, imgsz=imgsz + 2 * BORDER_PX, device=device, max_det=300, nms=nms_arg, verbose=False
    )[0]
    boxes = result.boxes.xyxy.cpu().numpy() - border
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])  # drop boxes wholly in the border
    return (h, w), boxes[keep], result.boxes.conf.cpu().numpy()[keep], result.boxes.cls.cpu().numpy()[keep], result.speed


def predictions_to_json(model, image_dir, head, imgsz, device):
    """Raw predictions above a low confidence floor, for offline analysis without a GPU."""
    rows = []
    speed = []
    nms_arg = False if head == "nms_free" else None
    for path in sorted(Path(image_dir).glob("*.png")):
        shape, boxes, conf, cls, timing = predict_with_border(model, path, 0.001, imgsz, device, nms_arg)
        rows.append(
            {
                "stem": path.stem,
                "shape": [int(v) for v in shape],
                "boxes": [[round(float(v), 2) for v in b] for b in boxes],
                "conf": [round(float(v), 5) for v in conf],
                "cls": [int(v) for v in cls],
            }
        )
        speed.append(timing)
    keys = speed[0].keys() if speed else []
    mean_speed = {k: round(sum(s[k] for s in speed) / len(speed), 3) for k in keys}
    return {"head": head, "border_px_at_640": BORDER_PX, "predictions": rows, "mean_speed_ms": mean_speed}


def final_evaluation(winner, best_path, data_yaml, names):
    from ultralytics import YOLO

    head = winner["best_head"]
    args = CONFIG["train_args"]
    emit("stage", stage="evaluate", message=f"Final evaluation of {winner['name']} ({head}) on test")

    if Path(best_path).resolve() != (OUT / "best.pt").resolve():
        shutil.copy2(best_path, OUT / "best.pt")
    # Validation predictions choose the operating thresholds. Test predictions then
    # measure them, so that the test split does not influence any decision.
    for split in ("val", "test"):
        model = YOLO(str(best_path))
        payload = predictions_to_json(model, SCRATCH / "images" / split, head, args["imgsz"], args["device"])
        (OUT / f"predictions_{split}.json").write_text(json.dumps(payload))
        del model
        release_gpu(f"predictions_{split}")

    nms_arg = False if head == "nms_free" else None
    metrics = YOLO(str(best_path)).val(
        data=str(data_yaml),
        split="test",
        imgsz=args["imgsz"],
        batch=args["batch"],
        device=args["device"],
        nms=nms_arg,
        plots=True,
        verbose=True,
        project=str(WORK / "runs"),
        name="final_test",
        exist_ok=True,
    )
    summary = {
        "winner": winner["name"],
        "head": head,
        "map50": round(float(metrics.box.map50), 5),
        "map50_95": round(float(metrics.box.map), 5),
        "map75": round(float(metrics.box.map75), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "per_class": class_metrics(metrics.box, names),
        "speed_ms": {k: round(float(v), 3) for k, v in metrics.speed.items()},
        "confusion_matrix": metrics.confusion_matrix.matrix.astype(int).tolist(),
    }
    (OUT / "ultralytics_test_metrics.json").write_text(json.dumps(summary, indent=1))
    emit("test_result", **summary)

    plots = OUT / "ultralytics_plots"
    plots.mkdir(parents=True, exist_ok=True)
    final_dir = WORK / "runs" / "final_test"
    for png in final_dir.glob("*.png"):
        shutil.copy2(png, plots / png.name)


def evaluate_saved_model(data_yaml, names):
    """Repeat the final evaluation from the artifacts of an earlier training run."""
    selection_file = sorted(INPUT_ROOT.rglob("selection.json"))[0]
    shutil.copytree(selection_file.parent, OUT, dirs_exist_ok=True)
    selection = json.loads(selection_file.read_text())
    winner = next(r for r in selection["results"] if r["name"] == selection["winner"])
    emit("selection", **selection)
    final_evaluation(winner, OUT / "best.pt", data_yaml, names)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=1))
    emit("job_start", kaggle=IS_KAGGLE, config=CONFIG)
    try:
        ensure_ultralytics()
        env = environment_info()
        (OUT / "environment.json").write_text(json.dumps(env, indent=1))
        if IS_KAGGLE and not env["cuda"]:
            raise RuntimeError("No GPU visible in the Kaggle session, refusing to train on CPU.")
        data_yaml = locate_dataset()
        names = CONFIG["names"]

        if CONFIG.get("evaluate_only"):
            evaluate_saved_model(data_yaml, names)
            emit("done", status="complete")
            return

        results, paths = [], {}
        for candidate in CONFIG["candidates"]:
            result, best_path = train_candidate(candidate, data_yaml, names)
            results.append(result)
            paths[result["name"]] = best_path
            (OUT / "candidates.json").write_text(json.dumps(results, indent=1))

        winner, best_score = select_winner(results)
        selection = {
            "winner": winner["name"],
            "best_val_map50_95": best_score,
            "margin": CONFIG["selection_margin"],
            "results": results,
        }
        (OUT / "selection.json").write_text(json.dumps(selection, indent=1))
        emit("selection", **selection)

        final_evaluation(winner, paths[winner["name"]], data_yaml, names)
        emit("done", status="complete")
    except Exception as exc:
        emit("failed", error=str(exc), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        # Only the curated artifacts stay in the output: full runs hold large checkpoints.
        if IS_KAGGLE:
            shutil.rmtree(WORK / "runs", ignore_errors=True)
            for weights in WORK.glob("*.pt"):
                weights.unlink()


if __name__ == "__main__":
    main()
