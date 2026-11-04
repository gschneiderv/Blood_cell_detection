#!/usr/bin/env python3
"""Follow a Kaggle kernel and keep the dashboard's data file up to date.

Reads the kernel's live log stream, replays its "##EVENT##" lines into a single state
object, and writes dashboard/data/live.js, which dashboard/index.html re-reads every few
seconds. Open the dashboard directly from disk: no web server is needed.

When the kernel finishes, the monitor downloads its output to results/raw, keeps the final
model in models/, evaluates it, rebuilds the reports and dashboard data, and exits.

With --base-kernel, the dashboard keeps the training history of an earlier kernel and adds
this kernel's progress on top. Evaluate-only runs use this after a failed final evaluation.

Usage: python scripts/monitor.py [--kernel owner/slug] [--base-kernel owner/slug]
                                 [--interval 30] [--no-download]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
LIVE_JS = ROOT / "dashboard" / "data" / "live.js"
RAW_DIR = ROOT / "results" / "raw"
EVENT_PREFIX = "##EVENT## "
LOG_TAIL = 300
WRITE_EVERY_S = 5
STATUS_TIMEOUT_S = 60
FINAL_READ_TIMEOUT_S = 180
KEPT_ARTIFACTS = ("selection.json", "environment.json", "config.json", "ultralytics_test_metrics.json")
# Kaggle's client splits events at the byte 0x85, which occurs inside some emoji.
EVENT_FRAGMENT = re.compile(r'^\{"stream_name":"\w+","time":[\d.]+,"data":"|\\n"\},?$')
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\[K|\[\d+m")
PROGRESS_BAR = re.compile(r"\d+%\s*[━─╸╺█▏▎▍▌▋▊▉ ]{3,}")
TERMINAL = {"complete", "error", "cancel_acknowledged", "cancel_requested"}


def new_state(kernel: str) -> dict:
    owner, slug = kernel.split("/")[:2]
    return {
        "kernel": kernel,
        "url": f"https://www.kaggle.com/code/{owner}/{slug}",
        "status": "unknown",
        "failure": None,
        "monitor_updated": None,
        "last_event_t": None,
        "job_started": None,
        "stage": None,
        "stage_message": None,
        "env": None,
        "config": None,
        "dataset": None,
        "runs": {},
        "selection": None,
        "test": None,
        "error": None,
        "log_tail": [],
    }


def apply_event(state: dict, ev: dict) -> None:
    """Fold one progress event into the state. Replaying all events rebuilds it exactly."""
    kind = ev.get("type")
    state["last_event_t"] = ev.get("t", state["last_event_t"])
    if kind == "job_start":
        state["job_started"] = ev["t"]
        state["config"] = ev.get("config")
    elif kind == "stage":
        state["stage"] = ev.get("stage")
        state["stage_message"] = ev.get("message")
    elif kind == "env":
        state["env"] = {k: v for k, v in ev.items() if k not in ("type", "t")}
    elif kind == "dataset":
        state["dataset"] = {"counts": ev.get("counts"), "source": ev.get("source")}
    elif kind == "run_start":
        state["runs"][ev["run"]] = {
            "status": "training",
            "started": ev["t"],
            "epochs": ev.get("epochs"),
            "batches_per_epoch": ev.get("batches_per_epoch"),
            "rows": [],
            "progress": None,
            "result": None,
        }
    elif kind == "batch" and ev.get("run") in state["runs"]:
        state["runs"][ev["run"]]["progress"] = {"epoch": ev.get("epoch"), "batch": ev.get("batch"), "of": ev.get("of"), "t": ev["t"]}
    elif kind == "epoch" and ev.get("run") in state["runs"]:
        state["runs"][ev["run"]]["rows"].append({k: v for k, v in ev.items() if k not in ("type", "run")})
    elif kind == "run_end" and ev.get("run") in state["runs"]:
        run = state["runs"][ev["run"]]
        run["status"] = "finished"
        run["ended"] = ev["t"]
        run["result"] = ev.get("result")
    elif kind == "selection":
        state["selection"] = {k: v for k, v in ev.items() if k not in ("type", "t")}
    elif kind == "test_result":
        state["test"] = {k: v for k, v in ev.items() if k not in ("type", "t")}
    elif kind == "done":
        state["stage"] = "done"
        state["stage_message"] = "Training and evaluation finished"
    elif kind == "gpu_memory":
        state.setdefault("gpu_memory", []).append({k: ev.get(k) for k in ("context", "allocated_gb", "reserved_gb")})
    elif kind == "failed":
        state["error"] = {"message": ev.get("error"), "traceback": ev.get("traceback")}
        state["stage"] = "failed"
        state["stage_message"] = ev.get("error")


def write_state(state: dict) -> None:
    state["monitor_updated"] = time.time()
    LIVE_JS.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_JS.with_suffix(".tmp")
    tmp.write_text("window.__LIVE__ = " + json.dumps(state) + ";\n")
    tmp.replace(LIVE_JS)


def clean_log_line(line: str) -> str:
    """Make a raw log line readable: fix its encoding and remove terminal control codes."""
    try:
        # The stream arrives as UTF-8 that the HTTP client decoded as Latin-1.
        line = line.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return ANSI.sub("", EVENT_FRAGMENT.sub("", line)).rstrip()


class LogFollower(threading.Thread):
    """Streams the kernel log. The server replays from the start on every reconnect, so
    each connection rebuilds the event state from scratch rather than deduplicating."""

    def __init__(self, api, kernel: str, state: dict, lock: threading.Lock, base: dict | None = None):
        super().__init__(daemon=True)
        self.api, self.kernel, self.state, self.lock = api, kernel, state, lock
        self.base = base or new_state(kernel)
        self.stop_requested = threading.Event()
        self.finished_stream = threading.Event()
        self.published_chunks = 0  # high-water mark, so a replay never shows older state

    def ingest(self, lines: list[str], fresh: dict, buffer: list[str]) -> None:
        for line in lines:
            if EVENT_PREFIX in line:
                payload = line.split(EVENT_PREFIX, 1)[1].strip()
                try:
                    apply_event(fresh, json.loads(payload))
                except (json.JSONDecodeError, KeyError):
                    buffer.append(line)
            else:
                text = clean_log_line(line)
                if text and not PROGRESS_BAR.search(text):
                    buffer.append(text)

    def run(self) -> None:
        while not self.stop_requested.is_set():
            fresh = copy.deepcopy(self.base)
            buffer: list[str] = []
            chunks = 0
            try:
                for event in self.api.kernels_logs_stream(self.kernel):
                    if self.stop_requested.is_set():
                        return
                    data = event.get("data")
                    if not data:
                        continue
                    self.ingest(data.splitlines(), fresh, buffer)
                    chunks += 1
                    if chunks >= self.published_chunks:
                        self.published_chunks = chunks
                        self.publish(fresh, buffer)
                self.publish(fresh, buffer)
                self.finished_stream.set()
                return
            except Exception as exc:  # network drops are routine on a long stream
                print(f"[monitor] log stream interrupted ({type(exc).__name__}); reconnecting in 10s", flush=True)
                time.sleep(10)

    def publish(self, fresh: dict, buffer: list[str]) -> None:
        with self.lock:
            for key, value in fresh.items():
                if key not in ("status", "failure", "kernel", "url", "monitor_updated"):
                    self.state[key] = value
            self.state["log_tail"] = buffer[-LOG_TAIL:]


def with_timeout(fn, seconds: float):
    """Run a Kaggle API call with a deadline. The client sets none, so a connection lost
    while the computer sleeps could otherwise block the monitor indefinitely."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"no response from Kaggle within {seconds}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def history_of(api, base_kernel: str, kernel: str) -> dict:
    """State from an earlier kernel's stored log, for a follow-up kernel to build on."""
    state = new_state(kernel)
    state["base_kernel"] = base_kernel
    buffer: list[str] = []
    follower = LogFollower(api, base_kernel, state, threading.Lock())
    for event in api.kernels_logs_stream(base_kernel):
        if event.get("data"):
            follower.ingest(event["data"].splitlines(), state, buffer)
    # Keep the earlier failure on record, but it no longer describes the current state.
    state["recovered_error"] = (state.pop("error", None) or {}).get("message")
    state["error"] = None
    print(f"[monitor] loaded history of {base_kernel}: {sum(len(r['rows']) for r in state['runs'].values())} epochs", flush=True)
    return state


def download_outputs(api, kernel: str) -> Path:
    slug = kernel.split("/")[1]
    target = RAW_DIR / slug
    target.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel, str(target), force=True, quiet=True)
    return target


def publish_results(artifacts: Path) -> None:
    """Keep the final model and provenance files, evaluate, and rebuild the reports."""
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    shutil.copy2(artifacts / "best.pt", ROOT / "models" / "yolo26_blood_cells.pt")
    for name in KEPT_ARTIFACTS:
        shutil.copy2(artifacts / name, ROOT / "results" / name)
    shutil.copytree(artifacts / "runs", ROOT / "results" / "runs", dirs_exist_ok=True)
    python = sys.executable
    subprocess.run([python, "scripts/evaluate.py", "--artifacts", str(artifacts)], cwd=ROOT, check=True)
    subprocess.run([python, "scripts/build_reports.py"], cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", help="owner/slug; defaults to the last pushed kernel")
    ap.add_argument("--base-kernel", help="earlier kernel whose training history to keep")
    ap.add_argument("--interval", type=int, default=30, help="seconds between status checks")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN missing from .env")
    from kaggle.api.kaggle_api_extended import KaggleApi

    kernel = args.kernel or json.loads((ROOT / "build" / "last_push.json").read_text())["kernel"]
    api = KaggleApi()
    api.authenticate()

    base = history_of(api, args.base_kernel, kernel) if args.base_kernel else None
    state = copy.deepcopy(base) if base else new_state(kernel)
    lock = threading.Lock()
    follower = LogFollower(api, kernel, state, lock, base)
    follower.start()
    print(f"[monitor] following {kernel}; writing {LIVE_JS.relative_to(ROOT)}", flush=True)

    last_status = None
    status = None
    next_status_check = 0.0
    while True:
        # Dashboard data is rewritten often, while Kaggle's status API is polled gently.
        if time.time() >= next_status_check:
            next_status_check = time.time() + args.interval
            try:
                response = with_timeout(lambda: api.kernels_status(kernel), STATUS_TIMEOUT_S)
                # The API returns an enum such as KernelWorkerStatus.RUNNING.
                status = str(getattr(response, "status", response)).split(".")[-1].lower()
                failure = getattr(response, "failure_message", None) or None
                with lock:
                    state["status"] = status
                    state["failure"] = failure
                if status != last_status:
                    print(f"[monitor] status: {status}" + (f" ({failure})" if failure else ""), flush=True)
                    last_status = status
            except Exception as exc:
                print(f"[monitor] status check failed: {exc}", flush=True)

        with lock:
            write_state(state)

        if status in TERMINAL:
            # The live stream may have been cut off or stuck, so finish from the stored log,
            # which always holds the complete history of the run.
            follower.stop_requested.set()
            final_read = LogFollower(api, kernel, state, lock, base)
            final_read.start()
            final_read.finished_stream.wait(timeout=FINAL_READ_TIMEOUT_S)
            final_read.stop_requested.set()
            with lock:
                write_state(state)
            if status == "complete" and not args.no_download:
                target = download_outputs(api, kernel)
                print(f"[monitor] outputs downloaded to {target.relative_to(ROOT)}", flush=True)
                if kernel.endswith("-smoke"):
                    print("[monitor] smoke run: the published model and results are left unchanged", flush=True)
                else:
                    publish_results(target / "artifacts")
                    print("[monitor] evaluation finished and reports rebuilt", flush=True)
            print(f"[monitor] finished with status {status}", flush=True)
            return 0 if status == "complete" else 1
        time.sleep(WRITE_EVERY_S)


if __name__ == "__main__":
    sys.exit(main())
