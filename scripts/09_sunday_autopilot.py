#!/usr/bin/env python3
"""Sunday autopilot: periodic aggregation and status snapshots.

This does not own the compute jobs. It watches the active KPM sweep directories,
periodically imports the peer run, runs the aggregator, and writes a compact
status file so the project keeps producing Sunday-ready artifacts without
manual checking.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def latest(pattern: str, base: Path) -> Path | None:
    paths = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def read_status(path: Path | None) -> dict:
    if path is None:
        return {"state": "missing"}
    status_path = path / "status.json"
    if not status_path.exists():
        return {"state": "missing", "run_root": str(path)}
    return json.loads(status_path.read_text())


def run_aggregate(local_run: Path, peer_remote: str, aggregate_root: Path, iteration: int) -> Path:
    out = aggregate_root / f"aggregate_{iteration:04d}"
    cmd = [
        "python3",
        str(REPO / "scripts" / "07_aggregate_kpm_sweeps.py"),
        "--run",
        str(local_run),
        "--remote",
        peer_remote,
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-run", default=None)
    parser.add_argument("--peer-remote", default=None)
    parser.add_argument("--interval-s", type=int, default=900)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means loop forever")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    local_run = Path(args.local_run) if args.local_run else latest("gpu_kpm_*_local", RUNS)
    peer_remote = args.peer_remote
    if peer_remote is None:
        peer_remote = "dgx-peer:/home/nvidia/Cursor/flash-modelling/runs/gpu_kpm_20260603_084631_peer"
    if local_run is None:
        raise SystemExit("No local gpu_kpm run found")

    run_id = datetime.now().strftime("sunday_autopilot_%Y%m%d_%H%M%S")
    root = RUNS / run_id
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "heartbeat.log"
    aggregate_root = root / "aggregates"
    iteration = 0

    while True:
        started = now()
        with heartbeat.open("a") as fh:
            fh.write(f"{started} iteration {iteration} start\n")
        aggregate_path = None
        error = None
        try:
            aggregate_path = run_aggregate(local_run, peer_remote, aggregate_root, iteration)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        local_status = read_status(local_run)
        peer_import = aggregate_path / "imports" / "dgx-peer" / Path(peer_remote.split(":", 1)[1]).name if aggregate_path else None
        peer_status = read_status(peer_import)
        payload = {
            "state": "running",
            "updated_at": now(),
            "iteration": iteration,
            "local_run": str(local_run),
            "peer_remote": peer_remote,
            "latest_aggregate": str(aggregate_path) if aggregate_path else None,
            "error": error,
            "local_completed_cases": local_status.get("completed_cases", len(local_status.get("cases", []))),
            "local_state": local_status.get("state"),
            "peer_completed_cases": peer_status.get("completed_cases", len(peer_status.get("cases", []))),
            "peer_state": peer_status.get("state"),
        }
        write_json(root / "status.json", payload)
        with heartbeat.open("a") as fh:
            fh.write(
                f"{now()} iteration {iteration} aggregate={aggregate_path} "
                f"local={payload['local_completed_cases']} peer={payload['peer_completed_cases']} error={error}\n"
            )
        print(
            "SUNDAY_AUTOPILOT "
            + json.dumps(
                {
                    "iteration": iteration,
                    "aggregate": str(aggregate_path) if aggregate_path else None,
                    "local_completed": payload["local_completed_cases"],
                    "peer_completed": payload["peer_completed_cases"],
                    "error": error,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        iteration += 1
        if args.once or (args.max_iterations and iteration >= args.max_iterations):
            payload["state"] = "completed"
            payload["updated_at"] = now()
            write_json(root / "status.json", payload)
            return 0 if error is None else 1
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
