#!/usr/bin/env python3
"""Wait for an old Phase 2 BigDFT label queue, then launch matrix-export labels."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def heartbeat(run_root: Path, message: str) -> None:
    with (run_root / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def discover_matrix_artifacts(root: Path) -> list[str]:
    patterns = [
        "**/*_hr.dat",
        "**/*hr.dat",
        "**/*Hamiltonian*.mtx",
        "**/*hamiltonian*.mtx",
        "**/*ham*.mtx",
    ]
    artifacts: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                artifacts.append(path)
                seen.add(path)
    return [str(path.relative_to(root)) for path in sorted(artifacts)]


def run_command(command: list[str], cwd: Path, log_path: Path) -> tuple[int, float]:
    start = time.time()
    with log_path.open("a") as log:
        log.write("$ " + " ".join(str(x) for x in command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, time.time() - start


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-status", required=True, type=Path)
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--max-wait-seconds", type=int, default=0, help="0 means wait indefinitely")
    p.add_argument("--label", default=os.environ.get("ELECTRODEFECT_MATRIX_FOLLOWUP_LABEL", socket.gethostname()))
    p.add_argument("--max-seconds", type=int, default=14400)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--run-wannier-scalar-smoke", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    old_status_path = args.old_status.resolve()
    run_root = RUNS / datetime.now().strftime(f"phase2_matrix_followup_%Y%m%d_%H%M%S_{args.label}")
    run_root.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "waiting_old_queue",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "old_status": str(old_status_path),
        "poll_seconds": args.poll_seconds,
        "max_wait_seconds": args.max_wait_seconds,
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "matrix follow-up watcher start")

    start_wait = time.time()
    terminal_states = {"completed", "completed_with_failures", "failed", "prepared", "time_budget_reached"}
    while True:
        old = read_json(old_status_path, {})
        old_state = old.get("state")
        status.update({"old_state": old_state, "updated_at": now()})
        write_json(run_root / "status.json", status)
        if old_state in terminal_states:
            break
        if args.max_wait_seconds and (time.time() - start_wait) > args.max_wait_seconds:
            status["state"] = "timed_out_waiting_old_queue"
            write_json(run_root / "status.json", status)
            heartbeat(run_root, "timed out waiting for old queue")
            return 1
        time.sleep(max(5, int(args.poll_seconds)))

    old_root = old_status_path.parent
    old_artifacts = discover_matrix_artifacts(old_root)
    status.update(
        {
            "state": "old_queue_finished",
            "old_state": old_state,
            "old_matrix_artifacts": old_artifacts,
            "old_matrix_artifact_count": len(old_artifacts),
            "updated_at": now(),
        }
    )
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"old queue finished state={old_state} matrix_artifacts={len(old_artifacts)}")

    roots = old.get("roots", [])
    command = [
        sys.executable,
        "scripts/18_phase2_bigdft_labels.py",
        "--label",
        f"matrix_after_{args.label}",
        "--max-seconds",
        str(args.max_seconds),
    ]
    for root in roots:
        command.extend(["--root", root])
    if args.max_jobs:
        command.extend(["--max-jobs", str(args.max_jobs)])
    status.update({"state": "running_matrix_queue", "matrix_command": command, "updated_at": now()})
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "launch matrix-export BigDFT label queue")
    rc, elapsed = run_command(command, REPO, run_root / "matrix_queue.out")
    status.update({"matrix_queue_returncode": rc, "matrix_queue_elapsed_s": elapsed, "updated_at": now()})
    if rc != 0:
        status["state"] = "matrix_queue_failed"
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"matrix queue failed rc={rc}")
        return rc

    # The child queue prints its run root as the last non-empty line.
    lines = (run_root / "matrix_queue.out").read_text(errors="replace").splitlines()
    matrix_roots = [line.strip() for line in lines if line.strip().startswith(str(RUNS))]
    matrix_root = Path(matrix_roots[-1]) if matrix_roots else None
    artifacts = discover_matrix_artifacts(matrix_root) if matrix_root else []
    status.update(
        {
            "state": "matrix_queue_completed",
            "matrix_run_root": str(matrix_root) if matrix_root else None,
            "matrix_artifact_count": len(artifacts),
            "matrix_artifacts": artifacts,
            "updated_at": now(),
        }
    )
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"matrix queue completed artifacts={len(artifacts)}")

    if args.run_wannier_scalar_smoke and matrix_root and artifacts:
        smoke_command = [
            sys.executable,
            "scripts/20_phase2_wannier_transport.py",
            "--source-root",
            str(matrix_root),
            "--device",
            "scipy",
            "--label",
            f"scalar_after_{args.label}",
        ]
        status.update({"state": "running_wannier_scalar_smoke", "smoke_command": smoke_command, "updated_at": now()})
        write_json(run_root / "status.json", status)
        rc, elapsed = run_command(smoke_command, REPO, run_root / "wannier_scalar_smoke.out")
        status.update(
            {
                "wannier_scalar_smoke_returncode": rc,
                "wannier_scalar_smoke_elapsed_s": elapsed,
                "state": "completed" if rc == 0 else "wannier_scalar_smoke_failed",
                "updated_at": now(),
            }
        )
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"wannier scalar smoke rc={rc}")
        return rc

    status["state"] = "completed"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "matrix follow-up complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
