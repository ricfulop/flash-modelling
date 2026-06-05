#!/usr/bin/env python3
"""Sunday watchdog: refresh synthesis when compute queues finish.

The watchdog is intentionally simple and file-based. It retries peer import,
combines any completed BigDFT follow-up queues, and writes a refreshed synthesis
package on each loop.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
LOG = RUNS / "sunday_watchdog.log"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(f"{now()} {message}\n")


def read_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def latest_followups() -> list[Path]:
    roots = []
    for status_path in sorted(RUNS.glob("bigdft_followup_*/status.json")):
        status = read_status(status_path)
        if status and status.get("state") == "completed":
            roots.append(status_path.parent)
    return roots


def active_followups() -> list[Path]:
    roots = []
    for status_path in sorted(RUNS.glob("bigdft_followup_*/status.json")):
        status = read_status(status_path)
        if status and status.get("state") == "running":
            roots.append(status_path.parent)
    return roots


def peer_available(host: str = "dgx-peer") -> bool:
    """Return whether the peer alias resolves before attempting rsync/ssh imports."""
    try:
        socket.getaddrinfo(host, None)
        return True
    except OSError:
        return False


def run_synthesis(include_peer: bool) -> int:
    cmd = [
        "python",
        str(REPO / "scripts" / "10_sunday_synthesis.py"),
    ]
    for root in latest_followups():
        cmd.extend(["--extra-bigdft", str(root)])
    if include_peer and not peer_available():
        log("peer import requested but dgx-peer does not resolve; running local-only synthesis")
        include_peer = False
    if not include_peer:
        cmd.extend(["--peer-bigdft-remote", ""])
    log("running synthesis: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(proc.stdout.strip() or f"synthesis exited rc={proc.returncode}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-s", type=int, default=900)
    parser.add_argument("--max-loops", type=int, default=0, help="0 means loop forever.")
    parser.add_argument("--include-peer", action="store_true", help="Retry peer import on each loop.")
    args = parser.parse_args()

    log("watchdog start")
    completed_once = False
    loops = 0
    while True:
        loops += 1
        active = active_followups()
        done = latest_followups()
        log(f"status active_followups={len(active)} completed_followups={len(done)}")
        if done and not completed_once:
            run_synthesis(include_peer=args.include_peer)
            completed_once = True
        elif not active:
            run_synthesis(include_peer=args.include_peer)
        if args.max_loops and loops >= args.max_loops:
            log("watchdog stop max-loops")
            return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
