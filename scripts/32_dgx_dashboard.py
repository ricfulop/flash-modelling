#!/usr/bin/env python3
"""Live terminal dashboard for the local DGX and dgx-peer."""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import shlex
import subprocess
import sys
import time


HOSTS = [
    ("spark-5da5", None),
    ("dgx-peer", "dgx-peer"),
]


def run(command: list[str], timeout: int = 8) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except subprocess.CalledProcessError as exc:
        return (exc.output or f"command failed rc={exc.returncode}").strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def host_run(ssh_host: str | None, script: str, timeout: int = 8) -> str:
    clean = ["env", "-i", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME=" + os.environ.get("HOME", "")]
    if ssh_host is None:
        return run([*clean, "bash", "--noprofile", "--norc", "-lc", script], timeout=timeout)
    remote = (
        "env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
        f"HOME=/home/nvidia bash --noprofile --norc -lc {shlex.quote(script)}"
    )
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", ssh_host, remote], timeout=timeout)


def cpu_percent(ssh_host: str | None) -> str:
    script = "ps -eo pcpu= | awk '{s+=$1} END {printf \"%.1f\", s/20}'"
    return host_run(ssh_host, script, timeout=6)


def loadavg(ssh_host: str | None) -> str:
    return host_run(ssh_host, "awk '{print $1,$2,$3}' /proc/loadavg", timeout=6)


def memory(ssh_host: str | None) -> str:
    return host_run(ssh_host, "free -h | awk '/Mem:/ {print $3 \"/\" $2 \" used, avail \" $7}'", timeout=6)


def gpu(ssh_host: str | None) -> str:
    script = "nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true"
    text = host_run(ssh_host, script, timeout=6)
    return text if text else "nvidia-smi unavailable"


def process_lines(ssh_host: str | None, kind: str) -> list[str]:
    if kind == "bigdft":
        script = "ps -C bigdft -o pid=,pcpu=,pmem=,etime= --sort=-pcpu 2>/dev/null | head -n 8"
    else:
        script = (
            "ps -eo pid=,pcpu=,pmem=,etime=,args= --sort=-pcpu | "
            "grep -E 'python .*14_large_lattice_kpm_showcase|python .*19_phase2_structure_transport|python .*20_phase2_wannier_transport|python .*04_transport' | "
            "grep -v grep | head -n 5"
        )
    text = host_run(ssh_host, script, timeout=6)
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def render_host(name: str, ssh_host: str | None, width: int) -> None:
    print(f"[{name}]")
    print(f"load: {strip_ansi(loadavg(ssh_host))}    cpu: {strip_ansi(cpu_percent(ssh_host))}% of 20-core machine")
    print(f"mem:  {strip_ansi(memory(ssh_host))}")
    print(f"gpu:  {strip_ansi(gpu(ssh_host))}")
    bigdft = process_lines(ssh_host, "bigdft")
    if bigdft:
        print("top BigDFT:")
        for line in bigdft:
            print(f"  {strip_ansi(line)}")
    pygpu = process_lines(ssh_host, "pygpu")
    if pygpu:
        print("top Python/GPU jobs:")
        for line in pygpu:
            clean = strip_ansi(line)
            print(f"  {clean[:width - 4]}")
    print("-" * width)


def main() -> int:
    interval = float(os.environ.get("DGX_DASHBOARD_INTERVAL", "10"))
    while True:
        width = shutil.get_terminal_size((120, 40)).columns
        os.system("clear")
        print("DGX_DASHBOARD live monitor".center(width))
        print(dt.datetime.now().isoformat(timespec="seconds").center(width))
        print("=" * width)
        for name, ssh_host in HOSTS:
            render_host(name, ssh_host, width)
        sys.stdout.flush()
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
