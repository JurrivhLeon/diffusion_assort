#!/usr/bin/env python3
"""Launch MMNL batch workers in detached sessions.

This starts the three MMNL batch workers used for the canonical sweep:
  - GPU 0: N=100
  - GPU 1: N=80 and N=20
  - GPU 2: N=60 and N=40

Each worker reuses existing outputs unless a report is missing, so completed
cases are not recomputed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path("/hpc/home/jl1344/diffusion_assort")
LOG_DIR = ROOT / "logs"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            [
                sys.executable,
                "run_mmnl_batch.py",
                "--device",
                "cuda:0",
                "--out-root",
                "outputs_mmnl",
                "--n",
                "100",
            ],
            LOG_DIR / "mmnl_gpu0_n100.log",
        ),
        (
            [
                sys.executable,
                "run_mmnl_batch.py",
                "--device",
                "cuda:1",
                "--out-root",
                "outputs_mmnl",
                "--n",
                "80",
                "--n",
                "20",
            ],
            LOG_DIR / "mmnl_gpu1_n80_n20.log",
        ),
        (
            [
                sys.executable,
                "run_mmnl_batch.py",
                "--device",
                "cuda:2",
                "--out-root",
                "outputs_mmnl",
                "--n",
                "60",
                "--n",
                "40",
            ],
            LOG_DIR / "mmnl_gpu2_n60_n40.log",
        ),
    ]

    children = []
    for cmd, log_path in jobs:
        log_file = log_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((proc.pid, log_path))

    for pid, log_path in children:
        print(f"started pid={pid} log={log_path}")


if __name__ == "__main__":
    main()
