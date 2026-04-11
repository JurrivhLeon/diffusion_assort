#!/usr/bin/env python3
"""Launch MCCM-EM-on-MMNL workers in detached sessions."""

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
                "run_mccm_em_on_mmnl_batch.py",
                "--device",
                "cuda:0",
                "--out-root",
                "outputs_mmnl",
                "--shard-index",
                "0",
                "--shard-count",
                "3",
            ],
            LOG_DIR / "mccm_em_mmnl_shard0.log",
        ),
        (
            [
                sys.executable,
                "run_mccm_em_on_mmnl_batch.py",
                "--device",
                "cuda:1",
                "--out-root",
                "outputs_mmnl",
                "--shard-index",
                "1",
                "--shard-count",
                "3",
            ],
            LOG_DIR / "mccm_em_mmnl_shard1.log",
        ),
        (
            [
                sys.executable,
                "run_mccm_em_on_mmnl_batch.py",
                "--device",
                "cuda:2",
                "--out-root",
                "outputs_mmnl",
                "--shard-index",
                "2",
                "--shard-count",
                "3",
            ],
            LOG_DIR / "mccm_em_mmnl_shard2.log",
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
