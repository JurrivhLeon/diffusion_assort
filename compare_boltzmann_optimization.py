#!/usr/bin/env python3
"""Compare exact and fitted-model assortment optimization on one dataset.

The experiment runs three solves:
1. Exhaustive search under the true MNL policy from the dataset metadata.
2. MIP optimization under the fitted GAsN checkpoint.
3. MIP optimization under the fitted RAsN checkpoint.

The output is a single JSON report with the solution, objective value, runtime,
and the saved training configuration for each fitted model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

from assortment_dataset import load_assortment_json
from evaluate_assortment import build_reward_fn, expected_reward
from exhaustive_search_assortment import build_policy_kind, iter_assortments, count_assortments
from mip_optimize_assortment import _solve_mip_backend


def assortment_to_string(assortment: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in assortment)


def run_true_reference(
    data_path: Path,
    reward_module: Path | None,
) -> dict:
    raw = load_assortment_json(data_path)
    meta = raw["meta"]
    n_products = int(meta["N"])
    reward_fn = build_reward_fn(str(reward_module) if reward_module else None)
    policy_fn = build_policy_kind(meta, "true_mnl", None)

    total = count_assortments(n_products)
    best_S: list[int] | None = None
    best_value = float("-inf")
    num_scored = 0

    start = time.monotonic()
    for S in iter_assortments(n_products):
        probs = policy_fn(S)
        value = float(sum(reward_fn(S, a, meta) * probs[a] for a in range(len(probs))))
        num_scored += 1
        if value > best_value:
            best_value = value
            best_S = list(S)
    elapsed = time.monotonic() - start

    if best_S is None:
        raise RuntimeError("No assortments were scored.")

    return {
        "policy": "true_mnl",
        "backend": "exhaustive",
        "best_assortment": best_S,
        "best_assortment_str": assortment_to_string(best_S),
        "best_value": best_value,
        "num_scored": num_scored,
        "total": total,
        "runtime_sec": elapsed,
        "reward_module": str(reward_module) if reward_module else None,
    }


def load_run_config(checkpoint: Path) -> dict | None:
    config_path = checkpoint.parent / "run_config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_fitted_mip(
    data_path: Path,
    checkpoint: Path,
    reward_module: Path | None,
    time_limit: float,
    exp_pieces: int,
    max_iterations: int,
    dinkelbach_tol: float,
    solver_verbose: bool,
) -> dict:
    raw = load_assortment_json(data_path)
    meta = raw["meta"]
    reward_fn = build_reward_fn(str(reward_module) if reward_module else None)
    start = time.monotonic()
    result = _solve_mip_backend(
        meta=meta,
        checkpoint_path=checkpoint,
        policy_kind="model",
        max_cardinality=None,
        exact_cardinality=None,
        exp_pieces=exp_pieces,
        time_limit=time_limit,
        max_iterations=max_iterations,
        tol=dinkelbach_tol,
        solver_verbose=solver_verbose,
    )
    result["runtime_sec"] = time.monotonic() - start
    result["checkpoint"] = str(checkpoint)
    result["run_config"] = load_run_config(checkpoint)
    result["best_assortment_str"] = assortment_to_string(result["best_assortment"])
    result["model_objective_value"] = float(result["best_value"])
    result["true_expected_reward"] = float(
        expected_reward(
            result["best_assortment"],
            meta,
            reward_fn,
            "true_mnl",
            None,
        )
    )
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run true exhaustive search and fitted-model MIP comparisons."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSON path.")
    parser.add_argument(
        "--gasn-checkpoint",
        type=Path,
        required=True,
        help="Path to the fitted GAsN checkpoint.",
    )
    parser.add_argument(
        "--rasn-checkpoint",
        type=Path,
        required=True,
        help="Path to the fitted RAsN checkpoint.",
    )
    parser.add_argument(
        "--reward-module",
        type=Path,
        default=None,
        help="Optional reward module used only for the exhaustive true-policy reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="Per-model MIP time limit in seconds.",
    )
    parser.add_argument(
        "--exp-pieces",
        type=int,
        default=8,
        help="Number of segments for the exp approximation in the MIP.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum Dinkelbach iterations.",
    )
    parser.add_argument(
        "--dinkelbach-tol",
        type=float,
        default=1e-6,
        help="Stopping tolerance for Dinkelbach iterations.",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Show solver logs during MIP solves.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    reference = run_true_reference(args.data, args.reward_module)
    gasn = run_fitted_mip(
        args.data,
        args.gasn_checkpoint,
        args.reward_module,
        time_limit=args.time_limit,
        exp_pieces=args.exp_pieces,
        max_iterations=args.max_iterations,
        dinkelbach_tol=args.dinkelbach_tol,
        solver_verbose=args.solver_verbose,
    )
    rasn = run_fitted_mip(
        args.data,
        args.rasn_checkpoint,
        args.reward_module,
        time_limit=args.time_limit,
        exp_pieces=args.exp_pieces,
        max_iterations=args.max_iterations,
        dinkelbach_tol=args.dinkelbach_tol,
        solver_verbose=args.solver_verbose,
    )

    report = {
        "data": str(args.data),
        "reference": reference,
        "mip_models": {
            "gasn": gasn,
            "rasn": rasn,
        },
        "settings": {
            "time_limit": args.time_limit,
            "exp_pieces": args.exp_pieces,
            "max_iterations": args.max_iterations,
            "dinkelbach_tol": args.dinkelbach_tol,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
