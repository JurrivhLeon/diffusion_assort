#!/usr/bin/env python3
"""Aggregate per-run MNL reports across the full experiment suite."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "q50": float("nan"), "q90": float("nan"), "max": float("nan")}

    sorted_values = sorted(float(v) for v in values)

    def quantile(q: float) -> float:
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        pos = (len(sorted_values) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_values[lo])
        weight = pos - lo
        return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)

    return {
        "mean": float(statistics.fmean(sorted_values)),
        "q50": quantile(0.5),
        "q90": quantile(0.9),
        "max": float(max(sorted_values)),
    }


def _extract_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = report["dataset"]
    n_products = int(dataset["N"])
    beta = float(dataset["boltzmann_beta"])
    rows: list[dict[str, Any]] = []
    for model_name in ("gasn", "rasn"):
        guided = report["guided_sampling"][model_name]
        ratio = guided["optimal_ratio"]
        rows.append(
            {
                "N": n_products,
                "beta": beta,
                "model": model_name,
                "run_best_optimal_ratio": float(ratio["max"]),
                "run_mean_optimal_ratio": float(ratio["mean"]),
                "run_q50_optimal_ratio": float(ratio["q50"]),
                "run_q90_optimal_ratio": float(ratio["q90"]),
                "best_guided_true_expected_reward": float(guided["best_guided_true_expected_reward"]),
                "optimal_reward": float(report["optimal"]["best_value"]),
            }
        )
    return rows


def build_summary(root: Path) -> dict[str, Any]:
    report_paths = sorted(root.glob("N*_beta*_dataset*/report.json"))
    if not report_paths:
        raise FileNotFoundError(f"No report.json files found under {root}.")

    rows: list[dict[str, Any]] = []
    for path in report_paths:
        rows.extend(_extract_rows(_load_json(path)))

    grouped: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_nb: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["N"], row["beta"], row["model"])].append(row)
        by_model[row["model"]].append(row)
        by_nb[(row["N"], row["beta"])].append(row)

    per_group = []
    for (n_products, beta, model_name), items in sorted(grouped.items()):
        per_group.append(
            {
                "N": n_products,
                "beta": beta,
                "model": model_name,
                "num_runs": len(items),
                "run_best_optimal_ratio": _summarize([x["run_best_optimal_ratio"] for x in items]),
                "run_mean_optimal_ratio": _summarize([x["run_mean_optimal_ratio"] for x in items]),
                "run_q50_optimal_ratio": _summarize([x["run_q50_optimal_ratio"] for x in items]),
                "run_q90_optimal_ratio": _summarize([x["run_q90_optimal_ratio"] for x in items]),
            }
        )

    per_model = []
    for model_name, items in sorted(by_model.items()):
        per_model.append(
            {
                "model": model_name,
                "num_runs": len(items),
                "run_best_optimal_ratio": _summarize([x["run_best_optimal_ratio"] for x in items]),
                "run_mean_optimal_ratio": _summarize([x["run_mean_optimal_ratio"] for x in items]),
                "run_q50_optimal_ratio": _summarize([x["run_q50_optimal_ratio"] for x in items]),
                "run_q90_optimal_ratio": _summarize([x["run_q90_optimal_ratio"] for x in items]),
            }
        )

    per_dataset_group = []
    for (n_products, beta), items in sorted(by_nb.items()):
        per_dataset_group.append(
            {
                "N": n_products,
                "beta": beta,
                "num_runs": len(items),
                "run_best_optimal_ratio": _summarize([x["run_best_optimal_ratio"] for x in items]),
                "run_mean_optimal_ratio": _summarize([x["run_mean_optimal_ratio"] for x in items]),
                "run_q50_optimal_ratio": _summarize([x["run_q50_optimal_ratio"] for x in items]),
                "run_q90_optimal_ratio": _summarize([x["run_q90_optimal_ratio"] for x in items]),
            }
        )

    return {
        "root": str(root),
        "num_reports": len(report_paths),
        "num_guided_runs": len(rows),
        "per_group": per_group,
        "per_model": per_model,
        "per_dataset_group": per_dataset_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing per-run report.json files.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the aggregated JSON report.")
    args = parser.parse_args()

    summary = build_summary(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
