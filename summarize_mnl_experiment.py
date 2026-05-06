#!/usr/bin/env python3
"""Summarize one synthetic MNL experiment run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from evaluate_assortment import load_assortment_json


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _as_sample_string(sample: list[int]) -> str:
    return "".join(str(int(v)) for v in sample)


def _training_summary(path: Path) -> dict[str, Any]:
    metrics = _load_json(path / "metrics.json")
    history = metrics["history"]
    best_key = "val_nll" if "val_nll" in history[0] else "val_loss"
    best = min(history, key=lambda row: row.get(best_key, float("inf")))
    final = history[-1]
    final_key = "val_nll" if "val_nll" in final else "val_loss"
    return {
        "best_epoch": metrics.get("best_epoch", best["epoch"]),
        "best_metric": metrics.get("best_val_nll", metrics.get("best_val_loss", best[best_key])),
        "final_metric": final[final_key],
        "final_accuracy": final.get("val_accuracy", final.get("val_bit_accuracy")),
        "run_config": _load_json(path / "run_config.json"),
    }


def _ratio_quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "q50": float("nan"), "q90": float("nan"), "max": float("nan")}

    sorted_values = sorted(values)

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


def _optimal_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    return {
        "best_assortment": _as_sample_string(data["best_assortment"]),
        "best_value": float(data["best_value"]),
        "num_selected": int(data["num_selected"]),
        "max_cardinality": data.get("max_cardinality"),
    }


def _guided_summary(path: Path, optimal_reward: float, reference_solution: str) -> dict[str, Any]:
    data = _load_json(path)
    true_rewards = [float(x) for x in data["true_expected_rewards"]]
    guided_scores = [float(x) for x in data["guided_scores"]]
    samples = [_as_sample_string(sample) for sample in data["samples"]]
    ratios = [reward / optimal_reward for reward in true_rewards]
    best_guided_idx = max(range(len(guided_scores)), key=guided_scores.__getitem__)
    best_true_idx = max(range(len(true_rewards)), key=true_rewards.__getitem__)
    return {
        "num_samples": int(data["num_samples"]),
        "guidance_scale": float(data["guidance_scale"]),
        "guidance_gamma": float(data["guidance_gamma"]),
        "runtime_sec": data.get("runtime_sec"),
        "true_reward_mean": float(statistics.fmean(true_rewards)),
        "true_reward_std": float(statistics.pstdev(true_rewards)),
        "best_true_sample": samples[best_true_idx],
        "best_true_reward": true_rewards[best_true_idx],
        "best_guided_score": guided_scores[best_guided_idx],
        "best_guided_sample": samples[best_guided_idx],
        "best_guided_true_expected_reward": true_rewards[best_guided_idx],
        "contains_reference_solution": reference_solution in samples,
        "count_reference_solution": sum(1 for sample in samples if sample == reference_solution),
        "optimal_ratio": _ratio_quantiles(ratios),
    }


def build_report(
    data_path: Path,
    choice_root: Path,
    diffusion_root: Path,
    optimal_path: Path,
    guided_gasn_path: Path,
    guided_rasn_path: Path,
) -> dict[str, Any]:
    dataset = load_assortment_json(data_path)
    meta = dataset["meta"]
    optimal = _optimal_summary(optimal_path)
    optimal_reward = optimal["best_value"]
    return {
        "dataset": {
            "path": str(data_path),
            "N": int(meta["N"]),
            "n_samples": int(meta["n_samples"]),
            "choice_model": meta["choice_model"],
            "assortment_policy": meta["assortment_policy"],
            "boltzmann_sampler": meta["assortment_policy"].get("boltzmann_sampler"),
            "boltzmann_beta": meta["assortment_policy"].get("boltzmann_beta"),
        },
        "training": {
            "gasn": _training_summary(choice_root / "gasn"),
            "rasn": _training_summary(choice_root / "rasn"),
            "diffusion": _training_summary(diffusion_root),
        },
        "optimal": optimal,
        "guided_sampling": {
            "gasn": _guided_summary(
                guided_gasn_path,
                optimal_reward,
                optimal["best_assortment"],
            ),
            "rasn": _guided_summary(
                guided_rasn_path,
                optimal_reward,
                optimal["best_assortment"],
            ),
        },
        "notes": [
            "Optimal ratios are computed against the true MNL optimum from dataset metadata.",
            "Guided sampling is evaluated with true expected reward under the generating MNL model.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--choice-root", type=Path, required=True)
    parser.add_argument("--diffusion-root", type=Path, required=True)
    parser.add_argument("--optimal", type=Path, required=True)
    parser.add_argument("--guided-gasn", type=Path, required=True)
    parser.add_argument("--guided-rasn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        args.data,
        args.choice_root,
        args.diffusion_root,
        args.optimal,
        args.guided_gasn,
        args.guided_rasn,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
