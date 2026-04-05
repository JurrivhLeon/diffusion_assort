#!/usr/bin/env python3
"""Summarize an MCCM experiment suite.

This script matches the outputs stored under:
  outputs_mccm/N100_beta1
  outputs_mccm/N100_beta0p1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from evaluate_assortment import load_assortment_json
from generate_assortment_data import build_mccm_model_from_meta


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_mip_json(path: Path) -> Any:
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("Restricted license"):
        lines = lines[1:]
    return json.loads("\n".join(lines))


def _as_sample_string(sample: list[int]) -> str:
    return "".join(str(int(v)) for v in sample)


def _true_expected_reward(meta: dict[str, Any], assortment: list[int]) -> float:
    if meta.get("choice_model") == "mccm":
        model = build_mccm_model_from_meta(meta)
        probs = model.choice_probabilities(assortment).tolist()
        reward_vec = [float(meta.get("reward_no_purchase", 0.0))] + [
            float(x) for x in meta["rewards"]
        ]
        return float(sum(p * r for p, r in zip(probs, reward_vec)))
    utilities = meta["utilities"]
    rewards = meta["rewards"]
    outside_utility = float(meta["outside_utility"])
    reward_no_purchase = float(meta["reward_no_purchase"])
    denom = math.exp(outside_utility)
    numer = reward_no_purchase * math.exp(outside_utility)
    for s, u, r in zip(assortment, utilities, rewards):
        if s:
            eu = math.exp(float(u))
            denom += eu
            numer += float(r) * eu
    return numer / denom


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


def _diffusion_summary(path: Path) -> dict[str, Any]:
    metrics = _load_json(path / "metrics.json")
    history = metrics["history"]
    best = min(history, key=lambda row: row["val_loss"])
    final = history[-1]
    return {
        "best_epoch": metrics.get("best_epoch", best["epoch"]),
        "best_val_loss": metrics.get("best_val_loss", best["val_loss"]),
        "final_val_loss": final["val_loss"],
        "final_bit_accuracy": final["val_bit_accuracy"],
        "run_config": _load_json(path / "run_config.json"),
    }


def _exhaustive_summary(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load_json(path)
    best = data["best_assortment"]
    return {
        "num_scored": data.get("num_scored"),
        "best_assortment": _as_sample_string(best),
        "best_value": data["best_value"],
        "true_expected_reward": _true_expected_reward(meta, best),
    }


def _mip_summary(path: Path, meta: dict[str, Any], optimal_reward: float) -> dict[str, Any]:
    data = _load_mip_json(path)
    best = data["best_assortment"]
    true_reward = _true_expected_reward(meta, best)
    return {
        "solver_status": data["solver_status"],
        "iterations": data["iterations"],
        "runtime_sec": data.get("runtime_sec"),
        "mip_value": data["mip_value"],
        "approx_value": data["approx_value"],
        "best_value": data["best_value"],
        "best_assortment": _as_sample_string(best),
        "true_expected_reward": true_reward,
        "optimal_ratio": true_reward / optimal_reward,
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


def _guided_summary(path: Path, optimal_reward: float, reference_solution: str) -> dict[str, Any]:
    data = _load_json(path)
    true_rewards = [float(x) for x in data["true_expected_rewards"]]
    guided_scores = [float(x) for x in data["guided_scores"]]
    samples = [_as_sample_string(sample) for sample in data["samples"]]
    ratios = [reward / optimal_reward for reward in true_rewards]
    best_guided_idx = max(range(len(guided_scores)), key=guided_scores.__getitem__)
    best_true_idx = max(range(len(true_rewards)), key=true_rewards.__getitem__)
    contains_reference = reference_solution in samples
    return {
        "num_samples": data["num_samples"],
        "guidance_scale": data["guidance_scale"],
        "guidance_gamma": data["guidance_gamma"],
        "runtime_sec": data.get("runtime_sec"),
        "true_reward_mean": float(statistics.fmean(true_rewards)),
        "true_reward_std": float(statistics.pstdev(true_rewards)),
        "true_reward_q50": _ratio_quantiles(true_rewards)["q50"],
        "true_reward_q90": _ratio_quantiles(true_rewards)["q90"],
        "best_true_sample": samples[best_true_idx],
        "best_true_reward": true_rewards[best_true_idx],
        "contains_reference_solution": contains_reference,
        "count_reference_solution": sum(1 for sample in samples if sample == reference_solution),
        "best_guided_score": guided_scores[best_guided_idx],
        "best_guided_sample": samples[best_guided_idx],
        "best_guided_true_expected_reward": true_rewards[best_guided_idx],
        "optimal_ratio_stats": _ratio_quantiles(ratios),
        "best_optimal_ratio": max(ratios),
    }


def build_report(data_path: Path, outdir: Path) -> dict[str, Any]:
    dataset = load_assortment_json(data_path)
    meta = dataset["meta"]
    exhaustive = _exhaustive_summary(outdir / "exhaustive_true_mccm.json", meta)
    optimal_reward = exhaustive["best_value"]
    return {
        "dataset": {
            "path": str(data_path),
            "N": meta["N"],
            "n_samples": meta["n_samples"],
            "choice_model": meta["choice_model"],
            "assortment_policy": meta["assortment_policy"],
            "boltzmann_sampler": meta["assortment_policy"].get("boltzmann_sampler"),
            "boltzmann_beta": meta["assortment_policy"].get("boltzmann_beta"),
        },
        "training": {
            "gasn": _training_summary(outdir / "gasn"),
            "rasn": _training_summary(outdir / "rasn"),
            "diffusion": _diffusion_summary(outdir / "diffusion"),
        },
        "exhaustive": exhaustive,
        "mip": {
            "gasn": _mip_summary(outdir / "mip_gasn.json", meta, optimal_reward),
            "rasn": _mip_summary(outdir / "mip_rasn.json", meta, optimal_reward),
        },
        "guided_sampling": {
            "gasn": _guided_summary(
                outdir / "guided_gasn_scale1000_gamma3_n256.json",
                optimal_reward,
                exhaustive["best_assortment"],
            ),
            "rasn": _guided_summary(
                outdir / "guided_rasn_scale1000_gamma3_n256.json",
                optimal_reward,
                exhaustive["best_assortment"],
            ),
        },
        "notes": [
            f"The exhaustive MCCM solve is the reference optimum for N={meta['N']}.",
            "MIP solutions are reported with runtime and optimal ratios against the true optimum.",
            "Guided sampling uses the trained choice model as the reward guidance signal.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.data, args.outdir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
