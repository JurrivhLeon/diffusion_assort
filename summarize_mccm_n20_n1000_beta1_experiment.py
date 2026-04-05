#!/usr/bin/env python3
"""Summarize the N=20, beta=1 Boltzmann experiment suite."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


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
    best_metric = min(history, key=lambda row: row.get("val_nll", row.get("val_loss", float("inf"))))
    final = history[-1]
    metric_key = "val_nll" if "val_nll" in best_metric else "val_loss"
    final_key = "val_nll" if "val_nll" in final else "val_loss"
    return {
        "best_epoch": metrics.get("best_epoch", best_metric["epoch"]),
        "best_metric": metrics.get("best_val_nll", metrics.get("best_val_loss", best_metric[metric_key])),
        "final_metric": final[final_key],
        "final_accuracy": final.get("val_accuracy", final.get("val_bit_accuracy")),
        "run_config": _load_json(path / "run_config.json"),
    }


def _diffusion_summary(path: Path) -> dict[str, Any]:
    metrics = _load_json(path / "metrics.json")
    history = metrics["history"]
    best_epoch = min(history, key=lambda row: row["val_loss"])
    final = history[-1]
    return {
        "best_epoch": metrics.get("best_epoch", best_epoch["epoch"]),
        "best_val_loss": metrics.get("best_val_loss", best_epoch["val_loss"]),
        "final_val_loss": final["val_loss"],
        "final_bit_accuracy": final["val_bit_accuracy"],
        "run_config": _load_json(path / "run_config.json"),
    }


def _exhaustive_summary(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load_json(path)
    best = data["best_assortment"]
    return {
        "num_scored": data["num_scored"],
        "best_assortment": _as_sample_string(best),
        "best_value": data["best_value"],
        "true_expected_reward": _true_expected_reward(meta, best),
    }


def _mip_summary(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load_mip_json(path)
    best = data["best_assortment"]
    return {
        "solver_status": data["solver_status"],
        "iterations": data["iterations"],
        "mip_value": data["mip_value"],
        "approx_value": data["approx_value"],
        "best_value": data["best_value"],
        "best_assortment": _as_sample_string(best),
        "true_expected_reward": _true_expected_reward(meta, best),
    }


def _guided_summary(path: Path, true_optimum: str) -> dict[str, Any]:
    data = _load_json(path)
    rewards = data["true_expected_rewards"]
    guided_scores = data["guided_scores"]
    samples = [_as_sample_string(sample) for sample in data["samples"]]
    best_true_idx = max(range(len(rewards)), key=rewards.__getitem__)
    best_guided_idx = max(range(len(guided_scores)), key=guided_scores.__getitem__)
    sorted_rewards = sorted(rewards)

    def quantile(q: float) -> float:
        if not sorted_rewards:
            return float("nan")
        if len(sorted_rewards) == 1:
            return float(sorted_rewards[0])
        pos = (len(sorted_rewards) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_rewards[lo])
        weight = pos - lo
        return float(sorted_rewards[lo] * (1.0 - weight) + sorted_rewards[hi] * weight)

    return {
        "num_samples": data["num_samples"],
        "guidance_scale": data["guidance_scale"],
        "guidance_gamma": data["guidance_gamma"],
        "mean_true_reward": sum(rewards) / len(rewards),
        "true_reward_std": statistics.pstdev(rewards),
        "true_reward_q50": quantile(0.5),
        "true_reward_q90": quantile(0.9),
        "best_true_reward": rewards[best_true_idx],
        "best_true_sample": samples[best_true_idx],
        "best_guided_score": guided_scores[best_guided_idx],
        "best_guided_sample": samples[best_guided_idx],
        "contains_true_optimum": true_optimum in samples,
        "count_true_optimum": sum(1 for sample in samples if sample == true_optimum),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/assortment_N20_n1000_boltzmann.json")
    parser.add_argument("--outdir", default="outputs_mccm_N20_n1000_beta1")
    parser.add_argument("--output", default="outputs_mccm_N20_n1000_beta1/report.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    dataset = _load_json(data_path)
    meta = dataset["meta"]

    exhaustive = _exhaustive_summary(outdir / "exhaustive_true.json", meta)
    true_optimum = exhaustive["best_assortment"]

    report = {
        "dataset": {
            "path": str(data_path),
            "N": meta["N"],
            "n_samples": meta["n_samples"],
            "choice_model": meta["choice_model"],
            "assortment_policy": meta["assortment_policy"],
            "boltzmann_sampler": meta.get("boltzmann_sampler", "exact"),
            "boltzmann_beta": meta["assortment_policy"].get("boltzmann_beta", 1.0)
            if isinstance(meta.get("assortment_policy"), dict)
            else 1.0,
        },
        "training": {
            "gasn": _training_summary(outdir / "gasn"),
            "rasn": _training_summary(outdir / "rasn"),
            "diffusion": _diffusion_summary(outdir / "diffusion"),
        },
        "exhaustive": exhaustive,
        "mip": {
            "gasn": _mip_summary(outdir / "mip_gasn.json", meta),
            "rasn": _mip_summary(outdir / "mip_rasn.json", meta),
        },
        "guided_sampling": {
            "gasn": _guided_summary(outdir / "guided_gasn_scale100_gamma2_n256.json", true_optimum),
            "rasn": _guided_summary(outdir / "guided_rasn_scale100_gamma2_n256.json", true_optimum),
        },
        "notes": [
            "This suite uses the N=20, beta=1 Boltzmann dataset.",
            "The exhaustive search is the exact reference solution.",
            "The MIP baselines optimize the fitted GAsN/RAsN surrogates.",
            "Guided sampling uses the scheduled diffusion guidance with guidance_scale=100 and gamma=2.",
        ],
    }

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
