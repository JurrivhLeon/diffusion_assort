#!/usr/bin/env python3
"""Summarize the N=20, beta=0.1 experiment suite.

This script reads the artifacts produced under ``outputs_N20_beta0.1`` and
writes a compact comparison report to ``outputs_N20_beta0.1/report.json``.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from evaluate_assortment import build_reward_fn, expected_reward, load_assortment_json


ROOT = Path("outputs_N20_beta0.1")
DATA_PATH = Path("datasets/assortment_N20_n1000_boltzmann_beta0p1.json")
TRUE_OPTIMUM = "10010100110100010010"


def load_json_maybe_banner(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if text.startswith("Restricted license"):
        text = "\n".join(text.splitlines()[1:])
    return json.loads(text)


def bits_to_str(bits) -> str:
    return "".join(str(int(x)) for x in bits)


def summarize_training(path: Path, *, metric_key: str, accuracy_key: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data["history"]
    best_metric = data["best_val_nll"] if "best_val_nll" in data else data["best_val_loss"]
    final_entry = history[-1]
    return {
        "best_epoch": data["best_epoch"],
        "best_metric": best_metric,
        "final_metric": final_entry[metric_key],
        "final_accuracy": final_entry[accuracy_key],
        "device": data.get("device"),
    }


def summarize_mip(path: Path, meta: dict, reward_fn) -> dict:
    data = load_json_maybe_banner(path)
    assortment = data["best_assortment"]
    true_reward = expected_reward(
        assortment=assortment,
        meta=meta,
        reward_fn=reward_fn,
        policy_kind="true_mnl",
        checkpoint_path=None,
    )
    return {
        "best_assortment": bits_to_str(assortment),
        "model_objective_value": data["mip_value"],
        "true_expected_reward": true_reward,
        "solver_status": data.get("solver_status"),
        "iterations": data.get("iterations"),
        "selected_matches_true_optimum": bits_to_str(assortment) == TRUE_OPTIMUM,
    }


def summarize_guided(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rewards = data["true_expected_rewards"]
    guided_scores = data["guided_scores"]
    samples = [bits_to_str(sample) for sample in data["samples"]]
    best_true_i = max(range(len(rewards)), key=lambda i: rewards[i])
    best_guided_i = max(range(len(guided_scores)), key=lambda i: guided_scores[i])
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
        "num_samples": len(samples),
        "mean_true_reward": sum(rewards) / len(rewards),
        "true_reward_std": statistics.pstdev(rewards),
        "true_reward_q50": quantile(0.5),
        "true_reward_q90": quantile(0.9),
        "best_true_reward": rewards[best_true_i],
        "best_true_sample": samples[best_true_i],
        "best_guided_score": guided_scores[best_guided_i],
        "best_guided_sample": samples[best_guided_i],
        "best_guided_true_reward": rewards[best_guided_i],
        "contains_true_optimum": TRUE_OPTIMUM in samples,
        "count_true_optimum": samples.count(TRUE_OPTIMUM),
    }


def main() -> None:
    raw = load_assortment_json(DATA_PATH)
    meta = raw["meta"]
    reward_fn = build_reward_fn(None)

    report = {
        "dataset": str(DATA_PATH),
        "beta": 0.1,
        "true_optimum": {
            "assortment": TRUE_OPTIMUM,
            "best_value": load_json_maybe_banner(ROOT / "exhaustive_true.json")["best_value"],
        },
        "training": {
            "gasn": summarize_training(
                ROOT / "gasn/metrics.json", metric_key="val_nll", accuracy_key="val_accuracy"
            ),
            "rasn": summarize_training(
                ROOT / "rasn/metrics.json", metric_key="val_nll", accuracy_key="val_accuracy"
            ),
            "diffusion": {
                "best_epoch": json.loads((ROOT / "diffusion/metrics.json").read_text())["best_epoch"],
                "best_metric": json.loads((ROOT / "diffusion/metrics.json").read_text())["best_val_loss"],
                "final_metric": json.loads((ROOT / "diffusion/metrics.json").read_text())["history"][-1]["val_loss"],
                "final_accuracy": json.loads((ROOT / "diffusion/metrics.json").read_text())["history"][-1]["val_bit_accuracy"],
            },
        },
        "mip": {
            "gasn": summarize_mip(ROOT / "mip_gasn.json", meta, reward_fn),
            "rasn": summarize_mip(ROOT / "mip_rasn.json", meta, reward_fn),
        },
        "guided_sampling": {
            "gasn": summarize_guided(ROOT / "guided_gasn_scale100_gamma2_n256.json"),
            "rasn": summarize_guided(ROOT / "guided_rasn_scale100_gamma2_n256.json"),
        },
    }

    out_path = ROOT / "report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
