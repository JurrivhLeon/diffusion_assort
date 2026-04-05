#!/usr/bin/env python3
"""Summarize the N=100, beta=1 Boltzmann experiment suite.

This collects:
- training metrics for GAsN, RAsN, and Bernoulli diffusion
- MIP results for GAsN and RAsN guidance models
- guided diffusion sampling summaries for GAsN and RAsN guidance

The report is written as JSON for easy inspection or downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _strip_gurobi_banner(path: Path) -> str:
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("Restricted license"):
        lines = lines[1:]
    return "\n".join(lines)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_mip_json(path: Path) -> Any:
    return json.loads(_strip_gurobi_banner(path))


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
    history = metrics["history"] if isinstance(metrics, dict) else metrics
    best_epoch = min(history, key=lambda row: row.get("val_nll", row.get("val_loss", float("inf"))))
    final = history[-1]
    best_metric_name = "val_nll" if "val_nll" in best_epoch else "val_loss"
    final_metric_name = "val_nll" if "val_nll" in final else "val_loss"
    return {
        "best_epoch": metrics.get("best_epoch", best_epoch["epoch"]) if isinstance(metrics, dict) else best_epoch["epoch"],
        "best_metric": metrics.get("best_val_nll", best_epoch[best_metric_name]) if isinstance(metrics, dict) else best_epoch[best_metric_name],
        "final_metric": final[final_metric_name],
        "final_accuracy": final.get("val_accuracy", final.get("val_bit_accuracy")),
        "run_config": _load_json(path / "run_config.json"),
    }


def _diffusion_summary(path: Path) -> dict[str, Any]:
    metrics = _load_json(path / "metrics.json")
    history = metrics["history"] if isinstance(metrics, dict) else metrics
    best_epoch = min(history, key=lambda row: row["val_loss"])
    final = history[-1]
    return {
        "best_epoch": metrics.get("best_epoch", best_epoch["epoch"]) if isinstance(metrics, dict) else best_epoch["epoch"],
        "best_val_loss": metrics.get("best_val_loss", best_epoch["val_loss"]) if isinstance(metrics, dict) else best_epoch["val_loss"],
        "final_val_loss": final["val_loss"],
        "final_bit_accuracy": final["val_bit_accuracy"],
        "run_config": _load_json(path / "run_config.json"),
    }


def _mip_summary(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load_mip_json(path)
    assortment = data["best_assortment"]
    return {
        "solver_status": data["solver_status"],
        "mip_value": data["mip_value"],
        "approx_value": data["approx_value"],
        "best_value": data["best_value"],
        "n_selected": sum(int(v) for v in assortment),
        "best_assortment": _as_sample_string(assortment),
        "true_expected_reward": _true_expected_reward(meta, assortment),
    }


def _guided_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    rewards = data["true_expected_rewards"]
    guided_scores = data.get("guided_scores", [])
    best_idx = max(range(len(rewards)), key=rewards.__getitem__)
    summary = {
        "num_samples": data["num_samples"],
        "guidance_scale": data["guidance_scale"],
        "guidance_gamma": data["guidance_gamma"],
        "mean_true_reward": sum(rewards) / len(rewards),
        "best_true_reward": rewards[best_idx],
        "best_true_sample": _as_sample_string(data["samples"][best_idx]),
    }
    if guided_scores:
        best_guided_idx = max(range(len(guided_scores)), key=guided_scores.__getitem__)
        summary.update(
            {
                "mean_guided_score": sum(guided_scores) / len(guided_scores),
                "best_guided_score": guided_scores[best_guided_idx],
                "best_guided_sample": _as_sample_string(data["samples"][best_guided_idx]),
                "best_guided_true_reward": rewards[best_guided_idx],
            }
        )
    return summary


def _maybe_guided_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _guided_summary(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/assortment_N100_n10000_boltzmann_beta1.json")
    parser.add_argument("--outdir", default="outputs_N100_beta1")
    parser.add_argument("--output", default="outputs_N100_beta1/report.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    report_path = Path(args.output)

    dataset = _load_json(data_path)
    meta = dataset["meta"]

    report = {
        "dataset": {
            "path": str(data_path),
            "N": meta["N"],
            "n_samples": meta["n_samples"],
            "choice_model": meta["choice_model"],
            "assortment_policy": meta["assortment_policy"],
            "boltzmann_sampler": meta.get("boltzmann_sampler", "mcmc"),
            "boltzmann_beta": 1.0,
        },
        "training": {
            "gasn": _training_summary(outdir / "gasn"),
            "rasn": _training_summary(outdir / "rasn"),
            "diffusion50": _diffusion_summary(outdir / "diffusion"),
            "diffusion100": _diffusion_summary(outdir / "diffusion100"),
        },
        "mip": {
            "gasn": _mip_summary(outdir / "mip_gasn.json", meta),
            "rasn": _mip_summary(outdir / "mip_rasn.json", meta),
        },
        "guided_sampling": {
            "gasn": {
                "diff50_scale100_gamma2": _maybe_guided_summary(outdir / "guided_gasn_diff50_scale100_gamma2_n256.json"),
                "diff50_scale1000_gamma3": _maybe_guided_summary(outdir / "guided_gasn_diff50_scale1000_gamma3_n256.json"),
                "diff100_scale100_gamma2": _maybe_guided_summary(outdir / "guided_gasn_diff100_scale100_gamma2_n256.json"),
                "diff100_scale1000_gamma3": _maybe_guided_summary(outdir / "guided_gasn_diff100_scale1000_gamma3_n256.json"),
            },
            "rasn": {
                "diff50_scale100_gamma2": _maybe_guided_summary(outdir / "guided_rasn_diff50_scale100_gamma2_n256.json"),
                "diff50_scale1000_gamma3": _maybe_guided_summary(outdir / "guided_rasn_diff50_scale1000_gamma3_n256.json"),
                "diff100_scale100_gamma2": _maybe_guided_summary(outdir / "guided_rasn_diff100_scale100_gamma2_n256.json"),
                "diff100_scale1000_gamma3": _maybe_guided_summary(outdir / "guided_rasn_diff100_scale1000_gamma3_n256.json"),
            },
        },
        "notes": [
            "The N=100 instance is too large for exhaustive search, so the MIP runs are the fitted-model baselines.",
            "Guided sampling includes both moderate and aggressive schedules, and both 50-step and 100-step diffusion models.",
        ],
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
