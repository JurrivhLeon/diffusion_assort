#!/usr/bin/env python3
"""Sample assortments from a trained Bernoulli diffusion model without guidance."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from evaluate_assortment import build_batched_expected_reward_fn, build_reward_fn, load_assortment_json
from sample_guided_diffusion import _load_diffusion_checkpoint


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    weight = pos - lo
    return float(values[lo] * (1.0 - weight) + values[hi] * weight)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "q50": _quantile(values, 0.5),
        "q90": _quantile(values, 0.9),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a Bernoulli diffusion model.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSON path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Diffusion checkpoint.")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    dataset = load_assortment_json(args.data)
    meta = dataset["meta"]
    diffusion = _load_diffusion_checkpoint(args.checkpoint, device=args.device)
    reward_fn = build_reward_fn(None)
    if meta.get("choice_model") == "mccm":
        true_policy_kind = "true_mccm"
    elif meta.get("choice_model") == "mmnl":
        true_policy_kind = "true_mmnl"
    else:
        true_policy_kind = "true_mnl"
    true_reward_fn = build_batched_expected_reward_fn(
        meta, reward_fn, true_policy_kind, None, device=args.device
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start = time.monotonic()
    samples_t = diffusion.sample(args.num_samples, device=args.device)
    runtime_sec = time.monotonic() - start

    true_rewards_t = true_reward_fn(samples_t.to(args.device))
    true_rewards = [float(x) for x in true_rewards_t.detach().cpu().tolist()]
    samples = [[int(v) for v in row] for row in samples_t.detach().cpu().tolist()]
    best_idx = int(max(range(len(true_rewards)), key=true_rewards.__getitem__))

    result = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "policy": "unguided",
        "num_samples": args.num_samples,
        "device": args.device,
        "samples": samples,
        "true_expected_rewards": true_rewards,
        "true_reward_stats": _stats(true_rewards),
        "best_true_sample_index": best_idx,
        "best_true_sample": samples[best_idx],
        "best_true_reward": float(true_rewards[best_idx]),
        "runtime_sec": runtime_sec,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["true_reward_stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
