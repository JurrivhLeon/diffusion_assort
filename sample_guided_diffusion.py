#!/usr/bin/env python3
"""Sample assortments from a guided Bernoulli diffusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from assortment_dataset import load_assortment_json
from bernoulli_diffusion import BernoulliDiffusion
from evaluate_assortment import (
    build_batched_expected_reward_fn,
    build_reward_fn,
)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_diffusion_checkpoint(path: Path, *, device: torch.device | str = "cpu") -> BernoulliDiffusion:
    device = torch.device(device)
    ckpt = torch.load(path, map_location=device)
    model = BernoulliDiffusion(
        n_products=int(ckpt["n_products"]),
        num_steps=int(ckpt["num_steps"]),
        beta_start=float(ckpt["beta_start"]),
        beta_end=float(ckpt["beta_end"]),
        hidden_sizes=ckpt.get("hidden_sizes", [128, 128]),
        time_emb_dim=int(ckpt.get("time_emb_dim", 32)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample from guided Bernoulli diffusion.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSON path.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained Bernoulli diffusion checkpoint.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="true_mnl",
        choices=("true_mnl", "model"),
        help="Policy used to score assortments for guidance.",
    )
    parser.add_argument(
        "--choice-checkpoint",
        type=Path,
        default=None,
        help="Choice-model checkpoint when --policy model is used.",
    )
    parser.add_argument(
        "--reward-module",
        type=Path,
        default=None,
        help="Optional reward module with reward(S, a, meta).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Total number of guided samples to generate.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=32,
        help="How many diffusion chains to sample in parallel.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Maximum multiplier lambda_max for the reward-difference guidance term.",
    )
    parser.add_argument(
        "--guidance-gamma",
        type=float,
        default=1.0,
        help="Exponent gamma in lambda_t=lambda_max*(1-(t-1)/(T-1))^gamma.",
    )
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=None,
        help="Optional chunk size for evaluating single-bit flip candidates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the sampled assortments.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use: auto, cpu, cuda, cuda:0, ...",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.sample_batch_size <= 0:
        raise ValueError("--sample-batch-size must be positive.")

    device = resolve_device(args.device)

    raw = load_assortment_json(args.data)
    if "meta" not in raw:
        raise ValueError("Dataset JSON must contain a 'meta' field.")
    meta = raw["meta"]

    reward_fn = build_reward_fn(str(args.reward_module) if args.reward_module else None)
    score_fn = build_batched_expected_reward_fn(
        meta,
        reward_fn,
        args.policy,
        str(args.choice_checkpoint) if args.choice_checkpoint else None,
        device=device,
    )
    true_reward_fn = build_batched_expected_reward_fn(meta, reward_fn, "true_mnl", None, device=device)
    diffusion = load_diffusion_checkpoint(args.checkpoint, device=device)

    samples = []
    scores = []
    true_rewards = []
    remaining = args.num_samples
    while remaining > 0:
        batch_size = min(args.sample_batch_size, remaining)
        batch = diffusion.guided_sample(
            batch_size,
            score_fn,
            guidance_scale=args.guidance_scale,
            guidance_gamma=args.guidance_gamma,
            candidate_chunk_size=args.candidate_chunk_size,
            device=device,
        )
        batch_scores = score_fn(batch).detach().cpu().tolist()
        batch_true = true_reward_fn(batch).detach().cpu().tolist()
        for sample, score, true_reward in zip(batch.tolist(), batch_scores, batch_true):
            samples.append([int(v) for v in sample])
            scores.append(float(score))
            true_rewards.append(float(true_reward))
        remaining -= batch_size

    result = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "policy": args.policy,
        "choice_checkpoint": str(args.choice_checkpoint) if args.choice_checkpoint else None,
        "reward_module": str(args.reward_module) if args.reward_module else None,
        "num_samples": args.num_samples,
        "guidance_scale": args.guidance_scale,
        "guidance_gamma": args.guidance_gamma,
        "device": str(device),
        "samples": samples,
        "guided_scores": scores,
        "true_expected_rewards": true_rewards,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
