#!/usr/bin/env python3
"""Sample assortments with guided Bernoulli diffusion.

This script loads a trained Bernoulli diffusion checkpoint and a trained choice
model checkpoint, then uses the choice-model expected reward as the guidance
signal during reverse diffusion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bernoulli_diffusion import BernoulliDiffusion
from diffusion_dataset import load_assortment_json
from evaluate_assortment import build_batched_expected_reward_fn, build_reward_fn


def _load_diffusion_checkpoint(path: Path, *, device: torch.device | str) -> BernoulliDiffusion:
    ckpt = torch.load(path, map_location=device)
    try:
        n_products = int(ckpt["n_products"])
        num_steps = int(ckpt["num_steps"])
        beta_start = float(ckpt["beta_start"])
        beta_end = float(ckpt["beta_end"])
        hidden_sizes = ckpt.get("hidden_sizes", [n_products])
        time_emb_dim = int(ckpt.get("time_emb_dim", 32))
        model_state_dict = ckpt["model_state_dict"]
    except KeyError as exc:
        raise KeyError(f"Missing expected diffusion checkpoint field: {exc.args[0]}") from exc

    model = BernoulliDiffusion(
        n_products=n_products,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        hidden_sizes=hidden_sizes,
        time_emb_dim=time_emb_dim,
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guided Bernoulli diffusion sampler.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSON path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Diffusion checkpoint (.pt).")
    parser.add_argument(
        "--choice-checkpoint",
        type=Path,
        required=True,
        help="Trained GAsN/RAsN checkpoint used as the guidance model.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="model",
        choices=("model", "true_mnl", "true_mmnl", "true_mccm"),
        help="Choice model used to score candidates during guidance.",
    )
    parser.add_argument(
        "--reward-module",
        type=Path,
        default=None,
        help="Optional reward module for custom reward functions.",
    )
    parser.add_argument("--num-samples", type=int, default=256, help="Number of guided samples.")
    parser.add_argument("--guidance-scale", type=float, default=100.0, help="Guidance scale.")
    parser.add_argument("--guidance-gamma", type=float, default=2.0, help="Guidance gamma.")
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=None,
        help="Optional chunk size for candidate scoring.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.guidance_scale < 0.0:
        raise ValueError("--guidance-scale must be non-negative.")
    if args.guidance_gamma < 1.0:
        raise ValueError("--guidance-gamma must be at least 1.")
    if args.candidate_chunk_size is not None and args.candidate_chunk_size <= 0:
        raise ValueError("--candidate-chunk-size must be positive.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    raw = load_assortment_json(args.data)
    if "meta" not in raw:
        raise ValueError("Dataset JSON must contain a 'meta' field.")
    meta = raw["meta"]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    diffusion = _load_diffusion_checkpoint(args.checkpoint, device=device)
    reward_fn = build_reward_fn(args.reward_module)
    score_fn = build_batched_expected_reward_fn(
        meta,
        reward_fn,
        args.policy,
        str(args.choice_checkpoint),
        device=device,
    )
    if meta.get("choice_model") == "mccm":
        true_policy_kind = "true_mccm"
    elif meta.get("choice_model") == "mmnl":
        true_policy_kind = "true_mmnl"
    else:
        true_policy_kind = "true_mnl"
    true_reward_fn = build_batched_expected_reward_fn(
        meta,
        reward_fn,
        true_policy_kind,
        None,
        device=device,
    )

    start = time.monotonic()
    samples_t = diffusion.guided_sample(
        args.num_samples,
        score_fn,
        guidance_scale=args.guidance_scale,
        guidance_gamma=args.guidance_gamma,
        candidate_chunk_size=args.candidate_chunk_size,
        device=device,
    )
    runtime_sec = time.monotonic() - start

    samples = [[int(v) for v in row] for row in samples_t.detach().cpu().tolist()]
    scores_t = score_fn(samples_t.to(device)).detach().cpu()
    true_rewards_t = true_reward_fn(samples_t.to(device)).detach().cpu()
    guided_scores = [float(x) for x in scores_t.tolist()]
    true_expected_rewards = [float(x) for x in true_rewards_t.tolist()]

    best_guided_sample_index = max(range(len(guided_scores)), key=guided_scores.__getitem__)
    best_guided_sample = samples[best_guided_sample_index]
    best_guided_score = guided_scores[best_guided_sample_index]
    best_guided_true_expected_reward = true_expected_rewards[best_guided_sample_index]

    result = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "policy": args.policy,
        "choice_checkpoint": str(args.choice_checkpoint),
        "reward_module": str(args.reward_module) if args.reward_module else None,
        "num_samples": args.num_samples,
        "guidance_scale": float(args.guidance_scale),
        "guidance_gamma": float(args.guidance_gamma),
        "device": str(device),
        "samples": samples,
        "guided_scores": guided_scores,
        "true_expected_rewards": true_expected_rewards,
        "best_guided_sample_index": best_guided_sample_index,
        "best_guided_sample": best_guided_sample,
        "best_guided_score": best_guided_score,
        "best_guided_true_expected_reward": best_guided_true_expected_reward,
        "runtime_sec": runtime_sec,
        "seed": args.seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
