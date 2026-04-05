#!/usr/bin/env python3
"""Exhaustive assortment search.

This script enumerates all feasible assortments and returns the one with the
largest expected reward. It is exact, so it serves as a reference method when
N is small enough for exhaustive enumeration.

By default, the search uses:
  - the true MNL policy from the dataset metadata
  - the reward function stored in metadata, or a custom reward module

You can also evaluate assortments with a trained choice-model checkpoint by
passing ``--policy model`` and ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Sequence

import torch
from tqdm.auto import tqdm

from evaluate_assortment import (
    build_reward_fn,
    load_assortment_json,
    load_policy_from_checkpoint,
)
from generate_assortment_data import build_mccm_model_from_meta


def bits_to_assortment(mask: int, n_products: int) -> list[int]:
    return [(mask >> j) & 1 for j in range(n_products)]


def iter_assortments(
    n_products: int,
    *,
    max_cardinality: int | None = None,
    exact_cardinality: int | None = None,
):
    if exact_cardinality is not None:
        if exact_cardinality < 0 or exact_cardinality > n_products:
            raise ValueError("--exact-cardinality must be in [0, N].")
        for idxs in combinations(range(n_products), exact_cardinality):
            S = [0] * n_products
            for j in idxs:
                S[j] = 1
            yield S
        return

    upper = 1 << n_products
    for mask in range(upper):
        if max_cardinality is not None and mask.bit_count() > max_cardinality:
            continue
        yield bits_to_assortment(mask, n_products)


def count_assortments(
    n_products: int,
    *,
    max_cardinality: int | None = None,
    exact_cardinality: int | None = None,
) -> int:
    if exact_cardinality is not None:
        if exact_cardinality < 0 or exact_cardinality > n_products:
            raise ValueError("--exact-cardinality must be in [0, N].")
        return math.comb(n_products, exact_cardinality)

    if max_cardinality is None:
        return 1 << n_products

    if max_cardinality < 0 or max_cardinality > n_products:
        raise ValueError("--max-cardinality must be in [0, N].")

    return sum(math.comb(n_products, k) for k in range(max_cardinality + 1))


def build_policy_kind(
    meta: dict,
    policy: str,
    checkpoint: str | None,
):
    if policy == "true_mnl":
        utilities = meta.get("utilities")
        if utilities is None:
            raise ValueError("Metadata does not contain 'utilities'.")
        outside_utility = float(meta.get("outside_utility", 0.0))

        def true_policy(S: Sequence[int]) -> list[float]:
            offered = [i for i, bit in enumerate(S) if bit]
            logits = [outside_utility] + [utilities[i] for i in offered]
            max_logit = max(logits)
            exp_logits = [math.exp(x - max_logit) for x in logits]
            total = sum(exp_logits)
            probs = [x / total for x in exp_logits]
            full_probs = [0.0] * (len(utilities) + 1)
            full_probs[0] = probs[0]
            for pos, product_idx in enumerate(offered, start=1):
                full_probs[product_idx + 1] = probs[pos]
            return full_probs

        return true_policy

    if policy == "true_mccm":
        model = build_mccm_model_from_meta(meta)

        def true_policy(S: Sequence[int]) -> list[float]:
            return [float(p) for p in model.choice_probabilities(S).tolist()]

        return true_policy

    if policy == "model":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --policy model is used.")
        model = load_policy_from_checkpoint(checkpoint)

        def model_policy(S: Sequence[int]) -> list[float]:
            tensor = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
            return [float(p) for p in model(tensor).squeeze(0).tolist()]

        return model_policy

    raise ValueError("--policy must be one of: true_mnl, true_mccm, model.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exhaustively search for the optimal assortment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="JSON dataset containing metadata for the true policy and rewards.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="true_mnl",
        choices=("true_mnl", "true_mccm", "model"),
        help="Use the true policy from metadata or a trained checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path when --policy model is used.",
    )
    parser.add_argument(
        "--reward-module",
        type=Path,
        default=None,
        help="Optional Python module with a reward(S, a, meta) function.",
    )
    parser.add_argument(
        "--max-cardinality",
        type=int,
        default=None,
        help="Optional upper bound on assortment size.",
    )
    parser.add_argument(
        "--exact-cardinality",
        type=int,
        default=None,
        help="If set, search only assortments of this cardinality.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    raw = load_assortment_json(args.data)
    if "meta" not in raw:
        raise ValueError("Dataset JSON must contain a 'meta' field.")
    meta = raw["meta"]
    n_products = int(meta["N"])

    if args.max_cardinality is not None and args.max_cardinality < 0:
        raise ValueError("--max-cardinality must be non-negative.")
    if args.max_cardinality is not None and args.max_cardinality > n_products:
        raise ValueError("--max-cardinality cannot exceed N.")
    if args.exact_cardinality is not None and args.exact_cardinality > n_products:
        raise ValueError("--exact-cardinality cannot exceed N.")

    total = count_assortments(
        n_products,
        max_cardinality=args.max_cardinality,
        exact_cardinality=args.exact_cardinality,
    )

    reward_fn = build_reward_fn(str(args.reward_module) if args.reward_module else None)
    policy_fn = build_policy_kind(
        meta,
        args.policy,
        str(args.checkpoint) if args.checkpoint else None,
    )

    best_S: list[int] | None = None
    best_value = float("-inf")
    num_scored = 0

    with tqdm(total=total, unit="assortment", desc="Exhaustive search") as pbar:
        for S in iter_assortments(
            n_products,
            max_cardinality=args.max_cardinality,
            exact_cardinality=args.exact_cardinality,
        ):
            probs = policy_fn(S)
            if len(probs) != n_products + 1:
                raise ValueError("Policy probability vector has unexpected length.")
            value = float(
                sum(reward_fn(S, a, meta) * probs[a] for a in range(len(probs)))
            )
            num_scored += 1
            pbar.update(1)
            if value > best_value:
                best_value = value
                best_S = list(S)

    if best_S is None:
        raise RuntimeError("No assortments were scored.")

    result = {
        "best_assortment": best_S,
        "best_value": best_value,
        "num_scored": num_scored,
        "policy": args.policy,
        "data": str(args.data),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "reward_module": str(args.reward_module) if args.reward_module else None,
        "max_cardinality": args.max_cardinality,
        "exact_cardinality": args.exact_cardinality,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"best value: {best_value}")
        print(f"best assortment: {''.join(str(x) for x in best_S)}")
        print(f"num scored: {num_scored}")


if __name__ == "__main__":
    main()
