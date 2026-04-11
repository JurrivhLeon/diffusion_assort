#!/usr/bin/env python3
"""Compute the true optimal assortment for a dataset generated under MNL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from assortment_dataset import load_assortment_json


def expected_reward_prefix(
    ordered_items: list[tuple[int, float, float]],
    prefix_len: int,
    outside_utility: float,
    reward_no_purchase: float,
) -> float:
    denom = math.exp(outside_utility)
    numer = reward_no_purchase * denom
    for _, reward, utility in ordered_items[:prefix_len]:
        weight = math.exp(utility)
        denom += weight
        numer += reward * weight
    return numer / denom


def optimal_mnl_assortment(
    meta: dict[str, Any],
    *,
    max_cardinality: int | None = None,
) -> tuple[list[int], float]:
    if meta.get("choice_model") != "mnl":
        raise ValueError("This script only supports datasets with meta.choice_model='mnl'.")

    n_products = int(meta["N"])
    utilities = [float(x) for x in meta["utilities"]]
    rewards = [float(x) for x in meta["rewards"]]
    outside_utility = float(meta.get("outside_utility", 0.0))
    reward_no_purchase = float(meta.get("reward_no_purchase", 0.0))

    if len(utilities) != n_products or len(rewards) != n_products:
        raise ValueError("Metadata lengths do not match N.")

    ordered_items = sorted(
        [(idx, rewards[idx], utilities[idx]) for idx in range(n_products)],
        key=lambda row: row[1],
        reverse=True,
    )

    best_prefix_len = 0
    best_value = expected_reward_prefix(
        ordered_items,
        0,
        outside_utility,
        reward_no_purchase,
    )

    upper = n_products if max_cardinality is None else min(n_products, int(max_cardinality))
    for prefix_len in range(1, upper + 1):
        value = expected_reward_prefix(
            ordered_items,
            prefix_len,
            outside_utility,
            reward_no_purchase,
        )
        if value > best_value:
            best_value = value
            best_prefix_len = prefix_len

    assortment = [0] * n_products
    for idx, _, _ in ordered_items[:best_prefix_len]:
        assortment[idx] = 1
    return assortment, best_value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the true optimal assortment for an MNL dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to JSON dataset.")
    parser.add_argument(
        "--max-cardinality",
        type=int,
        default=None,
        help="Optional upper bound on assortment size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset = load_assortment_json(args.data)
    meta = dataset["meta"]

    assortment, value = optimal_mnl_assortment(
        meta,
        max_cardinality=args.max_cardinality,
    )
    result = {
        "data": str(args.data),
        "choice_model": meta["choice_model"],
        "N": int(meta["N"]),
        "max_cardinality": args.max_cardinality,
        "best_assortment": assortment,
        "best_value": value,
        "num_selected": int(sum(assortment)),
    }

    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()