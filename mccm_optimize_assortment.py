#!/usr/bin/env python3
"""Exact optimal assortment under the Markov chain choice model.

For the absorbing Markov chain choice model used in this repository, offering a
product is equivalent to choosing a stop action at that state:

    V_0 = r_0
    V_i = max(r_i, sum_j P[i, j] * V_j)   for i = 1..N

where ``r_i`` is the reward of choosing product ``i`` and ``P`` is the
transition matrix. This is a finite Markov decision process with one binary
decision per product state, so Howard policy iteration gives an exact optimal
assortment without enumerating all subsets.

This script is intended to serve as the large-N reference solver in the offline
experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch

from assortment_dataset import load_assortment_json


def assortment_to_string(assortment: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in assortment)


def _load_mccm_tensors(meta: dict, *, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    device = torch.device(device)
    initial_probs = meta.get("mccm_initial_probs")
    transition_matrix = meta.get("mccm_transition_matrix")
    rewards = meta.get("rewards")
    if initial_probs is None:
        raise ValueError("Metadata does not contain 'mccm_initial_probs'.")
    if transition_matrix is None:
        raise ValueError("Metadata does not contain 'mccm_transition_matrix'.")
    if rewards is None:
        raise ValueError("Metadata does not contain 'rewards'.")

    initial = torch.tensor(initial_probs, dtype=torch.float64, device=device)
    transition = torch.tensor(transition_matrix, dtype=torch.float64, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float64, device=device)
    reward_no_purchase = float(meta.get("reward_no_purchase", 0.0))

    n_products = rewards_t.numel()
    expected_states = n_products + 1
    if initial.numel() != expected_states:
        raise ValueError(
            f"mccm_initial_probs must have length {expected_states}, got {initial.numel()}."
        )
    if transition.shape != (expected_states, expected_states):
        raise ValueError(
            f"mccm_transition_matrix must have shape [{expected_states}, {expected_states}], "
            f"got {tuple(transition.shape)}."
        )

    return initial, transition, rewards_t, reward_no_purchase


def _evaluate_policy(
    transition: torch.Tensor,
    rewards: torch.Tensor,
    reward_no_purchase: float,
    offered: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a fixed stop/continue policy exactly."""
    if offered.dtype != torch.bool:
        offered = offered.to(dtype=torch.bool)

    n_products = rewards.numel()
    device = transition.device
    dtype = transition.dtype

    values = torch.empty(n_products + 1, dtype=dtype, device=device)
    values[0] = reward_no_purchase
    values[1:][offered] = rewards[offered]

    continue_mask = ~offered
    if continue_mask.any():
        continue_idx = torch.nonzero(continue_mask, as_tuple=False).flatten() + 1
        offered_idx = torch.nonzero(offered, as_tuple=False).flatten() + 1

        mat = transition.index_select(0, continue_idx).index_select(1, continue_idx)
        rhs = transition.index_select(0, continue_idx)[:, 0] * reward_no_purchase
        if offered_idx.numel() > 0:
            rhs = rhs + transition.index_select(0, continue_idx).index_select(1, offered_idx) @ rewards.index_select(0, offered_idx - 1)

        eye = torch.eye(len(continue_idx), dtype=dtype, device=device)
        system = eye - mat
        try:
            solved = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            # Fall back to a least-squares solve if the system is numerically
            # ill-conditioned. The exact model should still recover the same
            # policy because the chain is absorbing in the generated datasets.
            solved = torch.linalg.lstsq(system, rhs.unsqueeze(-1)).solution.squeeze(-1)

        values.index_copy_(0, continue_idx, solved)

    return values


def solve_mccm_optimal_assortment(
    meta: dict,
    *,
    device: torch.device | str = "cpu",
    tol: float = 1e-10,
    max_iterations: int = 1000,
) -> dict:
    """Return the exact optimal assortment under the MCCM."""
    initial, transition, rewards, reward_no_purchase = _load_mccm_tensors(meta, device=device)
    n_products = rewards.numel()

    if max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    if tol < 0.0:
        raise ValueError("--tol must be non-negative.")

    policy = torch.zeros(n_products, dtype=torch.bool, device=transition.device)
    converged = False
    iterations = 0
    values = None

    for iterations in range(1, max_iterations + 1):
        values = _evaluate_policy(transition, rewards, reward_no_purchase, policy)
        continuation = transition @ values
        improved = policy.clone()

        better_offer = rewards > continuation[1:] + tol
        worse_offer = continuation[1:] > rewards + tol
        improved[better_offer] = True
        improved[worse_offer] = False

        if torch.equal(improved, policy):
            converged = True
            break
        policy = improved

    if not converged:
        raise RuntimeError(
            "Policy iteration did not converge within the allotted iterations."
        )

    if values is None:
        values = _evaluate_policy(transition, rewards, reward_no_purchase, policy)

    expected_value = float(torch.dot(initial, values).item())
    assortment = [1 if bool(x) else 0 for x in policy.tolist()]

    return {
        "best_assortment": assortment,
        "best_assortment_str": assortment_to_string(assortment),
        "best_value": expected_value,
        "state_values": [float(x) for x in values.tolist()],
        "iterations": iterations,
        "converged": converged,
        "solver_status": "OPTIMAL" if converged else "MAX_ITERATIONS_REACHED",
        "policy": "true_mccm",
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the optimal assortment exactly under the MCCM."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSON path.")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Policy-improvement tolerance used to break ties.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of policy-iteration steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for the linear algebra solve.",
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
    if meta.get("choice_model") != "mccm":
        raise ValueError("This solver is only defined for datasets with choice_model='mccm'.")

    start = time.monotonic()
    result = solve_mccm_optimal_assortment(
        meta,
        device=args.device,
        tol=args.tol,
        max_iterations=args.max_iterations,
    )
    result["data"] = str(args.data)
    result["runtime_sec"] = time.monotonic() - start
    result["tol"] = args.tol
    result["max_iterations"] = args.max_iterations
    result["device"] = args.device

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"solver status: {result['solver_status']}")
        print(f"best value: {result['best_value']}")
        print(f"best assortment: {result['best_assortment_str']}")
        print(f"iterations: {result['iterations']}")


if __name__ == "__main__":
    main()
