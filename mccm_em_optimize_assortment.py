#!/usr/bin/env python3
"""Estimate an MCCM by EM and solve the resulting optimal assortment exactly.

This baseline fits the absorbing Markov chain choice model to logged
assortment-choice data with an EM loop, then plugs the estimated parameters into
the exact MCCM optimizer from :mod:`mccm_optimize_assortment`.

The intent is to provide a much faster alternative to the neural-network MIP
baseline while still being model-based.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from assortment_dataset import load_assortment_json
from generate_assortment_data import build_mccm_model_from_meta
from mccm_optimize_assortment import solve_mccm_optimal_assortment


def assortment_to_string(assortment: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in assortment)


def _load_data(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = load_assortment_json(path)
    if "meta" not in raw:
        raise ValueError("Dataset JSON must contain a 'meta' field.")
    if "data" not in raw:
        raise ValueError("Dataset JSON must contain a 'data' field.")
    return raw["meta"], raw["data"]


def _preprocess_observations(
    data: list[dict[str, Any]],
    n_products: int,
) -> list[dict[str, torch.Tensor | int]]:
    n_states = n_products + 1
    all_states = torch.arange(n_states, dtype=torch.long)
    obs: list[dict[str, torch.Tensor | int]] = []

    for row in data:
        assortment = row["S"]
        choice = int(row["A"])
        if len(assortment) != n_products:
            raise ValueError(
                f"Assortment length {len(assortment)} does not match N={n_products}."
            )
        if not 0 <= choice <= n_products:
            raise ValueError(f"Choice {choice} is outside the valid range 0..{n_products}.")

        offered = torch.tensor([idx + 1 for idx, bit in enumerate(assortment) if int(bit)], dtype=torch.long)
        absorbing_mask = torch.zeros(n_states, dtype=torch.bool)
        absorbing_mask[0] = True
        if offered.numel() > 0:
            absorbing_mask.index_fill_(0, offered, True)
        transient = all_states[~absorbing_mask]

        obs.append(
            {
                "choice": choice,
                "offered": offered,
                "transient": transient,
            }
        )

    return obs


def _initialize_parameters(
    data: list[dict[str, Any]],
    n_products: int,
    *,
    pseudocount: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_states = n_products + 1
    choice_counts = torch.full((n_states,), float(pseudocount), dtype=dtype, device=device)
    for row in data:
        choice_counts[int(row["A"])] += 1.0
    initial_probs = choice_counts / choice_counts.sum()

    transition_matrix = torch.full(
        (n_states, n_states),
        float(pseudocount),
        dtype=dtype,
        device=device,
    )
    transition_matrix[0].zero_()
    transition_matrix[0, 0] = 1.0
    for i in range(1, n_states):
        transition_matrix[i, i] = 0.0
        transition_matrix[i] /= transition_matrix[i].sum()
    return initial_probs, transition_matrix


def _em_step(
    initial_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    observations: list[dict[str, torch.Tensor | int]],
    *,
    pseudocount: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device = initial_probs.device
    dtype = initial_probs.dtype
    n_states = initial_probs.numel()

    start_counts = torch.full((n_states,), float(pseudocount), dtype=dtype, device=device)
    trans_counts = torch.full((n_states, n_states), float(pseudocount), dtype=dtype, device=device)
    trans_counts[0].zero_()
    trans_counts[0, 0] = 1.0
    total_loglik = 0.0

    for obs in observations:
        choice = int(obs["choice"])
        transient = obs["transient"]
        if not isinstance(transient, torch.Tensor):
            raise TypeError("Transient index cache corrupted.")

        if transient.numel() == 0:
            p_obs = torch.clamp(initial_probs[choice], min=eps)
            start_counts[choice] += 1.0
            total_loglik += float(torch.log(p_obs).item())
            continue

        Q = transition_matrix.index_select(0, transient).index_select(1, transient)
        r = transition_matrix.index_select(0, transient)[:, choice]
        eye = torch.eye(transient.numel(), dtype=dtype, device=device)
        h = torch.linalg.solve(eye - Q, r.unsqueeze(-1)).squeeze(-1)

        start_prob_transient = initial_probs.index_select(0, transient)
        p_obs = initial_probs[choice] + torch.dot(start_prob_transient, h)
        p_obs = torch.clamp(p_obs, min=eps)
        total_loglik += float(torch.log(p_obs).item())

        post_transient = start_prob_transient * h / p_obs
        start_counts[choice] += float(initial_probs[choice] / p_obs)
        start_counts.index_add_(0, transient, post_transient)

        active = h > eps
        if not bool(active.any()):
            continue

        active_idx = transient[active]
        h_active = h[active]
        post_active = post_transient[active]

        Q_active = transition_matrix.index_select(0, active_idx).index_select(1, active_idx)
        ratio = h_active.unsqueeze(0) / h_active.unsqueeze(1)
        Q_tilde = Q_active * ratio

        eye_active = torch.eye(active_idx.numel(), dtype=dtype, device=device)
        mu = torch.linalg.solve((eye_active - Q_tilde).transpose(0, 1), post_active.unsqueeze(-1)).squeeze(-1)

        for row_pos, state_i in enumerate(active_idx.tolist()):
            row_mass = float(mu[row_pos].item())
            if row_mass <= 0.0:
                continue
            trans_counts[state_i, active_idx] += row_mass * Q_tilde[row_pos]
            trans_counts[state_i, choice] += row_mass * float(transition_matrix[state_i, choice] / h_active[row_pos])

    return start_counts, trans_counts, total_loglik


def _m_step(
    start_counts: torch.Tensor,
    trans_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = start_counts.device
    dtype = start_counts.dtype
    n_states = start_counts.numel()

    initial_probs = start_counts / start_counts.sum()

    transition_matrix = torch.zeros((n_states, n_states), dtype=dtype, device=device)
    transition_matrix[0, 0] = 1.0
    for i in range(1, n_states):
        row = trans_counts[i].clone()
        row[i] = 0.0
        row_sum = row.sum()
        if float(row_sum) <= 0.0:
            allowed = torch.ones(n_states, dtype=torch.bool, device=device)
            allowed[i] = False
            row = allowed.to(dtype=dtype)
            row[i] = 0.0
            row_sum = row.sum()
        transition_matrix[i] = row / row_sum
        transition_matrix[i, i] = 0.0

    return initial_probs, transition_matrix


def estimate_mccm_parameters(
    meta: dict[str, Any],
    data: list[dict[str, Any]],
    *,
    max_iterations: int = 50,
    tol: float = 1e-5,
    pseudocount: float = 1e-3,
    eps: float = 1e-12,
    max_observations: int | None = None,
    subset_seed: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    dtype = torch.float64
    n_products = int(meta["N"])
    if max_observations is not None and max_observations > 0 and max_observations < len(data):
        rng = random.Random(subset_seed)
        chosen = sorted(rng.sample(range(len(data)), k=int(max_observations)))
        data = [data[idx] for idx in chosen]
    observations = _preprocess_observations(data, n_products)
    initial_probs, transition_matrix = _initialize_parameters(
        data,
        n_products,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    )

    loglik_prev = float("-inf")
    converged = False
    iterations = 0
    start = time.monotonic()

    for iteration in range(1, max_iterations + 1):
        start_counts, trans_counts, loglik = _em_step(
            initial_probs,
            transition_matrix,
            observations,
            pseudocount=pseudocount,
            eps=eps,
        )
        initial_probs, transition_matrix = _m_step(start_counts, trans_counts)
        iterations = iteration

        if math.isfinite(loglik_prev):
            delta = abs(loglik - loglik_prev)
            scale = 1.0 + abs(loglik_prev)
            if delta <= tol * scale:
                converged = True
                loglik_prev = loglik
                break
        loglik_prev = loglik

    runtime = time.monotonic() - start

    return {
        "initial_probs": [float(x) for x in initial_probs.tolist()],
        "transition_matrix": [[float(v) for v in row] for row in transition_matrix.tolist()],
        "loglik": float(loglik_prev),
        "iterations": iterations,
        "converged": converged,
        "runtime_sec": runtime,
        "device": str(device),
        "pseudocount": pseudocount,
        "tol": tol,
        "max_iterations": max_iterations,
    }


def solve_estimated_mccm(
    meta: dict[str, Any],
    est: dict[str, Any],
    *,
    solver_tol: float = 1e-10,
    solver_max_iterations: int = 1000,
) -> dict[str, Any]:
    estimated_meta = dict(meta)
    estimated_meta["choice_model"] = "mccm"
    estimated_meta["mccm_initial_probs"] = est["initial_probs"]
    estimated_meta["mccm_transition_matrix"] = est["transition_matrix"]
    result = solve_mccm_optimal_assortment(
        estimated_meta,
        device="cpu",
        tol=solver_tol,
        max_iterations=solver_max_iterations,
    )
    result["estimated_meta"] = {
        "initial_probs": est["initial_probs"],
        "transition_matrix": est["transition_matrix"],
    }
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit an MCCM with EM and solve the estimated model exactly."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSON path.")
    parser.add_argument(
        "--em-max-iterations",
        type=int,
        default=50,
        help="Maximum EM iterations.",
    )
    parser.add_argument(
        "--em-tol",
        type=float,
        default=1e-5,
        help="Relative log-likelihood tolerance for EM convergence.",
    )
    parser.add_argument(
        "--em-pseudocount",
        type=float,
        default=1e-3,
        help="Pseudocount used to initialize and regularize EM updates.",
    )
    parser.add_argument(
        "--em-eps",
        type=float,
        default=1e-12,
        help="Numerical floor used when conditioning on a choice event.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for EM linear algebra.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=None,
        help="Optional cap on the number of observed (S, A) pairs used for EM fitting.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="Random seed used when subsampling observations for EM.",
    )
    parser.add_argument(
        "--solver-tol",
        type=float,
        default=1e-10,
        help="Tolerance passed to the exact MCCM policy-iteration solver.",
    )
    parser.add_argument(
        "--solver-max-iterations",
        type=int,
        default=1000,
        help="Max policy-iteration steps for the exact MCCM solver.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    meta, data = _load_data(args.data)
    if meta.get("choice_model") != "mccm":
        raise ValueError("This baseline is only defined for datasets with choice_model='mccm'.")

    total_start = time.monotonic()
    est = estimate_mccm_parameters(
        meta,
        data,
        max_iterations=args.em_max_iterations,
        tol=args.em_tol,
        pseudocount=args.em_pseudocount,
        eps=args.em_eps,
        max_observations=args.max_observations,
        subset_seed=args.subset_seed,
        device=args.device,
    )
    solve_start = time.monotonic()
    solved = solve_estimated_mccm(
        meta,
        est,
        solver_tol=args.solver_tol,
        solver_max_iterations=args.solver_max_iterations,
    )
    solve_runtime = time.monotonic() - solve_start

    true_model = build_mccm_model_from_meta(meta)
    true_reward = float(true_model.expected_reward(solved["best_assortment"]))

    result = {
        "backend": "em_exact",
        "policy": "estimated_mccm",
        "data": str(args.data),
        "em": est,
        "best_assortment": solved["best_assortment"],
        "best_assortment_str": solved["best_assortment_str"],
        "best_value": solved["best_value"],
        "solver_status": solved["solver_status"],
        "iterations": solved["iterations"],
        "state_values": solved["state_values"],
        "true_expected_reward": true_reward,
        "runtime_sec": time.monotonic() - total_start,
        "em_runtime_sec": est["runtime_sec"],
        "solve_runtime_sec": solve_runtime,
        "solver_tol": args.solver_tol,
        "solver_max_iterations": args.solver_max_iterations,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"backend: {result['backend']}")
        print(f"solver status: {result['solver_status']}")
        print(f"best value: {result['best_value']}")
        print(f"true expected reward: {result['true_expected_reward']}")
        print(f"best assortment: {result['best_assortment_str']}")
        print(f"em iterations: {est['iterations']}")


if __name__ == "__main__":
    main()
