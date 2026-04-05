#!/usr/bin/env python3
"""Synthetic assortment-choice dataset generator.

This script generates offline assortment optimization data under either a
multinomial logit (MNL) or Markov chain choice model (MCCM).

Output format:
{
  "meta": {
    "N": int,
    "choice_model": "mnl" | "mccm",
    "utilities": [...],  # MNL only
    "outside_utility": float,  # MNL only
    "mccm_initial_probs": [...],  # MCCM only
    "mccm_transition_matrix": [[...], ...],  # MCCM only
    "rewards": [...],
    "reward_no_purchase": 0.0,
    "assortment_policy": {...},
    "seed": int,
    "n_samples": int
  },
  "data": [
    {"S": [0,1,...,0], "A": int},
    ...
  ]
}

Each S is a binary vector in {0,1}^N and A is an observed choice in
{0,1,...,N}, where 0 denotes no-purchase.
"""

from __future__ import annotations

import argparse
from itertools import accumulate
import json
from dataclasses import dataclass
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch


@dataclass(frozen=True)
class ChoiceModelConfig:
    """Configuration for the synthetic assortment-choice generator."""

    n_products: int
    n_samples: int
    seed: int
    choice_model: str
    assortment_policy: str
    boltzmann_sampler: str
    assortment_size: int
    include_empty_assortment: bool
    boltzmann_beta: float
    mcmc_chains: int
    mcmc_burn_in: int
    mcmc_thinning: int
    mcmc_device: str
    mccm_concentration: float
    utility_mean: float
    utility_std: float
    reward_low: float
    reward_high: float
    outside_utility: float


def sample_dirichlet(rng: random.Random, concentration: float, size: int) -> List[float]:
    if size <= 0:
        raise ValueError("size must be positive.")
    if concentration <= 0.0:
        raise ValueError("concentration must be positive.")
    draws = [rng.gammavariate(concentration, 1.0) for _ in range(size)]
    total = sum(draws)
    if total <= 0.0:
        raise RuntimeError("Failed to sample a valid Dirichlet vector.")
    return [x / total for x in draws]


@dataclass(frozen=True)
class MCCMParameters:
    initial_probs: List[float]
    transition_matrix: List[List[float]]


def _emit_progress(prefix: str, current: int, total: int, *, width: int = 28) -> None:
    """Render a lightweight progress bar to stderr."""
    if total <= 0:
        return
    current = max(0, min(current, total))
    ratio = current / total
    filled = min(width, max(0, int(round(width * ratio))))
    bar = "#" * filled + "-" * (width - filled)
    message = f"{prefix} [{bar}] {current}/{total} ({ratio * 100:5.1f}%)"
    end = "\n" if current >= total else "\r"
    sys.stderr.write(message + end)
    sys.stderr.flush()


def sample_mccm_parameters(
    rng: random.Random,
    n_products: int,
    *,
    concentration: float = 1.0,
) -> MCCMParameters:
    """Sample a random absorbing Markov chain choice model."""
    if n_products <= 0:
        raise ValueError("n_products must be positive.")

    n_states = n_products + 1
    initial_probs = sample_dirichlet(rng, concentration, n_states)

    transition_matrix: List[List[float]] = [[0.0] * n_states for _ in range(n_states)]
    transition_matrix[0][0] = 1.0
    for i in range(1, n_states):
        row_without_self = sample_dirichlet(rng, concentration, n_states - 1)
        row = [0.0] * n_states
        idx = 0
        for j in range(n_states):
            if j == i:
                continue
            row[j] = row_without_self[idx]
            idx += 1
        transition_matrix[i] = row

    return MCCMParameters(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
    )


class MarkovChainChoiceModel:
    """Absorbing Markov chain choice model used for MCCM datasets."""

    def __init__(
        self,
        initial_probs: Sequence[float],
        transition_matrix: Sequence[Sequence[float]],
        rewards: Sequence[float],
        *,
        reward_no_purchase: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        device = torch.device(device)
        initial = torch.tensor(initial_probs, dtype=dtype, device=device)
        transition = torch.tensor(transition_matrix, dtype=dtype, device=device)
        rewards_t = torch.tensor(rewards, dtype=dtype, device=device)
        n_products = rewards_t.numel()
        expected_states = n_products + 1
        if initial.numel() != expected_states:
            raise ValueError(
                f"initial_probs must have length {expected_states}, got {initial.numel()}."
            )
        if transition.shape != (expected_states, expected_states):
            raise ValueError(
                f"transition_matrix must have shape [{expected_states}, {expected_states}], "
                f"got {tuple(transition.shape)}."
            )

        self.device = device
        self.dtype = dtype
        self.n_products = n_products
        self.initial_probs = initial
        self.transition_matrix = transition
        self.rewards = rewards_t
        self.reward_no_purchase = float(reward_no_purchase)
        self._expected_reward_cache: Dict[tuple[int, ...], float] = {}

    @property
    def reward_vector(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.tensor(
                    [self.reward_no_purchase], dtype=self.dtype, device=self.device
                ),
                self.rewards,
            ],
            dim=0,
        )

    def choice_probabilities(self, assortment: Sequence[int]) -> torch.Tensor:
        if len(assortment) != self.n_products:
            raise ValueError(
                f"Expected assortment length {self.n_products}, got {len(assortment)}."
            )

        offered = [idx + 1 for idx, bit in enumerate(assortment) if int(bit)]
        absorbing = [0] + offered
        n_states = self.n_products + 1
        absorbing_set = set(absorbing)
        transient = [state for state in range(n_states) if state not in absorbing_set]

        probs = torch.zeros(n_states, dtype=self.dtype, device=self.device)
        if not transient:
            probs[absorbing] = self.initial_probs[absorbing]
            return probs

        transient_idx = torch.tensor(transient, dtype=torch.long, device=self.device)
        absorbing_idx = torch.tensor(absorbing, dtype=torch.long, device=self.device)
        q = self.transition_matrix.index_select(0, transient_idx).index_select(1, transient_idx)
        r = self.transition_matrix.index_select(0, transient_idx).index_select(1, absorbing_idx)
        eye = torch.eye(len(transient), dtype=self.dtype, device=self.device)
        fundamental = torch.linalg.solve(eye - q, r)
        absorbed_from_transient = self.initial_probs.index_select(0, transient_idx) @ fundamental
        probs[absorbing_idx] = self.initial_probs.index_select(0, absorbing_idx) + absorbed_from_transient
        return probs

    def expected_reward(self, assortment: Sequence[int]) -> float:
        key = tuple(int(x) for x in assortment)
        cached = self._expected_reward_cache.get(key)
        if cached is not None:
            return cached
        probs = self.choice_probabilities(assortment)
        value = float(torch.dot(probs, self.reward_vector).item())
        self._expected_reward_cache[key] = value
        return value

    def batched_expected_reward(self, assortments: torch.Tensor) -> torch.Tensor:
        if assortments.dim() == 1:
            assortments = assortments.unsqueeze(0)
        if assortments.dim() != 2:
            raise ValueError("Expected assortments to have shape [N] or [B, N].")
        values = [self.expected_reward(row.tolist()) for row in assortments]
        return torch.tensor(values, dtype=torch.float32, device=assortments.device)

    def sample_choice(self, rng: random.Random, assortment: Sequence[int]) -> int:
        probs = self.choice_probabilities(assortment).tolist()
        return int(rng.choices(range(len(probs)), weights=probs, k=1)[0])


def build_mccm_model_from_meta(
    meta: Dict[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> MarkovChainChoiceModel:
    initial_probs = meta.get("mccm_initial_probs")
    transition_matrix = meta.get("mccm_transition_matrix")
    rewards = meta.get("rewards")
    if initial_probs is None:
        raise ValueError("Metadata does not contain 'mccm_initial_probs'.")
    if transition_matrix is None:
        raise ValueError("Metadata does not contain 'mccm_transition_matrix'.")
    if rewards is None:
        raise ValueError("Metadata does not contain 'rewards'.")
    return MarkovChainChoiceModel(
        initial_probs,
        transition_matrix,
        rewards,
        reward_no_purchase=float(meta.get("reward_no_purchase", 0.0)),
        device=device,
    )


def sample_utilities(
    rng: random.Random, n_products: int, mean: float, std: float
) -> List[float]:
    return [rng.gauss(mean, std) for _ in range(n_products)]


def sample_rewards(
    rng: random.Random, n_products: int, low: float, high: float
) -> List[float]:
    return [rng.uniform(low, high) for _ in range(n_products)]


def sample_assortment(
    rng: random.Random,
    n_products: int,
    policy: str,
    assortment_size: int,
    include_empty_assortment: bool,
) -> List[int]:
    if policy == "uniform_k_subset":
        k = max(0, min(int(assortment_size), n_products))
        if k == 0:
            mask = [0] * n_products
        else:
            idx = rng.sample(range(n_products), k=k)
            mask = [0] * n_products
            for j in idx:
                mask[j] = 1
        if not include_empty_assortment and sum(mask) == 0:
            mask[rng.randrange(n_products)] = 1
        return mask

    if policy == "bernoulli":
        mask = [1 if rng.random() < 0.5 else 0 for _ in range(n_products)]
        if not include_empty_assortment and sum(mask) == 0:
            mask[rng.randrange(n_products)] = 1
        return mask

    raise ValueError(
        f"Unknown assortment policy {policy!r}. "
        "Supported policies: 'uniform_k_subset', 'bernoulli', 'boltzmann'."
    )


def assortment_mask_to_list(mask: int, n_products: int) -> List[int]:
    """Decode a bitmask into a 0/1 assortment vector."""
    return [(mask >> j) & 1 for j in range(n_products)]


def expected_reward_from_mask(
    mask: int,
    n_products: int,
    utilities: List[float],
    rewards: List[float],
    outside_utility: float,
) -> float:
    """Expected reward under MNL for a given assortment bitmask."""
    offered: List[int] = []
    logits = [outside_utility]
    for j in range(n_products):
        if (mask >> j) & 1:
            offered.append(j)
            logits.append(utilities[j])

    if not offered:
        return 0.0

    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    total = sum(exp_logits)

    expected = 0.0
    for pos, product_idx in enumerate(offered, start=1):
        expected += (exp_logits[pos] / total) * rewards[product_idx]
    return float(expected)


def build_boltzmann_sampler(
    n_products: int,
    utilities: List[float],
    rewards: List[float],
    outside_utility: float,
    beta: float,
) -> List[float]:
    """Build cumulative weights for Boltzmann sampling over all assortments."""
    n_total = 1 << n_products
    scores = [
        beta
        * expected_reward_from_mask(mask, n_products, utilities, rewards, outside_utility)
        for mask in range(n_total)
    ]
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    return list(accumulate(weights))


def mnl_choice_probabilities(
    assortment: List[int],
    utilities: List[float],
    outside_utility: float,
) -> List[float]:
    offered = [i for i, bit in enumerate(assortment) if bit]
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


def sample_choice(
    rng: random.Random,
    assortment: List[int],
    utilities: List[float],
    outside_utility: float,
) -> int:
    probs = mnl_choice_probabilities(assortment, utilities, outside_utility)
    return int(rng.choices(range(len(probs)), weights=probs, k=1)[0])


def expected_reward(
    assortment: List[int],
    utilities: List[float],
    rewards: List[float],
    outside_utility: float,
) -> float:
    probs = mnl_choice_probabilities(assortment, utilities, outside_utility)
    return float(sum(p * r for p, r in zip(probs[1:], rewards)))


def batched_expected_reward(
    assortments: torch.Tensor,
    utilities: torch.Tensor,
    rewards: torch.Tensor,
    outside_utility: float,
) -> torch.Tensor:
    """Vectorized expected reward for a batch of assortments."""
    if assortments.dim() == 1:
        assortments = assortments.unsqueeze(0)
    if assortments.dim() != 2:
        raise ValueError("Expected assortments to have shape [N] or [B, N].")

    device = assortments.device
    mask = assortments.to(dtype=torch.bool)
    utilities = utilities.to(device=device, dtype=torch.float32)
    rewards = rewards.to(device=device, dtype=torch.float32)
    reward_vec = torch.cat(
        [torch.zeros(1, dtype=torch.float32, device=device), rewards], dim=0
    )

    product_logits = utilities.unsqueeze(0).expand(mask.shape[0], -1)
    neg_inf = torch.tensor(float("-inf"), device=device)
    product_logits = torch.where(mask, product_logits, neg_inf)
    outside_logits = torch.full(
        (mask.shape[0], 1), outside_utility, dtype=torch.float32, device=device
    )
    logits = torch.cat([outside_logits, product_logits], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return (probs * reward_vec.unsqueeze(0)).sum(dim=-1)


def sample_boltzmann_assortment_exact(
    rng: random.Random,
    n_products: int,
    utilities: List[float],
    rewards: List[float],
    outside_utility: float,
    beta: float,
) -> List[int]:
    n_total = 1 << n_products
    scores = [
        beta
        * expected_reward_from_mask(mask, n_products, utilities, rewards, outside_utility)
        for mask in range(n_total)
    ]
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    mask = rng.choices(range(n_total), weights=weights, k=1)[0]
    return assortment_mask_to_list(mask, n_products)


def sample_boltzmann_assortment_exact_from_fn(
    rng: random.Random,
    n_products: int,
    expected_reward_fn,
    beta: float,
) -> List[int]:
    """Exact Boltzmann sampling using a generic expected-reward callable."""
    n_total = 1 << n_products
    scores = [
        beta * expected_reward_fn(assortment_mask_to_list(mask, n_products))
        for mask in range(n_total)
    ]
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    mask = rng.choices(range(n_total), weights=weights, k=1)[0]
    return assortment_mask_to_list(mask, n_products)


def sample_boltzmann_assortments_gibbs(
    rng: random.Random,
    n_products: int,
    batched_expected_reward_fn,
    beta: float,
    n_samples: int,
    *,
    num_chains: int = 32,
    burn_in_sweeps: int = 100,
    thinning_sweeps: int = 10,
    device: str = "cpu",
) -> List[List[int]]:
    """Sample assortments from P(S) proportional to exp(beta * R(S)) using Gibbs.

    Each sweep updates every product coordinate once in a random order. Because the
    target is a Boltzmann distribution over expected reward, the single-site
    conditional is Bernoulli with logit beta * (R(S_i=1) - R(S_i=0)).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if num_chains <= 0:
        raise ValueError("num_chains must be positive.")
    if burn_in_sweeps < 0 or thinning_sweeps < 0:
        raise ValueError("burn_in_sweeps and thinning_sweeps must be non-negative.")
    if beta < 0.0:
        raise ValueError("boltzmann_beta must be non-negative.")

    device_t = torch.device(device)
    num_chains = min(int(num_chains), n_samples)
    beta_t = torch.tensor(float(beta), dtype=torch.float32, device=device_t)

    states = torch.randint(
        low=0,
        high=2,
        size=(num_chains, n_products),
        device=device_t,
        dtype=torch.float32,
    )

    def sweep(current: torch.Tensor) -> torch.Tensor:
        order = list(range(n_products))
        rng.shuffle(order)
        for j in order:
            state0 = current.clone()
            state0[:, j] = 0.0
            state1 = current.clone()
            state1[:, j] = 1.0
            reward0 = batched_expected_reward_fn(state0)
            reward1 = batched_expected_reward_fn(state1)
            probs = torch.sigmoid(beta_t * (reward1 - reward0))
            current[:, j] = torch.bernoulli(probs)
        return current

    samples: List[List[int]] = []
    if burn_in_sweeps > 0:
        burn_in_step = max(1, burn_in_sweeps // 100)
        for sweep_idx in range(burn_in_sweeps):
            states = sweep(states)
            if (
                (sweep_idx + 1) % burn_in_step == 0
                or sweep_idx + 1 == burn_in_sweeps
            ):
                _emit_progress("Boltzmann burn-in", sweep_idx + 1, burn_in_sweeps)

    sample_step = max(1, n_samples // 100)
    while len(samples) < n_samples:
        for _ in range(thinning_sweeps):
            states = sweep(states)
        for row in states.detach().cpu().tolist():
            samples.append([int(v) for v in row])
            if len(samples) % sample_step == 0 or len(samples) == n_samples:
                _emit_progress("Boltzmann samples", len(samples), n_samples)
            if len(samples) >= n_samples:
                break

    return samples


def generate_dataset(config: ChoiceModelConfig) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    utilities = sample_utilities(
        rng, config.n_products, config.utility_mean, config.utility_std
    )
    rewards = sample_rewards(
        rng, config.n_products, config.reward_low, config.reward_high
    )
    mccm_params: MCCMParameters | None = None
    mccm_model: MarkovChainChoiceModel | None = None
    if config.choice_model == "mccm":
        mccm_params = sample_mccm_parameters(
            rng,
            config.n_products,
            concentration=config.mccm_concentration,
        )
        mccm_model = MarkovChainChoiceModel(
            mccm_params.initial_probs,
            mccm_params.transition_matrix,
            rewards,
            reward_no_purchase=0.0,
        )

    data: List[Dict[str, Any]] = []
    boltzmann_samples: List[List[int]] = []
    if config.choice_model == "mnl":
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        utilities_t = torch.tensor(utilities, dtype=torch.float32)

        def batched_reward_fn(assortments: torch.Tensor) -> torch.Tensor:
            return batched_expected_reward(
                assortments,
                utilities_t,
                rewards_t,
                config.outside_utility,
            )

        def sample_choice_fn(assortment: List[int]) -> int:
            return sample_choice(
                rng,
                assortment,
                utilities,
                config.outside_utility,
            )

        def exact_boltzmann_fn() -> List[int]:
            return sample_boltzmann_assortment_exact(
                rng,
                config.n_products,
                utilities,
                rewards,
                config.outside_utility,
                config.boltzmann_beta,
            )

    elif config.choice_model == "mccm":
        assert mccm_model is not None

        def batched_reward_fn(assortments: torch.Tensor) -> torch.Tensor:
            return mccm_model.batched_expected_reward(assortments)

        def sample_choice_fn(assortment: List[int]) -> int:
            return mccm_model.sample_choice(rng, assortment)

        def exact_boltzmann_fn() -> List[int]:
            return sample_boltzmann_assortment_exact_from_fn(
                rng,
                config.n_products,
                mccm_model.expected_reward,
                config.boltzmann_beta,
            )

    else:
        raise ValueError(
            f"Unknown choice_model {config.choice_model!r}. Supported models: 'mnl', 'mccm'."
        )

    show_loop_progress = not (
        config.assortment_policy == "boltzmann"
        and config.boltzmann_sampler == "mcmc"
    )
    loop_step = max(1, config.n_samples // 100) if show_loop_progress else 0
    for sample_idx in range(config.n_samples):
        if config.assortment_policy == "boltzmann":
            if config.boltzmann_sampler == "exact":
                assortment = exact_boltzmann_fn()
            elif config.boltzmann_sampler == "mcmc":
                if not boltzmann_samples:
                    boltzmann_samples = sample_boltzmann_assortments_gibbs(
                        rng,
                        config.n_products,
                        batched_reward_fn,
                        config.boltzmann_beta,
                        config.n_samples,
                        num_chains=config.mcmc_chains,
                        burn_in_sweeps=config.mcmc_burn_in,
                        thinning_sweeps=config.mcmc_thinning,
                        device=config.mcmc_device,
                    )
                assortment = boltzmann_samples.pop(0)
            else:
                raise ValueError(
                    f"Unknown boltzmann_sampler {config.boltzmann_sampler!r}."
                )
        else:
                assortment = sample_assortment(
                    rng,
                    config.n_products,
                    config.assortment_policy,
                    config.assortment_size,
                    config.include_empty_assortment,
                )
        choice = sample_choice_fn(assortment)
        data.append({"S": assortment, "A": choice})
        if show_loop_progress and (
            (sample_idx + 1) % loop_step == 0 or sample_idx + 1 == config.n_samples
        ):
            _emit_progress("Dataset samples", sample_idx + 1, config.n_samples)

    meta = {
        "N": config.n_products,
        "choice_model": config.choice_model,
        "rewards": rewards,
        "reward_no_purchase": 0.0,
        "assortment_policy": {
            "type": config.assortment_policy,
            "boltzmann_sampler": config.boltzmann_sampler,
            "assortment_size": config.assortment_size,
            "include_empty_assortment": config.include_empty_assortment,
            "boltzmann_beta": config.boltzmann_beta,
            "mcmc_chains": config.mcmc_chains,
            "mcmc_burn_in": config.mcmc_burn_in,
            "mcmc_thinning": config.mcmc_thinning,
            "mcmc_device": config.mcmc_device,
        },
        "seed": config.seed,
        "n_samples": config.n_samples,
    }

    if config.choice_model == "mnl":
        meta["utilities"] = utilities
        meta["outside_utility"] = config.outside_utility
    else:
        assert mccm_params is not None
        meta["mccm_initial_probs"] = mccm_params.initial_probs
        meta["mccm_transition_matrix"] = mccm_params.transition_matrix
        meta["mccm_concentration"] = config.mccm_concentration

    return {"meta": meta, "data": data}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic assortment-choice data under an MNL or MCCM model."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--n-products", type=int, default=20, help="Number of products.")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of offline pairs.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--choice-model",
        type=str,
        default="mnl",
        choices=("mnl", "mccm"),
        help="Choice model used to sample customer actions.",
    )
    parser.add_argument(
        "--assortment-policy",
        type=str,
        default="uniform_k_subset",
        choices=("uniform_k_subset", "bernoulli", "boltzmann"),
        help="How to sample assortments.",
    )
    parser.add_argument(
        "--assortment-size",
        type=int,
        default=None,
        help="Subset size for uniform_k_subset assortments; defaults to min(5, N).",
    )
    parser.add_argument(
        "--include-empty-assortment",
        action="store_true",
        help="Allow empty assortments when using the assortment policy.",
    )
    parser.add_argument(
        "--boltzmann-beta",
        type=float,
        default=1.0,
        help="Inverse temperature for Boltzmann assortment sampling: P(S) propto exp(beta * R(S)).",
    )
    parser.add_argument(
        "--boltzmann-sampler",
        type=str,
        default="auto",
        choices=("auto", "exact", "mcmc"),
        help="How to sample Boltzmann assortments. 'auto' uses exact enumeration for small N and MCMC otherwise.",
    )
    parser.add_argument(
        "--mcmc-chains",
        type=int,
        default=32,
        help="Number of parallel Gibbs chains used when --boltzmann-sampler mcmc is selected.",
    )
    parser.add_argument(
        "--mcmc-burn-in",
        type=int,
        default=100,
        help="Number of Gibbs sweeps used for MCMC burn-in.",
    )
    parser.add_argument(
        "--mcmc-thinning",
        type=int,
        default=10,
        help="Number of Gibbs sweeps between collected MCMC samples.",
    )
    parser.add_argument(
        "--mcmc-device",
        type=str,
        default="cpu",
        help="Torch device for MCMC sampling, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--mccm-concentration",
        type=float,
        default=1.0,
        help="Dirichlet concentration used to sample MCCM initial and transition probabilities.",
    )
    parser.add_argument(
        "--utility-mean",
        type=float,
        default=0.0,
        help="Mean of the Gaussian product utilities.",
    )
    parser.add_argument(
        "--utility-std",
        type=float,
        default=1.0,
        help="Std. dev. of the Gaussian product utilities.",
    )
    parser.add_argument(
        "--reward-low",
        type=float,
        default=0.0,
        help="Lower bound for uniform reward sampling.",
    )
    parser.add_argument(
        "--reward-high",
        type=float,
        default=1.0,
        help="Upper bound for uniform reward sampling.",
    )
    parser.add_argument(
        "--outside-utility",
        type=float,
        default=0.0,
        help="Utility of the no-purchase option.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.n_products <= 0:
        raise ValueError("--n-products must be positive.")
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")
    if args.choice_model not in ("mnl", "mccm"):
        raise ValueError("--choice-model must be one of: mnl, mccm.")
    if args.assortment_policy == "uniform_k_subset":
        if args.assortment_size is None:
            args.assortment_size = min(5, args.n_products)
        if args.assortment_size < 0:
            raise ValueError("--assortment-size must be non-negative.")
        if args.assortment_size > args.n_products:
            raise ValueError("--assortment-size cannot exceed --n-products.")
    elif args.assortment_policy == "boltzmann":
        if args.assortment_size is None:
            args.assortment_size = 0
        if args.boltzmann_sampler == "auto":
            args.boltzmann_sampler = "exact" if args.n_products <= 20 else "mcmc"
    if args.reward_high <= args.reward_low:
        raise ValueError("--reward-high must be greater than --reward-low.")
    if args.choice_model == "mccm" and args.mccm_concentration <= 0:
        raise ValueError("--mccm-concentration must be positive.")
    if args.boltzmann_sampler == "exact" and args.n_products > 24:
        raise ValueError(
            "Exact Boltzmann enumeration is not practical for this N; use --boltzmann-sampler mcmc."
        )
    if args.mcmc_chains <= 0:
        raise ValueError("--mcmc-chains must be positive.")
    if args.mcmc_burn_in < 0:
        raise ValueError("--mcmc-burn-in must be non-negative.")
    if args.mcmc_thinning < 0:
        raise ValueError("--mcmc-thinning must be non-negative.")

    config = ChoiceModelConfig(
        n_products=args.n_products,
        n_samples=args.n_samples,
        seed=args.seed,
        choice_model=args.choice_model,
        assortment_policy=args.assortment_policy,
        boltzmann_sampler=args.boltzmann_sampler,
        assortment_size=args.assortment_size,
        include_empty_assortment=bool(args.include_empty_assortment),
        boltzmann_beta=args.boltzmann_beta,
        mcmc_chains=args.mcmc_chains,
        mcmc_burn_in=args.mcmc_burn_in,
        mcmc_thinning=args.mcmc_thinning,
        mcmc_device=args.mcmc_device,
        mccm_concentration=args.mccm_concentration,
        utility_mean=args.utility_mean,
        utility_std=args.utility_std,
        reward_low=args.reward_low,
        reward_high=args.reward_high,
        outside_utility=args.outside_utility,
    )
    dataset = generate_dataset(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
