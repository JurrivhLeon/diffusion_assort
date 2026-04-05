#!/usr/bin/env python3
"""Markov chain choice model utilities.

The choice model follows the standard absorbing Markov chain formulation:

* state ``0`` is the no-purchase state and is absorbing
* each product ``j`` corresponds to state ``j``
* for an offered assortment, offered products are made absorbing
* choice probabilities are the absorption probabilities of the resulting chain

This module keeps the linear algebra in one place so both data generation and
evaluation can reuse the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import torch


def _sample_dirichlet(
    rng: random.Random,
    concentration: float,
    size: int,
) -> list[float]:
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
    """Parameters for the Markov chain choice model."""

    initial_probs: list[float]
    transition_matrix: list[list[float]]


def sample_mccm_parameters(
    rng: random.Random,
    n_products: int,
    *,
    concentration: float = 1.0,
) -> MCCMParameters:
    """Sample a random Markov chain choice model.

    The returned model uses state ``0`` as the no-purchase state. State ``0`` is
    absorbing, while every product state transitions to every other state except
    itself with positive probability.
    """
    if n_products <= 0:
        raise ValueError("n_products must be positive.")

    n_states = n_products + 1
    initial_probs = _sample_dirichlet(rng, concentration, n_states)

    transition_matrix: list[list[float]] = [[0.0] * n_states for _ in range(n_states)]
    transition_matrix[0][0] = 1.0

    for i in range(1, n_states):
        row_without_self = _sample_dirichlet(rng, concentration, n_states - 1)
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
    """Absorbing Markov chain choice model."""

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

    def _assortment_to_states(self, assortment: Sequence[int]) -> tuple[list[int], list[int]]:
        if len(assortment) != self.n_products:
            raise ValueError(
                f"Expected assortment length {self.n_products}, got {len(assortment)}."
            )
        offered = [idx + 1 for idx, bit in enumerate(assortment) if int(bit)]
        absorbing = [0] + offered
        return offered, absorbing

    def choice_probabilities(self, assortment: Sequence[int]) -> torch.Tensor:
        """Return the full distribution over choices 0..N."""
        _, absorbing = self._assortment_to_states(assortment)
        n_states = self.n_products + 1
        absorbing_set = set(absorbing)
        transient = [state for state in range(n_states) if state not in absorbing_set]

        if not transient:
            probs = torch.zeros(n_states, dtype=self.dtype, device=self.device)
            probs[absorbing] = self.initial_probs[absorbing]
            return probs

        transient_idx = torch.tensor(transient, dtype=torch.long, device=self.device)
        absorbing_idx = torch.tensor(absorbing, dtype=torch.long, device=self.device)
        q = self.transition_matrix.index_select(0, transient_idx).index_select(1, transient_idx)
        r = self.transition_matrix.index_select(0, transient_idx).index_select(1, absorbing_idx)
        eye = torch.eye(len(transient), dtype=self.dtype, device=self.device)
        fundamental = torch.linalg.solve(eye - q, r)
        absorbed_from_transient = self.initial_probs.index_select(0, transient_idx) @ fundamental

        probs = torch.zeros(n_states, dtype=self.dtype, device=self.device)
        probs[absorbing_idx] = self.initial_probs.index_select(0, absorbing_idx) + absorbed_from_transient
        return probs

    def expected_reward(self, assortment: Sequence[int]) -> float:
        probs = self.choice_probabilities(assortment)
        return float(torch.dot(probs, self.reward_vector).item())

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

