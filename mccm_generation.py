#!/usr/bin/env python3
"""Shared MCCM parameter generators.

This module keeps the random and structured MCCM samplers in one place so the
dataset generator and model utilities stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class MCCMParameters:
    """Parameters for an absorbing Markov chain choice model."""

    initial_probs: list[float]
    transition_matrix: list[list[float]]


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


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must be non-empty.")
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    total = sum(exp_logits)
    if total <= 0.0:
        raise RuntimeError("Failed to normalize softmax logits.")
    return [x / total for x in exp_logits]


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


def sample_alt_mccm_parameters(
    rng: random.Random,
    n_products: int,
) -> MCCMParameters:
    """Sample the structured `alt` MCCM regime described in the screenshot.

    The regime uses contiguous groups of five products, with within-group
    transition logits centered at 2*sigma and cross-group logits centered at 0.
    The no-purchase state remains absorbing in the output schema.
    """
    if n_products <= 0:
        raise ValueError("n_products must be positive.")
    if n_products % 5 != 0:
        raise ValueError("alt MCCM regime requires n_products to be divisible by 5.")

    sigma = n_products / 10.0
    group_size = 5
    n_states = n_products + 1

    initial_logits = [rng.gauss(0.0, sigma) for _ in range(n_products)]
    initial_probs = [0.0] + _softmax(initial_logits)

    transition_matrix: list[list[float]] = [[0.0] * n_states for _ in range(n_states)]
    transition_matrix[0][0] = 1.0

    for i in range(1, n_states):
        row_states: list[int] = []
        row_logits: list[float] = []
        i_group = (i - 1) // group_size
        for j in range(n_states):
            if j == i:
                continue
            mean = 0.0
            if j != 0:
                j_group = (j - 1) // group_size
                if j_group == i_group:
                    mean = 2.0 * sigma
            row_states.append(j)
            row_logits.append(rng.gauss(mean, sigma))
        row_probs = _softmax(row_logits)
        for j, prob in zip(row_states, row_probs):
            transition_matrix[i][j] = prob

    return MCCMParameters(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
    )
