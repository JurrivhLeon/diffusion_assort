#!/usr/bin/env python3
"""Discrete Bernoulli diffusion model for assortment states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffusionSchedule:
    num_steps: int
    beta_start: float
    beta_end: float


def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 1:
        return x.unsqueeze(0), True
    if x.dim() != 2:
        raise ValueError("Expected x to have shape [N] or [B, N].")
    return x, False


def _score_in_chunks(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    candidates: torch.Tensor,
    *,
    chunk_size: int | None,
) -> torch.Tensor:
    if chunk_size is None or candidates.shape[0] <= chunk_size:
        scores = score_fn(candidates)
        if scores.dim() != 1 or scores.shape[0] != candidates.shape[0]:
            raise ValueError("score_fn must return a 1D tensor aligned with candidates.")
        return scores

    chunks: list[torch.Tensor] = []
    for start in range(0, candidates.shape[0], chunk_size):
        end = min(start + chunk_size, candidates.shape[0])
        scores = score_fn(candidates[start:end])
        if scores.dim() != 1 or scores.shape[0] != end - start:
            raise ValueError("score_fn must return a 1D tensor aligned with candidates.")
        chunks.append(scores)
    return torch.cat(chunks, dim=0)


class BernoulliDiffusion(nn.Module):
    """Fully factorized Bernoulli diffusion model.

    The model takes a noisy assortment x_t and a timestep t, and outputs
    logits for the clean assortment bits.
    """

    def __init__(
        self,
        n_products: int,
        *,
        num_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        hidden_sizes: Sequence[int] | None = None,
        time_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        if n_products <= 0:
            raise ValueError("n_products must be positive.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if beta_start <= 0.0 or beta_end <= 0.0:
            raise ValueError("beta_start and beta_end must be positive.")
        if beta_end >= 1.0:
            raise ValueError("beta_end must be smaller than 1.")
        if time_emb_dim <= 0:
            raise ValueError("time_emb_dim must be positive.")

        self.n_products = int(n_products)
        self.num_steps = int(num_steps)
        self.schedule = DiffusionSchedule(
            num_steps=self.num_steps, beta_start=float(beta_start), beta_end=float(beta_end)
        )
        self.time_emb_dim = int(time_emb_dim)

        if hidden_sizes is None:
            hidden_sizes = (128, 128)
        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")

        self.time_embedding = nn.Embedding(self.num_steps + 1, self.time_emb_dim)

        layers: list[nn.Module] = []
        in_dim = self.n_products + self.time_emb_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.n_products))
        self.network = nn.Sequential(*layers)

        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bar)

    def _check_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        x, squeezed = _ensure_batch(x)
        if x.shape[-1] != self.n_products:
            raise ValueError(f"Expected last dimension {self.n_products}, got {x.shape[-1]}.")
        return x.float(), squeezed

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() != 1:
            raise ValueError("t must have shape [] or [B].")
        if torch.any(t < 1) or torch.any(t > self.num_steps):
            raise ValueError(f"Timesteps must be in [1, {self.num_steps}].")
        return self.time_embedding(t.long())

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) under the Bernoulli corruption process."""
        x0, squeezed = self._check_input(x0)
        if t.dim() == 0:
            t = t.expand(x0.shape[0])
        if t.shape[0] != x0.shape[0]:
            raise ValueError("Batch size of x0 and t must match.")
        alpha_bar = self.alpha_bars[t.long() - 1].unsqueeze(-1)
        probs = alpha_bar * x0 + (1.0 - alpha_bar) * 0.5
        xt = torch.bernoulli(probs)
        return xt.squeeze(0) if squeezed else xt

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return logits for each assortment bit."""
        x_t, squeezed = self._check_input(x_t)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        if t.shape[0] != x_t.shape[0]:
            raise ValueError("Batch size of x_t and t must match.")
        h = torch.cat([x_t, self._time_embedding(t)], dim=-1)
        logits = self.network(h)
        return logits.squeeze(0) if squeezed else logits

    def denoising_loss(
        self,
        x0: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        x0, _ = self._check_input(x0)
        batch_size = x0.shape[0]
        device = x0.device
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (batch_size,), device=device)
        else:
            if t.dim() == 0:
                t = t.expand(batch_size)
        xt = self.corrupt(x0, t)
        logits = self.forward(xt, t)
        loss = F.binary_cross_entropy_with_logits(logits, x0, reduction=reduction)
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
        device = torch.device(device)
        x = torch.bernoulli(torch.full((num_samples, self.n_products), 0.5, device=device))
        for step in range(self.num_steps, 0, -1):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            logits = self.forward(x, t)
            probs = torch.sigmoid(logits)
            x = torch.bernoulli(probs)
        return x

    def _guided_scale(
        self,
        step: int,
        guidance_scale: float,
        guidance_gamma: float,
        guidance_schedule: Sequence[float] | None,
    ) -> float:
        if guidance_schedule is None:
            if self.num_steps == 1:
                return float(guidance_scale)
            if guidance_gamma < 1.0:
                raise ValueError("guidance_gamma must be at least 1.0.")
            frac = 1.0 - float(step - 1) / float(self.num_steps - 1)
            return float(guidance_scale) * (frac ** float(guidance_gamma))
        if len(guidance_schedule) != self.num_steps:
            raise ValueError(
                f"guidance_schedule must have length {self.num_steps}, got {len(guidance_schedule)}."
            )
        return float(guidance_scale) * float(guidance_schedule[step - 1])

    def _bit_value_candidates(self, x: torch.Tensor) -> torch.Tensor:
        """Return paired candidates with each coordinate forced to 0 and 1.

        Output shape:
            [B, N, 2, N]
        where candidates[b, i, 0] has bit i set to 0 and candidates[b, i, 1]
        has bit i set to 1.
        """
        x, squeezed = _ensure_batch(x)
        if x.shape[-1] != self.n_products:
            raise ValueError(f"Expected last dimension {self.n_products}, got {x.shape[-1]}.")
        candidates = x.unsqueeze(1).unsqueeze(2).expand(-1, self.n_products, 2, -1).clone()
        idx = torch.arange(self.n_products, device=x.device)
        candidates[:, idx, 0, idx] = 0.0
        candidates[:, idx, 1, idx] = 1.0
        return candidates.squeeze(0) if squeezed else candidates

    @torch.no_grad()
    def guided_reverse_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        guidance_scale: float = 1.0,
        guidance_gamma: float = 1.0,
        guidance_schedule: Sequence[float] | None = None,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """One batched guided reverse step.

        `score_fn` must accept a batch of assortments of shape [B, N] and return
        a 1D tensor of expected rewards or energies.
        """
        x_t, squeezed = self._check_input(x_t)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        if t.shape[0] != x_t.shape[0]:
            raise ValueError("Batch size of x_t and t must match.")

        base_logits = self.forward(x_t, t)
        if base_logits.dim() != 2 or base_logits.shape[0] != x_t.shape[0]:
            raise ValueError("forward() must return shape [B, N] for a batch of inputs.")

        candidates = self._bit_value_candidates(x_t)
        flat_candidates = candidates.reshape(-1, self.n_products)
        candidate_scores = _score_in_chunks(
            score_fn, flat_candidates, chunk_size=candidate_chunk_size
        ).reshape(x_t.shape[0], self.n_products, 2)
        score_zero = candidate_scores[..., 0]
        score_one = candidate_scores[..., 1]
        if score_zero.dim() != 2 or score_zero.shape != (x_t.shape[0], self.n_products):
            raise ValueError("score_fn must return shape [B] for a batch of inputs.")

        scale = self._guided_scale(
            int(t[0].item()),
            guidance_scale,
            guidance_gamma,
            guidance_schedule,
        )
        guided_logits = base_logits + scale * (score_one - score_zero)
        probs = torch.sigmoid(guided_logits)
        x_prev = torch.bernoulli(probs)
        return x_prev.squeeze(0) if squeezed else x_prev

    @torch.no_grad()
    def guided_sample(
        self,
        num_samples: int,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        guidance_scale: float = 1.0,
        guidance_gamma: float = 1.0,
        guidance_schedule: Sequence[float] | None = None,
        candidate_chunk_size: int | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Sample multiple assortments with guided reverse diffusion."""
        device = torch.device(device)
        x = torch.bernoulli(torch.full((num_samples, self.n_products), 0.5, device=device))
        for step in range(self.num_steps, 0, -1):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            x = self.guided_reverse_step(
                x,
                t,
                score_fn,
                guidance_scale=guidance_scale,
                guidance_gamma=guidance_gamma,
                guidance_schedule=guidance_schedule,
                candidate_chunk_size=candidate_chunk_size,
            )
        return x
