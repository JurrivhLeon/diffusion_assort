#!/usr/bin/env python3
"""PyTorch implementation of the feature-free neural choice models."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _ensure_batch(S: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if S.dim() == 1:
        return S.unsqueeze(0), True
    if S.dim() != 2:
        raise ValueError("Expected S to have shape [N] or [B, N].")
    return S, False


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return F.softmax(masked_logits, dim=-1)


class GAsN(nn.Module):
    """Gated-Assort-Net from Section 4.1.

    Input:
      S in {0,1}^N
    Output:
      Probability distribution over {0,1,...,N}

    The no-purchase option (index 0) is always available. Product j>0 has
    probability zero whenever S[j-1] = 0.
    """

    def __init__(
        self,
        n_products: int,
        hidden_sizes: Sequence[int] | None = None,
        *,
        learnable_outside_logit: bool = True,
    ) -> None:
        super().__init__()
        if n_products <= 0:
            raise ValueError("n_products must be positive.")
        self.n_products = int(n_products)

        if hidden_sizes is None:
            hidden_sizes = (n_products,)
        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")

        layers: List[nn.Module] = []
        in_dim = self.n_products
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.n_products))
        self.layers = nn.ModuleList(layers)
        self.learnable_outside_logit = bool(learnable_outside_logit)
        if self.learnable_outside_logit:
            self.outside_logit = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("outside_logit", torch.zeros(()))

    def logits(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        x = S.float()
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = F.relu(self.layers[-1](x))
        return x.squeeze(0) if squeezed else x

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        product_logits = self.logits(S)
        if product_logits.dim() == 1:
            product_logits = product_logits.unsqueeze(0)
        outside = self.outside_logit.expand(product_logits.shape[0], 1)
        logits = torch.cat([outside, product_logits], dim=-1)
        mask = torch.cat(
            [torch.ones_like(S[:, :1], dtype=torch.bool), S.to(dtype=torch.bool)],
            dim=-1,
        )
        probs = _masked_softmax(logits, mask)
        return probs.squeeze(0) if squeezed else probs

    def log_prob(self, S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        probs = self.forward(S)
        A = A.long()
        if probs.dim() == 1:
            return torch.log(probs[A].clamp_min(1e-12))
        if A.dim() == 0:
            A = A.unsqueeze(0)
        return torch.log(probs.gather(-1, A.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))

    def negative_log_likelihood(
        self,
        S: torch.Tensor,
        A: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logp = self.log_prob(S, A)
        nll = -logp
        if reduction == "none":
            return nll
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError("reduction must be one of: none, mean, sum.")


class MNL(nn.Module):
    """Feature-free multinomial logit with learnable product utilities."""

    def __init__(
        self,
        n_products: int,
        *,
        learnable_outside_logit: bool = True,
    ) -> None:
        super().__init__()
        if n_products <= 0:
            raise ValueError("n_products must be positive.")
        self.n_products = int(n_products)
        self.product_utilities = nn.Parameter(torch.zeros(self.n_products))
        self.learnable_outside_logit = bool(learnable_outside_logit)
        if self.learnable_outside_logit:
            self.outside_logit = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("outside_logit", torch.zeros(()))

    def logits(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        logits = self.product_utilities.unsqueeze(0).expand(S.shape[0], -1)
        return logits.squeeze(0) if squeezed else logits

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        product_logits = self.logits(S)
        if product_logits.dim() == 1:
            product_logits = product_logits.unsqueeze(0)
        outside = self.outside_logit.expand(product_logits.shape[0], 1)
        logits = torch.cat([outside, product_logits], dim=-1)
        mask = torch.cat(
            [torch.ones_like(S[:, :1], dtype=torch.bool), S.to(dtype=torch.bool)],
            dim=-1,
        )
        probs = _masked_softmax(logits, mask)
        return probs.squeeze(0) if squeezed else probs

    def log_prob(self, S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        probs = self.forward(S)
        A = A.long()
        if probs.dim() == 1:
            return torch.log(probs[A].clamp_min(1e-12))
        if A.dim() == 0:
            A = A.unsqueeze(0)
        return torch.log(probs.gather(-1, A.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))

    def negative_log_likelihood(
        self,
        S: torch.Tensor,
        A: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logp = self.log_prob(S, A)
        nll = -logp
        if reduction == "none":
            return nll
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError("reduction must be one of: none, mean, sum.")


class RAsN(nn.Module):
    """Res-Assort-Net from Section 4.1.

    Each block has the residual form:
        z^l = ReLU(W^l z^{l-1} + b^l) + z^{l-1}
    """

    def __init__(
        self,
        n_products: int,
        num_blocks: int = 1,
        *,
        learnable_outside_logit: bool = True,
    ) -> None:
        super().__init__()
        if n_products <= 0:
            raise ValueError("n_products must be positive.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        self.n_products = int(n_products)
        self.num_blocks = int(num_blocks)

        self.blocks = nn.ModuleList(
            [nn.Linear(self.n_products, self.n_products) for _ in range(self.num_blocks)]
        )
        self.learnable_outside_logit = bool(learnable_outside_logit)
        if self.learnable_outside_logit:
            self.outside_logit = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("outside_logit", torch.zeros(()))

    def logits(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        x = S.float()
        for block in self.blocks:
            x = F.relu(block(x)) + x
        return x.squeeze(0) if squeezed else x

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        product_logits = self.logits(S)
        if product_logits.dim() == 1:
            product_logits = product_logits.unsqueeze(0)
        outside = self.outside_logit.expand(product_logits.shape[0], 1)
        logits = torch.cat([outside, product_logits], dim=-1)
        mask = torch.cat(
            [torch.ones_like(S[:, :1], dtype=torch.bool), S.to(dtype=torch.bool)],
            dim=-1,
        )
        probs = _masked_softmax(logits, mask)
        return probs.squeeze(0) if squeezed else probs

    def log_prob(self, S: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        probs = self.forward(S)
        A = A.long()
        if probs.dim() == 1:
            return torch.log(probs[A].clamp_min(1e-12))
        if A.dim() == 0:
            A = A.unsqueeze(0)
        return torch.log(probs.gather(-1, A.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))

    def negative_log_likelihood(
        self,
        S: torch.Tensor,
        A: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logp = self.log_prob(S, A)
        nll = -logp
        if reduction == "none":
            return nll
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError("reduction must be one of: none, mean, sum.")


if __name__ == "__main__":
    # Tiny smoke test for manual runs.
    model = GAsN(n_products=5)
    print(model(torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)))
