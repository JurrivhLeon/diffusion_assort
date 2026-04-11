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


class RAsN(nn.Module):
    """Res-Assort-Net from Section 4.1.

    In legacy mode, each block has the residual form:
        z^l = ReLU(W^l z^{l-1} + b^l) + z^{l-1}

    In the newer hidden-layer mode, each block is an MLP with residual skip:
        z^l = f^l(z^{l-1}) + z^{l-1}
    where f^l is a feed-forward network with hidden widths `hidden_sizes`.
    """

    def __init__(
        self,
        n_products: int,
        hidden_sizes: Sequence[int] | None = None,
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

        self.hidden_sizes = None if hidden_sizes is None else tuple(int(h) for h in hidden_sizes)
        if self.hidden_sizes is not None and any(h <= 0 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")

        if self.hidden_sizes is None:
            self.blocks = nn.ModuleList(
                [nn.Linear(self.n_products, self.n_products) for _ in range(self.num_blocks)]
            )
        else:
            blocks: List[nn.Module] = []
            for _ in range(self.num_blocks):
                layers: List[nn.Module] = []
                in_dim = self.n_products
                for hidden in self.hidden_sizes:
                    layers.append(nn.Linear(in_dim, hidden))
                    layers.append(nn.ReLU())
                    in_dim = hidden
                layers.append(nn.Linear(in_dim, self.n_products))
                blocks.append(nn.Sequential(*layers))
            self.blocks = nn.ModuleList(blocks)
        self.learnable_outside_logit = bool(learnable_outside_logit)
        if self.learnable_outside_logit:
            self.outside_logit = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("outside_logit", torch.zeros(()))

    def logits(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        x = S.float()
        if self.hidden_sizes is None:
            for block in self.blocks:
                x = F.relu(block(x)) + x
        else:
            for block in self.blocks:
                x = block(x) + x
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


class _TCNetEncoderBlock(nn.Module):
    """Self-attention + feed-forward block used by TCNet."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_out))
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class TCNet(nn.Module):
    """Single-choice Transformer Choice Net.

    The model follows the paper's single-choice reduction by treating the
    offered assortment as both the assortment set and the candidate set.
    Product identities are represented by learned embeddings, contextualized
    with assortment-masked self-attention, and mapped to per-item utilities
    through a lightweight decoder.
    """

    def __init__(
        self,
        n_products: int,
        embedding_dim: int | None = None,
        num_layers: int = 1,
        num_heads: int = 1,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        *,
        learnable_outside_logit: bool = True,
    ) -> None:
        super().__init__()
        if n_products <= 0:
            raise ValueError("n_products must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.n_products = int(n_products)
        self.embedding_dim = int(self.n_products if embedding_dim is None else embedding_dim)
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(4 * self.embedding_dim if ffn_dim is None else ffn_dim)
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive.")

        self.dropout = float(dropout)
        self.item_embedding = nn.Embedding(self.n_products, self.embedding_dim)
        self.assortment_encoder = nn.ModuleList(
            [
                _TCNetEncoderBlock(
                    dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.candidate_encoder = nn.ModuleList(
            [
                _TCNetEncoderBlock(
                    dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.embedding_dim)
        self.cross_dropout = nn.Dropout(self.dropout)
        self.utility_decoder = nn.Linear(self.embedding_dim, 1)

        self.learnable_outside_logit = bool(learnable_outside_logit)
        if self.learnable_outside_logit:
            self.outside_logit = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("outside_logit", torch.zeros(()))

    def _encode_stack(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        active = mask.any(dim=-1)
        encoded = x.new_zeros(x.shape)
        if active.any():
            x_active = x[active]
            padding_mask = ~mask[active]
            for block in blocks:
                x_active = block(x_active, padding_mask)
            encoded[active] = x_active
        return encoded

    def _cross_attention(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        active = mask.any(dim=-1)
        output = query.new_zeros(query.shape)
        if active.any():
            query_active = query[active]
            context_active = context[active]
            padding_mask = ~mask[active]
            attn_out, _ = self.cross_attention(
                query_active,
                context_active,
                context_active,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            query_active = self.cross_norm(query_active + self.cross_dropout(attn_out))
            query_active = query_active.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            output[active] = query_active
        return output

    def logits(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        mask = S.to(dtype=torch.bool)
        batch_size = S.shape[0]
        item_ids = torch.arange(self.n_products, device=S.device).unsqueeze(0).expand(batch_size, -1)
        item_embeddings = self.item_embedding(item_ids)
        item_embeddings = item_embeddings * mask.unsqueeze(-1).to(dtype=item_embeddings.dtype)

        assortment_features = self._encode_stack(item_embeddings, mask, self.assortment_encoder)
        candidate_features = self._encode_stack(item_embeddings, mask, self.candidate_encoder)
        contextual_features = self._cross_attention(candidate_features, assortment_features, mask)

        product_logits = self.utility_decoder(contextual_features).squeeze(-1)
        product_logits = product_logits.masked_fill(~mask, 0.0)
        return product_logits.squeeze(0) if squeezed else product_logits

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        S, squeezed = _ensure_batch(S)
        mask = S.to(dtype=torch.bool)
        product_logits = self.logits(S)
        if product_logits.dim() == 1:
            product_logits = product_logits.unsqueeze(0)
        outside = self.outside_logit.expand(product_logits.shape[0], 1)
        logits = torch.cat([outside, product_logits], dim=-1)
        choice_mask = torch.cat(
            [torch.ones_like(S[:, :1], dtype=torch.bool), mask],
            dim=-1,
        )
        probs = torch.zeros_like(logits)
        active = mask.any(dim=-1)
        if active.any():
            probs[active] = _masked_softmax(logits[active], choice_mask[active])
        if (~active).any():
            probs[~active, 0] = 1.0
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
    # Tiny smoke tests for manual runs.
    gasn = GAsN(n_products=5)
    print(gasn(torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)))

    tcn = TCNet(n_products=5, embedding_dim=8, num_layers=1, num_heads=1)
    sample = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
    empty = torch.zeros(5, dtype=torch.float32)
    sample_probs = tcn(sample)
    empty_probs = tcn(empty)
    assert sample_probs.shape == (6,)
    assert empty_probs.shape == (6,)
    assert torch.isclose(sample_probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(empty_probs.sum(), torch.tensor(1.0), atol=1e-5)
    print(sample_probs)
    print(empty_probs)
