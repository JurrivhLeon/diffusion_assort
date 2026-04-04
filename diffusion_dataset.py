#!/usr/bin/env python3
"""Dataset helpers for discrete diffusion training on assortments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


def load_assortment_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class AssortmentStateDataset(Dataset):
    """Dataset of binary assortment states S only.

    This is the natural training dataset for the Bernoulli diffusion model.
    The original offline data contain (S, A) pairs, but the diffusion model
    only needs the assortment state S.
    """

    def __init__(self, path: str | Path, *, store_meta: bool = True) -> None:
        raw = load_assortment_json(path)
        if "meta" not in raw or "data" not in raw:
            raise ValueError("Expected JSON object with 'meta' and 'data' fields.")
        if not isinstance(raw["data"], list):
            raise ValueError("'data' must be a list of examples.")
        if "N" not in raw["meta"]:
            raise ValueError("Dataset metadata must contain N.")

        self.path = str(path)
        self.meta = raw["meta"] if store_meta else {}
        self.n_products = int(raw["meta"]["N"])

        states: List[torch.Tensor] = []
        for idx, row in enumerate(raw["data"]):
            if not isinstance(row, dict):
                raise ValueError(f"Example {idx} is not an object.")
            if "S" not in row:
                raise ValueError(f"Example {idx} must contain 'S'.")
            S = row["S"]
            if not isinstance(S, list):
                raise ValueError(f"Example {idx} has non-list assortment.")
            if len(S) != self.n_products:
                raise ValueError(
                    f"Example {idx} assortment length {len(S)} does not match N={self.n_products}."
                )
            if any(int(x) not in (0, 1) for x in S):
                raise ValueError(f"Example {idx} assortment must be binary.")
            states.append(torch.tensor(S, dtype=torch.float32))

        self.states = states

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"S": self.states[index]}

