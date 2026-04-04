#!/usr/bin/env python3
"""PyTorch dataset loader for assortment-choice JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


def load_assortment_json(path: str | Path) -> Dict[str, Any]:
    """Load the raw JSON structure produced by generate_assortment_data.py."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class AssortmentChoiceDataset(Dataset):
    """Dataset of offline (S, A) assortment-choice pairs.

    Each item is returned as a dict with:
      - ``S``: float tensor of shape [N], values in {0, 1}
      - ``A``: long tensor scalar in {0, ..., N}
    """

    def __init__(
        self,
        path: str | Path,
        *,
        store_meta: bool = True,
    ) -> None:
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

        assortments: List[torch.Tensor] = []
        choices: List[torch.Tensor] = []
        for idx, row in enumerate(raw["data"]):
            if not isinstance(row, dict):
                raise ValueError(f"Example {idx} is not an object.")
            if "S" not in row or "A" not in row:
                raise ValueError(f"Example {idx} must contain 'S' and 'A'.")
            S = row["S"]
            A = row["A"]
            if not isinstance(S, list):
                raise ValueError(f"Example {idx} has non-list assortment.")
            if len(S) != self.n_products:
                raise ValueError(
                    f"Example {idx} assortment length {len(S)} does not match N={self.n_products}."
                )
            if any(int(x) not in (0, 1) for x in S):
                raise ValueError(f"Example {idx} assortment must be binary.")
            if not isinstance(A, int):
                raise ValueError(f"Example {idx} choice must be an integer.")
            if A < 0 or A > self.n_products:
                raise ValueError(f"Example {idx} choice out of range.")
            assortments.append(torch.tensor(S, dtype=torch.float32))
            choices.append(torch.tensor(A, dtype=torch.long))

        self.assortments = assortments
        self.choices = choices

    def __len__(self) -> int:
        return len(self.choices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"S": self.assortments[index], "A": self.choices[index]}


def batch_from_json(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load all assortments/choices into a single tensor pair."""
    ds = AssortmentChoiceDataset(path)
    return torch.stack(ds.assortments, dim=0), torch.stack(ds.choices, dim=0)
