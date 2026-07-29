from __future__ import annotations

import torch
from torch import Tensor


def normalized_cosine_similarity(a: Tensor, b: Tensor, delta: float = 1e-6) -> Tensor:
    """Equation (2): cosine similarity with the manuscript's stabilizer."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    numerator = (a.float() * b.float()).sum(dim=-1)
    denominator = a.float().norm(dim=-1) * b.float().norm(dim=-1) + delta
    return numerator / denominator


def normalized_l2_error(candidate: Tensor, reference: Tensor, delta: float = 1e-6) -> Tensor:
    """Return ||candidate-reference||_2 / (||reference||_2 + delta)."""
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(candidate.shape)} != {tuple(reference.shape)}")
    numerator = (candidate.float() - reference.float()).norm(dim=-1)
    denominator = reference.float().norm(dim=-1) + delta
    return numerator / denominator


def normalized_probabilities(values: Tensor, eps: float = 1e-12) -> Tensor:
    """Validate and normalize non-negative probability-like values."""
    values = values.float()
    if torch.any(values < 0):
        raise ValueError("probabilities must be non-negative")
    denominator = values.sum(dim=-1, keepdim=True)
    if torch.any(denominator <= eps):
        raise ValueError("probability rows must have positive mass")
    return values / denominator
