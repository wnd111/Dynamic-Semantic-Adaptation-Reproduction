from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ComplexitySignals:
    entropy: Tensor
    information_gain: Tensor
    dependency_span: Tensor

    def stacked(self) -> Tensor:
        return torch.stack([self.entropy, self.information_gain, self.dependency_span], dim=-1)


class _SignalHead(nn.Module):
    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.network(hidden).squeeze(-1)


class ComplexityPredictor(nn.Module):
    """Equations (10)-(14): three learned signal heads and a fusion network."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        width = max(1, hidden_size // 2)
        dependency_width = max(1, hidden_size // 4)
        self.entropy_head = _SignalHead(hidden_size, width)
        self.information_head = _SignalHead(hidden_size, width)
        self.dependency_head = _SignalHead(hidden_size, dependency_width)
        self.fusion = nn.Sequential(
            nn.Linear(3, max(2, width)),
            nn.GELU(),
            nn.Linear(max(2, width), 1),
            nn.Sigmoid(),
        )

    def predict_signals(self, hidden: Tensor) -> ComplexitySignals:
        if hidden.ndim == 3:
            pooled = hidden.mean(dim=1)
        elif hidden.ndim == 2:
            pooled = hidden
        else:
            raise ValueError("hidden must have shape [batch, hidden] or [batch, sequence, hidden]")
        return ComplexitySignals(
            entropy=self.entropy_head(pooled),
            information_gain=self.information_head(pooled),
            dependency_span=self.dependency_head(pooled),
        )

    def forward(self, hidden: Tensor, targets: ComplexitySignals | None = None) -> Tensor:
        predicted = self.predict_signals(hidden)
        if targets is not None and targets.stacked().shape != predicted.stacked().shape:
            raise ValueError("complexity target shape must match predictor output")
        return self.fusion(predicted.stacked()).squeeze(-1)


@dataclass(frozen=True)
class MaskDecision:
    mode: Literal["full", "window", "topk", "stochastic"]
    indices: Tensor
    total_keys: int
    reason: str = "complexity_policy"

    @classmethod
    def full(
        cls,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
        reason: str = "complexity_policy",
    ) -> MaskDecision:
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        return cls(
            mode="full",
            indices=torch.arange(seq_len, device=device, dtype=torch.long),
            total_keys=seq_len,
            reason=reason,
        )

    @property
    def keep_fraction(self) -> float:
        return float(self.indices.numel()) / float(self.total_keys)


class MaskSelector:
    """Equation (15) with deterministic deployment top-k selection."""

    def __init__(
        self,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
        topk_fraction: float = 0.5,
        window_fraction: float = 0.9,
    ) -> None:
        if not 0.0 <= low_threshold < high_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
        if not 0.0 < topk_fraction <= 1.0 or not 0.0 < window_fraction <= 1.0:
            raise ValueError("mask fractions must be in (0, 1]")
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.topk_fraction = float(topk_fraction)
        self.window_fraction = float(window_fraction)

    @classmethod
    def default(cls) -> MaskSelector:
        return cls()

    @staticmethod
    def importance(mean_attention: Tensor, recency_weight: float = 0.3) -> Tensor:
        if mean_attention.ndim != 1:
            raise ValueError("mean_attention must be one-dimensional")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError("recency_weight must be in [0, 1]")
        scores = (1.0 - recency_weight) * mean_attention.float()
        scores = scores.clone()
        scores[-1] += recency_weight
        return scores

    @staticmethod
    def _include_current(indices: Tensor, importance: Tensor, current: int) -> Tensor:
        if bool(torch.any(indices == current)):
            return indices.sort().values
        if indices.numel() == 1:
            return torch.tensor([current], device=indices.device, dtype=torch.long)
        selected_scores = importance[indices]
        lowest_position = int(torch.argmin(selected_scores).item())
        indices = indices.clone()
        indices[lowest_position] = current
        return torch.unique(indices, sorted=True)

    def select(self, score: Tensor | float, importance: Tensor, seq_len: int) -> MaskDecision:
        if seq_len < 1 or importance.ndim != 1 or importance.numel() < seq_len:
            raise ValueError("importance must contain at least seq_len scalar values")
        scalar = float(torch.as_tensor(score).detach().cpu())
        device = importance.device
        if scalar > self.high_threshold:
            return MaskDecision.full(seq_len, device=device)
        if scalar >= self.low_threshold:
            count = max(1, math.ceil(seq_len * self.window_fraction))
            indices = torch.arange(seq_len - count, seq_len, device=device, dtype=torch.long)
            return MaskDecision("window", indices, seq_len)
        count = max(1, math.ceil(seq_len * self.topk_fraction))
        indices = importance[:seq_len].topk(count, largest=True, sorted=False).indices
        indices = self._include_current(indices, importance[:seq_len], seq_len - 1)
        return MaskDecision("topk", indices, seq_len)

    def select_stochastic(
        self,
        importance: Tensor,
        seq_len: int,
        *,
        generator: torch.Generator,
    ) -> MaskDecision:
        count = max(1, math.ceil(seq_len * self.topk_fraction))
        probabilities = importance[:seq_len].float().clamp_min(0.0)
        if float(probabilities.sum()) == 0.0:
            probabilities = torch.ones_like(probabilities)
        indices = torch.multinomial(
            probabilities,
            num_samples=min(count, seq_len),
            replacement=False,
            generator=generator,
        )
        indices = self._include_current(indices, probabilities, seq_len - 1)
        return MaskDecision("stochastic", indices, seq_len, reason="stochastic_ablation")


class ProgressiveRefresh:
    """Refresh triggers described after Equations (16)-(17)."""

    def __init__(
        self,
        interval: int = 20,
        precision_change_limit: int = 4,
        error_sq_limit: float = 0.1,
    ) -> None:
        self.interval = int(interval)
        self.precision_change_limit = int(precision_change_limit)
        self.error_sq_limit = float(error_sq_limit)

    def update(
        self,
        *,
        step: int,
        audit_error: float | None,
        decision: MaskDecision,
        precision_changes: int,
    ) -> MaskDecision:
        reason: str | None = None
        if step > 0 and step % self.interval == 0:
            reason = "periodic_refresh"
        elif precision_changes > self.precision_change_limit:
            reason = "precision_change_refresh"
        elif audit_error is not None and audit_error**2 >= self.error_sq_limit:
            reason = "audit_error_refresh"
        if reason is None or decision.mode == "full":
            return decision
        return MaskDecision.full(
            decision.total_keys,
            device=decision.indices.device,
            reason=reason,
        )
