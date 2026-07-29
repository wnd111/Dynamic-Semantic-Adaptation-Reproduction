from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .precision import PrecisionPath
from .signals import local_precision_error


@dataclass(frozen=True)
class LocalAuditRecord:
    layer_idx: int
    path: PrecisionPath
    relative_error: float
    normalized_error: float


class LocalAuditController:
    """Equation (27): paired operator-level FP32 audit and calibration normalization."""

    def __init__(
        self,
        interval: int = 20,
        path_change_limit: int = 4,
        q05: float = 0.0,
        q95: float = 1.0,
        delta: float = 1e-6,
    ) -> None:
        if interval < 1 or path_change_limit < 0:
            raise ValueError("invalid audit schedule")
        if q95 <= q05:
            raise ValueError("q95 must be larger than q05")
        self.interval = int(interval)
        self.path_change_limit = int(path_change_limit)
        self.q05 = float(q05)
        self.q95 = float(q95)
        self.delta = float(delta)
        self.records: list[LocalAuditRecord] = []

    def should_audit(self, step: int, path_changes: int, reference_event: bool = False) -> bool:
        return (
            reference_event
            or (step > 0 and step % self.interval == 0)
            or path_changes > self.path_change_limit
        )

    def observe(
        self,
        *,
        layer_idx: int,
        candidate: Tensor,
        reference_fp32: Tensor,
        path: PrecisionPath,
    ) -> float:
        relative = float(
            local_precision_error(candidate, reference_fp32, delta=self.delta).mean().detach().cpu()
        )
        normalized = min(
            1.0,
            max(0.0, (relative - self.q05) / (self.q95 - self.q05 + self.delta)),
        )
        self.records.append(
            LocalAuditRecord(
                layer_idx=int(layer_idx),
                path=path,
                relative_error=relative,
                normalized_error=normalized,
            )
        )
        return normalized


class EndToEndFeedback:
    """Equation (28): bounded momentum updates for ordered path thresholds."""

    def __init__(
        self,
        thresholds: tuple[float, float, float] = (0.7, 0.5, 0.3),
        target_band: tuple[float, float] = (0.45, 0.55),
        momentum: float = 0.9,
        step_size: float = 0.01,
        minimum_gap: float = 0.05,
        complexity_margin: float = 0.05,
    ) -> None:
        if not thresholds[0] > thresholds[1] > thresholds[2]:
            raise ValueError("thresholds must be strictly descending")
        if not 0.0 <= target_band[0] < target_band[1] <= 1.0:
            raise ValueError("invalid target band")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.thresholds = torch.tensor(thresholds, dtype=torch.float64)
        self.target_band = target_band
        self.momentum = float(momentum)
        self.step_size = float(step_size)
        self.minimum_gap = float(minimum_gap)
        self.complexity_margin = float(complexity_margin)
        self.velocity = torch.zeros(3, dtype=torch.float64)
        self.last_error: float | None = None

    @classmethod
    def default(cls) -> EndToEndFeedback:
        return cls()

    def _gradient(self, error: float) -> float:
        low, high = self.target_band
        if error > high:
            return 1.0
        if error < low:
            return -1.0
        return 0.0

    def _project(self, values: Tensor) -> Tensor:
        gap = self.minimum_gap
        tau8 = float(values[2].clamp(0.1, 0.9 - 2 * gap))
        tau16 = min(0.9 - gap, max(float(values[1]), tau8 + gap))
        tau32 = min(0.9, max(float(values[0]), tau16 + gap))
        return torch.tensor([tau32, tau16, tau8], dtype=torch.float64)

    def observe(self, *, error: float, complexity: float) -> tuple[float, float, float]:
        if not 0.0 <= error <= 1.0 or not 0.0 <= complexity <= 1.0:
            raise ValueError("error and complexity must be in [0, 1]")
        gradient = self._gradient(error)
        if gradient != 0.0:
            self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * gradient
            self.thresholds = self._project(self.thresholds - self.step_size * self.velocity)
        self.last_error = float(error)
        return tuple(float(item) for item in self.thresholds)  # type: ignore[return-value]

    def effective_thresholds(self, complexity: float) -> tuple[float, float, float]:
        if not 0.0 <= complexity <= 1.0:
            raise ValueError("complexity must be in [0, 1]")
        offset = self.complexity_margin * (complexity - 0.5)
        return tuple(float(item - offset) for item in self.thresholds)  # type: ignore[return-value]

    def fallback_requested(self, stored_complexity: float, current_complexity: float) -> bool:
        return bool(
            (self.last_error is not None and self.last_error > self.target_band[1])
            or stored_complexity + current_complexity > 1.2
        )
