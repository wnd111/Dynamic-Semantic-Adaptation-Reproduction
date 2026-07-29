from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


class PrecisionPath(str, Enum):
    FP32_ACC = "fp32_acc"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


def precision_score(uncertainty: Tensor, ambiguity: Tensor, confidence: Tensor) -> Tensor:
    """Equation (22): convex precision-sensitivity score."""
    return 0.4 * uncertainty + 0.3 * ambiguity + 0.3 * (1.0 - confidence)


class ConfidenceUncertaintyEstimator(nn.Module):
    """Equations (18)-(19), trained with the Appendix B objectives."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        confidence_width = max(1, hidden_size // 2)
        uncertainty_width = max(1, hidden_size // 4)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, confidence_width),
            nn.GELU(),
            nn.Linear(confidence_width, 1),
            nn.Sigmoid(),
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_size, uncertainty_width),
            nn.GELU(),
            nn.Linear(uncertainty_width, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.confidence(hidden_states).squeeze(-1),
            self.uncertainty(hidden_states).squeeze(-1),
        )


class PrecisionScheduler:
    """Equation (23): ordered precision path selection."""

    def __init__(
        self,
        thresholds: tuple[float, float, float] = (0.7, 0.5, 0.3),
        middle_float: PrecisionPath = PrecisionPath.FP16,
    ) -> None:
        if not thresholds[0] > thresholds[1] > thresholds[2]:
            raise ValueError("thresholds must be strictly descending")
        if middle_float not in {PrecisionPath.FP16, PrecisionPath.BF16}:
            raise ValueError("middle_float must be FP16 or BF16")
        self.thresholds = tuple(float(item) for item in thresholds)
        self.middle_float = middle_float

    @classmethod
    def default(cls) -> PrecisionScheduler:
        return cls()

    def select(self, score: float | Tensor) -> PrecisionPath:
        value = float(torch.as_tensor(score).detach().cpu())
        tau32, tau16, tau8 = self.thresholds
        if value > tau32:
            return PrecisionPath.FP32_ACC
        if value > tau16:
            return self.middle_float
        if value > tau8:
            return PrecisionPath.INT8
        return PrecisionPath.INT4


@dataclass(frozen=True)
class QuantizedTensor:
    values: Tensor
    scale: Tensor
    bits: int
    source_dtype: torch.dtype

    def dequantize(self, dtype: torch.dtype | None = None) -> Tensor:
        target = dtype or self.source_dtype
        return (self.values.float() * self.scale).to(target)


def fake_quantize(tensor: Tensor, bits: int, eps: float = 1e-8) -> QuantizedTensor:
    """Equation (26): symmetric max-absolute fake quantization."""
    if bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8")
    limit = 127 if bits == 8 else 7
    float_tensor = tensor.float()
    if tensor.ndim == 0:
        maximum = float_tensor.abs()
    else:
        maximum = float_tensor.abs().amax(dim=-1, keepdim=True)
    scale = (maximum / float(limit)).clamp_min(eps)
    values = (float_tensor / scale).round().clamp(-limit, limit).to(torch.int8)
    return QuantizedTensor(values=values, scale=scale, bits=bits, source_dtype=tensor.dtype)


class ResidentPackStore:
    """Fixed resident precision packs; missing paths are never silently emulated."""

    def __init__(self) -> None:
        self._packs: dict[tuple[int, PrecisionPath], Tensor] = {}

    def register(self, layer_idx: int, path: PrecisionPath, packed: Tensor) -> None:
        key = (int(layer_idx), path)
        if key in self._packs:
            raise ValueError(
                f"resident pack already registered for layer={layer_idx}, path={path.value}"
            )
        self._packs[key] = packed

    def get(self, layer_idx: int, path: PrecisionPath) -> Tensor:
        key = (int(layer_idx), path)
        if key not in self._packs:
            raise KeyError(f"resident pack unavailable for layer={layer_idx}, path={path.value}")
        return self._packs[key]

    def available(self, layer_idx: int) -> tuple[PrecisionPath, ...]:
        return tuple(path for (index, path) in self._packs if index == layer_idx)
