from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .math_utils import normalized_cosine_similarity


@dataclass(frozen=True)
class AnchorEntry:
    layer_idx: int
    hidden_state: Tensor
    kv: Any
    chain_depth: int


class AnchorBank:
    """Keeps only fully computed states and releases them after a forward step."""

    def __init__(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self._staged: dict[int, AnchorEntry] = {}
        self._eligible: dict[int, AnchorEntry] = {}

    def _check_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers - 1}]")

    def stage_full(
        self,
        layer_idx: int,
        hidden_state: Tensor,
        kv: Any,
        chain_depth: int = 0,
    ) -> None:
        self._check_layer(layer_idx)
        if chain_depth < 0:
            raise ValueError("chain_depth cannot be negative")
        self._staged[layer_idx] = AnchorEntry(
            layer_idx=layer_idx,
            hidden_state=hidden_state.detach(),
            kv=kv,
            chain_depth=chain_depth,
        )

    def release_after_step(self) -> None:
        self._eligible = dict(self._staged)
        self._staged.clear()

    def nearest_eligible(self, layer_idx: int) -> AnchorEntry | None:
        self._check_layer(layer_idx)
        candidates = [index for index in self._eligible if index < layer_idx]
        if not candidates:
            return None
        return self._eligible[max(candidates)]

    def clear(self) -> None:
        self._staged.clear()
        self._eligible.clear()


class ResidualMapper(nn.Module):
    """Equation (3): two residual branches mixed by a learned gate."""

    def __init__(self, hidden_size: int, residual_blend: float = 1.0) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.branch_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.branch_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.residual_blend = float(residual_blend)

    def forward(self, anchor: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(anchor))
        delta = gate * self.branch_a(anchor) + (1.0 - gate) * self.branch_b(anchor)
        return anchor + self.residual_blend * delta


class DriftDetector(nn.Module):
    """Equation (5): probability that a tentative approximation has drifted."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        width = max(1, hidden_size // 4)
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_size, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, candidate: Tensor, layer_input: Tensor) -> Tensor:
        if candidate.shape != layer_input.shape:
            raise ValueError("candidate and layer_input must have the same shape")
        features = torch.cat([candidate, layer_input], dim=-1)
        return torch.sigmoid(self.network(features)).mean()


class ResidualCorrector(nn.Module):
    """Equation (6): a single bounded residual correction."""

    def __init__(self, hidden_size: int, blend: float = 0.1) -> None:
        super().__init__()
        width = max(1, 4 * hidden_size)
        self.up = nn.Linear(hidden_size, width)
        self.down = nn.Linear(width, hidden_size)
        self.blend = float(blend)

    def forward(self, candidate: Tensor) -> Tensor:
        return candidate + self.blend * self.down(F.gelu(self.up(candidate)))


@dataclass(frozen=True)
class ApproximationDecision:
    accepted: bool
    hidden_state: Tensor
    use_full_layer: bool
    corrected: bool
    similarity: float
    drift: float
    chain_depth: int
    anchor_layer: int | None
    reason: str


class AnchorApproximator(nn.Module):
    """Equations (1)-(6) plus the manuscript's maximum-chain containment rule."""

    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 128,
        gate_threshold: float = 0.85,
        drift_threshold: float = 0.15,
        max_chain: int = 3,
        *,
        projector: nn.Module | None = None,
        mapper: nn.Module | None = None,
        drift_detector: nn.Module | None = None,
        corrector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.gate_threshold = float(gate_threshold)
        self.drift_threshold = float(drift_threshold)
        self.max_chain = int(max_chain)
        self.projector = projector or nn.Linear(hidden_size, projection_dim, bias=False)
        self.mapper = mapper or ResidualMapper(hidden_size)
        self.drift_detector = drift_detector or DriftDetector(hidden_size)
        self.corrector = corrector or ResidualCorrector(hidden_size)

    @staticmethod
    def _pool(hidden_state: Tensor) -> Tensor:
        if hidden_state.ndim == 2:
            return hidden_state
        if hidden_state.ndim != 3:
            raise ValueError(
                "hidden_state must have shape [batch, hidden] or [batch, sequence, hidden]"
            )
        return hidden_state.mean(dim=1)

    def _full_decision(
        self,
        current_hidden: Tensor,
        reason: str,
        *,
        similarity: float = 0.0,
        drift: float = 0.0,
        anchor_layer: int | None = None,
    ) -> ApproximationDecision:
        return ApproximationDecision(
            accepted=False,
            hidden_state=current_hidden,
            use_full_layer=True,
            corrected=False,
            similarity=similarity,
            drift=drift,
            chain_depth=0,
            anchor_layer=anchor_layer,
            reason=reason,
        )

    def try_approximate(
        self,
        layer_idx: int,
        current_hidden: Tensor,
        bank: AnchorBank,
    ) -> ApproximationDecision:
        anchor = bank.nearest_eligible(layer_idx)
        if anchor is None:
            return self._full_decision(current_hidden, "no_anchor")
        if anchor.chain_depth >= self.max_chain:
            return self._full_decision(
                current_hidden,
                "max_chain_reached",
                anchor_layer=anchor.layer_idx,
            )

        current_projection = self.projector(self._pool(current_hidden))
        anchor_projection = self.projector(self._pool(anchor.hidden_state))
        similarity = float(
            normalized_cosine_similarity(current_projection, anchor_projection)
            .mean()
            .detach()
            .cpu()
        )
        if similarity <= self.gate_threshold:
            return self._full_decision(
                current_hidden,
                "similarity_below_gate",
                similarity=similarity,
                anchor_layer=anchor.layer_idx,
            )

        proposal = self.mapper(anchor.hidden_state)
        pre_drift = float(self.drift_detector(proposal, current_hidden).detach().cpu())
        if pre_drift <= self.drift_threshold:
            return ApproximationDecision(
                accepted=True,
                hidden_state=proposal,
                use_full_layer=False,
                corrected=False,
                similarity=similarity,
                drift=pre_drift,
                chain_depth=anchor.chain_depth + 1,
                anchor_layer=anchor.layer_idx,
                reason="accepted_pre_correction",
            )

        corrected = self.corrector(proposal)
        post_drift = float(self.drift_detector(corrected, current_hidden).detach().cpu())
        if post_drift <= self.drift_threshold:
            return ApproximationDecision(
                accepted=True,
                hidden_state=corrected,
                use_full_layer=False,
                corrected=True,
                similarity=similarity,
                drift=post_drift,
                chain_depth=anchor.chain_depth + 1,
                anchor_layer=anchor.layer_idx,
                reason="accepted_post_correction",
            )
        return self._full_decision(
            current_hidden,
            "post_correction_drift",
            similarity=similarity,
            drift=post_drift,
            anchor_layer=anchor.layer_idx,
        )
