from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .anchor import DriftDetector, ResidualCorrector, ResidualMapper
from .calibration import CalibrationTrace, build_pseudo_labels, load_trace_shard
from .precision import ConfidenceUncertaintyEstimator
from .pruning import ComplexityPredictor


def mapper_loss(mapped: Tensor, full: Tensor, eps: float = 1e-6) -> Tensor:
    numerator = (mapped.float() - full.float()).norm(dim=-1)
    denominator = full.float().norm(dim=-1).clamp_min(eps)
    return (numerator / denominator).mean()


def complexity_target(entropy: Tensor, information_gain: Tensor, dependency: Tensor) -> Tensor:
    """Equation (31), with fixed pre-evaluation weights."""
    return 0.4 * entropy + 0.35 * information_gain + 0.25 * dependency


class AuxiliaryModules(nn.Module):
    """All offline-trained modules; the base LLaMA is deliberately absent."""

    def __init__(self, hidden_size: int, residual_blend: float = 1.0) -> None:
        super().__init__()
        projection_dim = min(128, hidden_size)
        self.projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.gate_head = nn.Sequential(
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, 1),
        )
        self.residual_mapper = ResidualMapper(hidden_size, residual_blend=residual_blend)
        self.residual_corrector = ResidualCorrector(hidden_size)
        self.drift_detector = DriftDetector(hidden_size)
        self.complexity_predictor = ComplexityPredictor(hidden_size)
        self.confidence_estimator = ConfidenceUncertaintyEstimator(hidden_size)

    def gate_logit(self, current: Tensor, anchor: Tensor) -> Tensor:
        current_pool = current.mean(dim=1) if current.ndim == 3 else current
        anchor_pool = anchor.mean(dim=1) if anchor.ndim == 3 else anchor
        features = torch.cat([self.projection(current_pool), self.projection(anchor_pool)], dim=-1)
        return self.gate_head(features).squeeze(-1)


def make_optimizer(
    base_model: nn.Module,
    auxiliary: AuxiliaryModules,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    base_model.requires_grad_(False).eval()
    auxiliary.requires_grad_(True).train()
    return torch.optim.AdamW(
        auxiliary.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_synthetic_steps(
    base_model: nn.Module,
    auxiliary: AuxiliaryModules,
    *,
    steps: int,
    seed: int,
) -> list[float]:
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    hidden_size = auxiliary.residual_mapper.gate.in_features
    optimizer = make_optimizer(
        base_model,
        auxiliary,
        learning_rate=1e-3,
        weight_decay=0.01,
    )
    losses: list[float] = []
    for _ in range(steps):
        layer_input = torch.randn(8, 1, hidden_size)
        anchor = torch.randn(8, 1, hidden_size)
        with torch.no_grad():
            full = base_model(layer_input.squeeze(1)).unsqueeze(1)
        mapped = auxiliary.residual_mapper(anchor)
        corrected = auxiliary.residual_corrector(mapped)
        gate_logits = auxiliary.gate_logit(layer_input, anchor)
        gate_target = (normalized_error(mapped, full) <= 0.05).float().squeeze(-1)
        drift_prediction = auxiliary.drift_detector(corrected, layer_input)
        drift_target = (normalized_error(corrected, full).mean() > 0.05).float()
        complexity_prediction = auxiliary.complexity_predictor(layer_input)
        synthetic_complexity = torch.sigmoid(layer_input.float().mean(dim=(1, 2)))
        confidence, uncertainty = auxiliary.confidence_estimator(layer_input)
        confidence_target = (normalized_error(mapped, full) <= 0.03).float()
        entropy_target = torch.sigmoid(layer_input.float().std(dim=-1, unbiased=False))
        loss = (
            mapper_loss(mapped, full)
            + 0.5 * mapper_loss(corrected, full)
            + F.binary_cross_entropy_with_logits(gate_logits, gate_target)
            + F.binary_cross_entropy(drift_prediction, drift_target)
            + F.mse_loss(complexity_prediction, synthetic_complexity)
            + F.binary_cross_entropy(confidence, confidence_target)
            + F.mse_loss(uncertainty, entropy_target)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def normalized_error(candidate: Tensor, reference: Tensor, eps: float = 1e-6) -> Tensor:
    return (candidate.float() - reference.float()).norm(dim=-1) / reference.float().norm(
        dim=-1
    ).clamp_min(eps)


def _batch(
    traces: list[CalibrationTrace],
    field: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.cat([getattr(trace, field) for trace in traces], dim=0).to(
        device=device,
        dtype=dtype,
    )


def train_trace_shards(
    shard_paths: list[object],
    auxiliary: AuxiliaryModules,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    """Train all auxiliary heads from immutable full-model calibration traces."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    traces = [trace for path in shard_paths for trace in load_trace_shard(path)]
    if not traces:
        raise ValueError("no calibration traces were loaded")
    torch.manual_seed(seed)
    parameter = next(auxiliary.parameters())
    device = parameter.device
    dtype = parameter.dtype
    optimizer = torch.optim.AdamW(
        auxiliary.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    auxiliary.train()
    epoch_losses: list[float] = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(traces), generator=generator).tolist()
        running: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = [traces[index] for index in order[start : start + batch_size]]
            layer_input = _batch(batch, "layer_input", device=device, dtype=dtype)
            anchor = _batch(batch, "anchor_hidden", device=device, dtype=dtype)
            full = _batch(batch, "full_hidden", device=device, dtype=dtype)
            mapped = auxiliary.residual_mapper(anchor)
            corrected = auxiliary.residual_corrector(mapped)
            labels = [build_pseudo_labels(trace) for trace in batch]
            gate_target = torch.stack([label.mapper_acceptable for label in labels]).to(
                device=device, dtype=dtype
            )
            drift_target = torch.stack([label.drift_target for label in labels]).to(
                device=device, dtype=dtype
            )
            entropy_target = torch.stack([label.entropy_target for label in labels]).to(
                device=device, dtype=dtype
            )
            complexity_targets = torch.stack(
                [
                    complexity_target(
                        label.entropy_target,
                        label.information_gain_target,
                        label.dependency_target,
                    )
                    for label in labels
                ]
            ).to(device=device, dtype=dtype)
            precision_target = torch.stack([label.precision_acceptable for label in labels]).to(
                device=device, dtype=dtype
            )
            gate_logits = auxiliary.gate_logit(layer_input, anchor)
            drift_prediction = torch.stack(
                [
                    auxiliary.drift_detector(corrected[i : i + 1], layer_input[i : i + 1])
                    for i in range(len(batch))
                ]
            )
            complexity_prediction = auxiliary.complexity_predictor(layer_input)
            confidence, uncertainty = auxiliary.confidence_estimator(layer_input)
            confidence = confidence.mean(dim=-1)
            uncertainty = uncertainty.mean(dim=-1)
            loss = (
                mapper_loss(mapped, full)
                + 0.5 * mapper_loss(corrected, full)
                + F.binary_cross_entropy_with_logits(gate_logits, gate_target)
                + F.binary_cross_entropy(drift_prediction, drift_target)
                + F.mse_loss(complexity_prediction, complexity_targets)
                + F.binary_cross_entropy(confidence, precision_target)
                + F.mse_loss(uncertainty, entropy_target)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach().cpu()))
        epoch_losses.append(sum(running) / len(running))
    return epoch_losses
