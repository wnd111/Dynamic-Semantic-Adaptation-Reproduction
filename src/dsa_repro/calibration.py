from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .math_utils import normalized_l2_error
from .signals import dependency_span, information_gain, semantic_entropy

TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CalibrationTrace:
    layer_idx: int
    layer_input: Tensor
    anchor_hidden: Tensor
    mapped_hidden: Tensor
    corrected_hidden: Tensor
    full_hidden: Tensor
    logits_prefix: Tensor
    logits_next: Tensor
    attention: Tensor
    low_precision_output: Tensor
    fp32_output: Tensor


@dataclass(frozen=True)
class MapperLabel:
    acceptable: bool
    error: float


@dataclass(frozen=True)
class PseudoLabels:
    residual_target: Tensor
    mapper_acceptable: Tensor
    drift_target: Tensor
    entropy_target: Tensor
    information_gain_target: Tensor
    dependency_target: Tensor
    precision_acceptable: Tensor


def build_mapper_label(mapped: Tensor, full: Tensor, epsilon: float = 0.05) -> MapperLabel:
    error = float(normalized_l2_error(mapped, full).mean().detach().cpu())
    return MapperLabel(acceptable=error <= epsilon, error=error)


def build_pseudo_labels(
    trace: CalibrationTrace,
    *,
    approximation_epsilon: float = 0.05,
    precision_epsilon: float = 0.03,
) -> PseudoLabels:
    mapper = build_mapper_label(trace.mapped_hidden, trace.full_hidden, approximation_epsilon)
    corrected_error = normalized_l2_error(trace.corrected_hidden, trace.full_hidden).mean()
    precision_error = normalized_l2_error(trace.low_precision_output, trace.fp32_output).mean()
    prefix_probs = trace.logits_prefix.float().softmax(dim=-1)
    next_probs = trace.logits_next.float().softmax(dim=-1)
    return PseudoLabels(
        residual_target=trace.full_hidden - trace.anchor_hidden,
        mapper_acceptable=torch.tensor(float(mapper.acceptable)),
        drift_target=(corrected_error > approximation_epsilon).float(),
        entropy_target=semantic_entropy(trace.logits_next).mean(),
        information_gain_target=information_gain(prefix_probs, next_probs).mean(),
        dependency_target=dependency_span(trace.attention).mean(),
        precision_acceptable=(precision_error <= precision_epsilon).float(),
    )


def _serialize_trace(trace: CalibrationTrace) -> dict[str, Any]:
    return asdict(trace)


def save_trace_shard(traces: Iterable[CalibrationTrace], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "traces": [_serialize_trace(trace) for trace in traces],
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_trace_shard(path: Path) -> list[CalibrationTrace]:
    payload = _torch_load(Path(path))
    if not isinstance(payload, dict) or payload.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema_version in {path}")
    traces = payload.get("traces")
    if not isinstance(traces, list):
        raise ValueError("trace shard is missing a traces list")
    return [CalibrationTrace(**item) for item in traces]
