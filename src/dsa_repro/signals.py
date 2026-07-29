from __future__ import annotations

import math

import torch
from torch import Tensor

from .math_utils import normalized_l2_error, normalized_probabilities


def semantic_entropy(logits: Tensor, eps: float = 1e-12) -> Tensor:
    """Equation (7), normalized to [0, 1] with natural logarithms."""
    vocabulary = logits.shape[-1]
    if vocabulary < 2:
        raise ValueError("vocabulary must contain at least two entries")
    probabilities = logits.float().softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    return (entropy / math.log(vocabulary)).clamp(0.0, 1.0)


def information_gain(
    prefix: Tensor,
    next_step: Tensor,
    *,
    inputs_are_probabilities: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Equation (8): base-2 normalized Jensen-Shannon divergence."""
    if prefix.shape != next_step.shape:
        raise ValueError(f"shape mismatch: {tuple(prefix.shape)} != {tuple(next_step.shape)}")
    if inputs_are_probabilities:
        p = normalized_probabilities(prefix, eps)
        q = normalized_probabilities(next_step, eps)
    else:
        p = prefix.float().softmax(dim=-1)
        q = next_step.float().softmax(dim=-1)
    midpoint = 0.5 * (p + q)
    kl_p = (p * (p.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=-1)
    kl_q = (q * (q.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=-1)
    return ((0.5 * (kl_p + kl_q)) / math.log(2.0)).clamp(0.0, 1.0)


def dependency_span(attention: Tensor) -> Tensor:
    """Equation (9), averaged over heads with no second normalization."""
    if attention.ndim not in {3, 4}:
        raise ValueError(
            "attention must have shape [batch, query, key] or [batch, head, query, key]"
        )
    averaged = attention.float().mean(dim=1) if attention.ndim == 4 else attention.float()
    batch, queries, keys = averaged.shape
    if queries > keys:
        raise ValueError("query length cannot exceed key length")
    result = torch.zeros(batch, queries, device=attention.device, dtype=torch.float32)
    key_positions = torch.arange(keys, device=attention.device, dtype=torch.float32)
    for query_index in range(1, queries):
        distances = (float(query_index) - key_positions).clamp_min(0.0)
        result[:, query_index] = (averaged[:, query_index, :] * distances.unsqueeze(0)).sum(
            dim=-1
        ) / float(query_index)
    return result.clamp(0.0, 1.0)


def hidden_ambiguity(hidden_states: Tensor, delta: float = 1e-6) -> Tensor:
    """Equation (21): normalized token-to-token hidden-state change."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    batch, sequence, _ = hidden_states.shape
    result = torch.zeros(batch, sequence, device=hidden_states.device, dtype=torch.float32)
    if sequence < 2:
        return result
    previous = hidden_states[:, :-1].float()
    current = hidden_states[:, 1:].float()
    ratio = (current - previous).norm(dim=-1) / (previous.norm(dim=-1) + delta)
    result[:, 1:] = ratio.clamp(max=1.0)
    return result


def local_precision_error(candidate: Tensor, reference_fp32: Tensor, delta: float = 1e-6) -> Tensor:
    """First line of Equation (27), returned per observation."""
    return normalized_l2_error(candidate.float(), reference_fp32.float(), delta=delta)


def ppr_divergence(
    controller_distribution: Tensor,
    full_distribution: Tensor,
    *,
    previous: float | Tensor = 0.3,
    momentum: float = 0.9,
) -> Tensor:
    """Equation (16): exponential moving average of output JS divergence."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    divergence = information_gain(
        controller_distribution,
        full_distribution,
        inputs_are_probabilities=True,
    )
    previous_tensor = torch.as_tensor(previous, dtype=divergence.dtype, device=divergence.device)
    return momentum * previous_tensor + (1.0 - momentum) * divergence
