from __future__ import annotations

import math

import pytest
import torch

from dsa_repro.math_utils import normalized_cosine_similarity, normalized_l2_error
from dsa_repro.signals import (
    dependency_span,
    hidden_ambiguity,
    information_gain,
    local_precision_error,
    ppr_divergence,
    semantic_entropy,
)


def test_semantic_entropy_is_normalized() -> None:
    uniform = torch.zeros(1, 4)
    certain = torch.tensor([[40.0, -40.0, -40.0, -40.0]])

    assert semantic_entropy(uniform).item() == pytest.approx(1.0)
    assert semantic_entropy(certain).item() == pytest.approx(0.0, abs=1e-5)


def test_semantic_entropy_rejects_vocabularies_smaller_than_two() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        semantic_entropy(torch.zeros(1, 1))


def test_information_gain_is_symmetric_and_bounded() -> None:
    p = torch.tensor([[0.75, 0.25]])
    q = torch.tensor([[0.25, 0.75]])

    pq = information_gain(p, q)
    qp = information_gain(q, p)

    assert pq.item() == pytest.approx(qp.item())
    assert 0.0 <= pq.item() <= 1.0


def test_information_gain_of_identical_distributions_is_zero() -> None:
    p = torch.tensor([[0.1, 0.2, 0.7]])
    assert information_gain(p, p).item() == pytest.approx(0.0, abs=1e-7)


def test_dependency_span_matches_equation_9() -> None:
    attention = torch.tensor([[[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]]])
    span = dependency_span(attention)

    assert span.shape == (1, 3)
    assert span[0, 0].item() == 0.0
    assert span[0, 1].item() == pytest.approx(0.5)
    assert span[0, 2].item() == pytest.approx((0.2 * 2 + 0.3 * 1) / 2)


def test_ambiguity_first_position_is_zero_and_ratio_is_clipped() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]]])
    ambiguity = hidden_ambiguity(hidden)

    assert ambiguity[0, 0].item() == 0.0
    assert ambiguity[0, 1].item() == pytest.approx(1.0)
    assert ambiguity[0, 2].item() == pytest.approx(1.0)


def test_normalized_l2_error_is_scale_relative() -> None:
    reference = torch.tensor([[3.0, 4.0]])
    candidate = torch.tensor([[6.0, 8.0]])
    assert normalized_l2_error(candidate, reference).item() == pytest.approx(1.0)


def test_normalized_cosine_similarity_handles_zero_vectors() -> None:
    zero = torch.zeros(1, 4)
    assert normalized_cosine_similarity(zero, zero).item() == pytest.approx(0.0)


def test_local_precision_error_matches_equation_27() -> None:
    quantized = torch.tensor([[2.0, 4.0]])
    reference = torch.tensor([[1.0, 2.0]])
    expected = math.sqrt(5.0) / math.sqrt(5.0)
    assert local_precision_error(quantized, reference).item() == pytest.approx(expected)


def test_ppr_divergence_applies_manuscript_momentum() -> None:
    controller = torch.tensor([[0.9, 0.1]])
    full = torch.tensor([[0.1, 0.9]])
    instantaneous = information_gain(controller, full, inputs_are_probabilities=True)

    result = ppr_divergence(controller, full, previous=0.3, momentum=0.9)

    assert result.item() == pytest.approx(0.9 * 0.3 + 0.1 * instantaneous.item())
