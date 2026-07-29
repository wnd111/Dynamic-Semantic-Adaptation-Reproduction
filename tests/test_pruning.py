from __future__ import annotations

import pytest
import torch

from dsa_repro.pruning import (
    ComplexityPredictor,
    ComplexitySignals,
    MaskDecision,
    MaskSelector,
    ProgressiveRefresh,
)


@pytest.mark.parametrize(
    ("score", "mode", "keep"),
    [(0.8, "full", 10), (0.5, "window", 9), (0.2, "topk", 5)],
)
def test_three_tier_mask(score: float, mode: str, keep: int) -> None:
    importance = torch.arange(10, dtype=torch.float32)
    decision = MaskSelector(
        high_threshold=0.7,
        low_threshold=0.3,
        topk_fraction=0.5,
        window_fraction=0.9,
    ).select(torch.tensor(score), importance, seq_len=10)

    assert decision.mode == mode
    assert decision.indices.numel() == keep
    assert decision.indices.tolist() == sorted(decision.indices.tolist())


def test_topk_has_no_rng_dependence() -> None:
    importance = torch.tensor([0.4, 0.1, 0.3, 0.2])
    selector = MaskSelector.default()

    first = selector.select(torch.tensor(0.1), importance, 4)
    second = selector.select(torch.tensor(0.1), importance, 4)

    assert torch.equal(first.indices, second.indices)
    assert first.indices.tolist() == [0, 3]


def test_topk_always_contains_current_token() -> None:
    importance = torch.tensor([100.0, 90.0, 80.0, -100.0])
    decision = MaskSelector.default().select(torch.tensor(0.1), importance, 4)

    assert 3 in decision.indices.tolist()


def test_importance_combines_attention_and_recency() -> None:
    mean_attention = torch.tensor([0.6, 0.3, 0.1])
    importance = MaskSelector.importance(mean_attention, recency_weight=0.3)

    assert importance.tolist() == pytest.approx([0.42, 0.21, 0.37])


def test_full_decision_constructs_all_indices() -> None:
    decision = MaskDecision.full(4)

    assert decision.mode == "full"
    assert decision.indices.tolist() == [0, 1, 2, 3]


def test_periodic_refresh_enlarges_to_full() -> None:
    refresh = ProgressiveRefresh(interval=20, precision_change_limit=4, error_sq_limit=0.1)
    sparse = MaskDecision(mode="topk", indices=torch.tensor([0, 3]), total_keys=4)

    decision = refresh.update(step=20, audit_error=None, decision=sparse, precision_changes=0)

    assert decision.mode == "full"
    assert decision.reason == "periodic_refresh"


def test_precision_change_refresh_triggers_above_four_changes() -> None:
    refresh = ProgressiveRefresh(interval=20, precision_change_limit=4, error_sq_limit=0.1)
    sparse = MaskDecision(mode="window", indices=torch.tensor([1, 2, 3]), total_keys=4)

    decision = refresh.update(step=3, audit_error=None, decision=sparse, precision_changes=5)

    assert decision.mode == "full"
    assert decision.reason == "precision_change_refresh"


def test_audit_error_refresh_uses_squared_threshold() -> None:
    refresh = ProgressiveRefresh(interval=20, precision_change_limit=4, error_sq_limit=0.1)
    sparse = MaskDecision(mode="topk", indices=torch.tensor([0, 3]), total_keys=4)

    decision = refresh.update(step=3, audit_error=0.32, decision=sparse, precision_changes=0)

    assert decision.mode == "full"
    assert decision.reason == "audit_error_refresh"


def test_complexity_predictor_output_is_bounded() -> None:
    predictor = ComplexityPredictor(hidden_size=8)
    hidden = torch.randn(2, 1, 8)
    signals = ComplexitySignals(
        entropy=torch.tensor([0.2, 0.8]),
        information_gain=torch.tensor([0.1, 0.9]),
        dependency_span=torch.tensor([0.3, 0.7]),
    )

    score = predictor(hidden, signals)

    assert score.shape == (2,)
    assert torch.all((0.0 <= score) & (score <= 1.0))
