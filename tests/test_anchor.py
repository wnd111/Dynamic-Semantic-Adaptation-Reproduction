from __future__ import annotations

import pytest
import torch
from torch import nn

from dsa_repro.anchor import AnchorApproximator, AnchorBank, ResidualMapper


class IdentityProjection(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class IdentityMapper(nn.Module):
    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        return anchor


class SequenceDriftDetector(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = list(values)

    def forward(self, candidate: torch.Tensor, layer_input: torch.Tensor) -> torch.Tensor:
        del candidate, layer_input
        return torch.tensor(self.values.pop(0))


class AdditiveCorrector(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return candidate + self.increment


def make_bank(anchor: torch.Tensor, *, chain_depth: int = 0) -> AnchorBank:
    bank = AnchorBank(num_layers=4)
    bank.stage_full(layer_idx=1, hidden_state=anchor, kv=None, chain_depth=chain_depth)
    bank.release_after_step()
    return bank


def make_approximator(drift_values: list[float]) -> AnchorApproximator:
    return AnchorApproximator(
        hidden_size=2,
        projection_dim=2,
        gate_threshold=0.85,
        drift_threshold=0.15,
        max_chain=3,
        projector=IdentityProjection(),
        mapper=IdentityMapper(),
        drift_detector=SequenceDriftDetector(drift_values),
        corrector=AdditiveCorrector(0.25),
    )


def test_anchor_is_not_visible_until_step_release() -> None:
    bank = AnchorBank(num_layers=4)
    bank.stage_full(1, torch.ones(1, 1, 2), kv=None)

    assert bank.nearest_eligible(2) is None

    bank.release_after_step()
    assert bank.nearest_eligible(2).layer_idx == 1


def test_nearest_eligible_anchor_is_selected() -> None:
    bank = AnchorBank(num_layers=5)
    bank.stage_full(0, torch.zeros(1, 1, 2), kv=None)
    bank.stage_full(2, torch.ones(1, 1, 2), kv=None)
    bank.release_after_step()

    assert bank.nearest_eligible(4).layer_idx == 2
    assert bank.nearest_eligible(2).layer_idx == 0


def test_bank_rejects_noncausal_anchor_lookup() -> None:
    bank = AnchorBank(num_layers=3)
    with pytest.raises(ValueError, match="layer_idx"):
        bank.nearest_eligible(3)


def test_high_similarity_and_low_drift_accepts_without_correction() -> None:
    current = torch.tensor([[[1.0, 0.0]]])
    bank = make_bank(torch.tensor([[[0.99, 0.01]]]))

    decision = make_approximator([0.1]).try_approximate(2, current, bank)

    assert decision.accepted
    assert not decision.use_full_layer
    assert not decision.corrected
    assert decision.chain_depth == 1
    assert decision.reason == "accepted_pre_correction"


def test_low_similarity_forces_full_layer_before_drift_check() -> None:
    current = torch.tensor([[[1.0, 0.0]]])
    bank = make_bank(torch.tensor([[[0.0, 1.0]]]))

    decision = make_approximator([]).try_approximate(2, current, bank)

    assert decision.use_full_layer
    assert decision.reason == "similarity_below_gate"


def test_one_successful_correction_is_accepted() -> None:
    current = torch.tensor([[[1.0, 0.0]]])
    bank = make_bank(torch.tensor([[[1.0, 0.0]]]))

    decision = make_approximator([0.2, 0.1]).try_approximate(2, current, bank)

    assert decision.accepted
    assert decision.corrected
    assert decision.drift == pytest.approx(0.1)
    assert decision.hidden_state[0, 0, 0].item() == pytest.approx(1.25)
    assert decision.reason == "accepted_post_correction"


def test_failed_correction_forces_full_layer() -> None:
    current = torch.tensor([[[1.0, 0.0]]])
    bank = make_bank(torch.tensor([[[1.0, 0.0]]]))

    decision = make_approximator([0.2, 0.2]).try_approximate(2, current, bank)

    assert decision.use_full_layer
    assert not decision.accepted
    assert decision.reason == "post_correction_drift"


def test_maximum_chain_forces_full_layer() -> None:
    current = torch.tensor([[[1.0, 0.0]]])
    bank = make_bank(torch.tensor([[[1.0, 0.0]]]), chain_depth=3)

    decision = make_approximator([]).try_approximate(2, current, bank)

    assert decision.use_full_layer
    assert decision.reason == "max_chain_reached"


def test_residual_mapper_implements_gated_two_branch_delta() -> None:
    mapper = ResidualMapper(hidden_size=2, residual_blend=1.0)
    with torch.no_grad():
        mapper.gate.weight.zero_()
        mapper.gate.bias.zero_()
        mapper.branch_a.weight.copy_(torch.eye(2))
        mapper.branch_b.weight.copy_(2.0 * torch.eye(2))
    anchor = torch.tensor([[[1.0, 2.0]]])

    mapped = mapper(anchor)

    assert torch.allclose(mapped, torch.tensor([[[2.5, 5.0]]]))
