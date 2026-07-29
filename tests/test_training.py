from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from dsa_repro.calibration import CalibrationTrace, save_trace_shard
from dsa_repro.checkpoints import load_checkpoint, save_checkpoint
from dsa_repro.training import (
    AuxiliaryModules,
    complexity_target,
    make_optimizer,
    mapper_loss,
    train_synthetic_steps,
    train_trace_shards,
)


def test_mapper_loss_is_normalized_residual_error() -> None:
    mapped = torch.tensor([[2.0, 0.0]])
    full = torch.tensor([[1.0, 0.0]])

    assert mapper_loss(mapped, full).item() == pytest.approx(1.0)


def test_complexity_target_matches_equation_31() -> None:
    target = complexity_target(
        entropy=torch.tensor(0.2),
        information_gain=torch.tensor(0.4),
        dependency=torch.tensor(0.6),
    )

    assert target.item() == pytest.approx(0.4 * 0.2 + 0.35 * 0.4 + 0.25 * 0.6)


def test_optimizer_never_receives_base_parameters() -> None:
    base = nn.Linear(4, 4)
    auxiliary = AuxiliaryModules(hidden_size=4)
    optimizer = make_optimizer(base, auxiliary, learning_rate=1e-4, weight_decay=0.01)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(id(parameter) not in optimized for parameter in base.parameters())
    assert all(id(parameter) in optimized for parameter in auxiliary.parameters())


def test_synthetic_training_updates_auxiliary_but_not_base() -> None:
    torch.manual_seed(4)
    base = nn.Linear(4, 4)
    auxiliary = AuxiliaryModules(hidden_size=4)
    base_before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    aux_before = {name: value.detach().clone() for name, value in auxiliary.state_dict().items()}

    losses = train_synthetic_steps(base, auxiliary, steps=2, seed=5)

    assert len(losses) == 2
    assert all(torch.equal(base.state_dict()[name], value) for name, value in base_before.items())
    assert any(
        not torch.equal(auxiliary.state_dict()[name], value) for name, value in aux_before.items()
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    auxiliary = AuxiliaryModules(hidden_size=4)
    path = save_checkpoint(
        tmp_path / "auxiliary.pt",
        auxiliary,
        metadata={"model_id": "tiny", "epoch": 1},
    )
    target = AuxiliaryModules(hidden_size=4)

    metadata = load_checkpoint(path, target)

    assert metadata["model_id"] == "tiny"
    for name, value in auxiliary.state_dict().items():
        assert torch.equal(value, target.state_dict()[name])


def test_trace_training_consumes_saved_calibration_shard(tmp_path: Path) -> None:
    torch.manual_seed(9)
    trace = CalibrationTrace(
        layer_idx=1,
        layer_input=torch.randn(1, 1, 4),
        anchor_hidden=torch.randn(1, 1, 4),
        mapped_hidden=torch.randn(1, 1, 4),
        corrected_hidden=torch.randn(1, 1, 4),
        full_hidden=torch.randn(1, 1, 4),
        logits_prefix=torch.randn(1, 8),
        logits_next=torch.randn(1, 8),
        attention=torch.softmax(torch.randn(1, 2, 1, 3), dim=-1),
        low_precision_output=torch.randn(1, 1, 4),
        fp32_output=torch.randn(1, 1, 4),
    )
    shard = save_trace_shard([trace, trace], tmp_path / "trace-00000.pt")
    auxiliary = AuxiliaryModules(hidden_size=4)

    losses = train_trace_shards(
        [shard],
        auxiliary,
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        seed=42,
    )

    assert len(losses) == 1
    assert losses[0] > 0
