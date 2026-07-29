from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dsa_repro.calibration import (
    CalibrationTrace,
    build_mapper_label,
    build_pseudo_labels,
    load_trace_shard,
    save_trace_shard,
)


def make_trace() -> CalibrationTrace:
    return CalibrationTrace(
        layer_idx=2,
        layer_input=torch.tensor([[[1.0, 0.0]]]),
        anchor_hidden=torch.tensor([[[0.9, 0.1]]]),
        mapped_hidden=torch.tensor([[[1.01, 0.0]]]),
        corrected_hidden=torch.tensor([[[1.005, 0.0]]]),
        full_hidden=torch.tensor([[[1.0, 0.0]]]),
        logits_prefix=torch.tensor([[2.0, 1.0]]),
        logits_next=torch.tensor([[1.0, 2.0]]),
        attention=torch.tensor([[[[1.0]]]]),
        low_precision_output=torch.tensor([[[0.99, 0.0]]]),
        fp32_output=torch.tensor([[[1.0, 0.0]]]),
    )


def test_mapper_label_uses_paper_error_threshold() -> None:
    label = build_mapper_label(
        mapped=torch.ones(4),
        full=torch.ones(4) * 1.01,
        epsilon=0.05,
    )

    assert label.acceptable
    assert label.error < 0.05


def test_pseudo_labels_contain_all_appendix_targets() -> None:
    labels = build_pseudo_labels(make_trace(), approximation_epsilon=0.05, precision_epsilon=0.03)

    assert labels.mapper_acceptable.item() == 1.0
    assert labels.drift_target.item() == 0.0
    assert labels.precision_acceptable.item() == 1.0
    assert 0.0 <= labels.entropy_target.item() <= 1.0
    assert 0.0 <= labels.information_gain_target.item() <= 1.0
    assert 0.0 <= labels.dependency_target.item() <= 1.0


def test_trace_shard_round_trip(tmp_path: Path) -> None:
    path = save_trace_shard([make_trace()], tmp_path / "trace-00000.pt")
    loaded = load_trace_shard(path)

    assert len(loaded) == 1
    assert loaded[0].layer_idx == 2
    assert torch.equal(loaded[0].full_hidden, make_trace().full_hidden)


def test_trace_shard_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 99, "traces": []}, path)

    with pytest.raises(ValueError, match="schema_version"):
        load_trace_shard(path)
