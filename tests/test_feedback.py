from __future__ import annotations

import pytest
import torch

from dsa_repro.feedback import EndToEndFeedback, LocalAuditController
from dsa_repro.precision import PrecisionPath


def test_observed_high_error_lowers_all_thresholds() -> None:
    controller = EndToEndFeedback.default()
    before = controller.thresholds.clone()

    controller.observe(error=0.8, complexity=0.4)

    assert torch.all(controller.thresholds < before)


def test_observed_low_error_raises_all_thresholds() -> None:
    controller = EndToEndFeedback.default()
    before = controller.thresholds.clone()

    controller.observe(error=0.1, complexity=0.4)

    assert torch.all(controller.thresholds > before)


def test_inside_target_band_does_not_move_thresholds() -> None:
    controller = EndToEndFeedback.default()
    before = controller.thresholds.clone()

    controller.observe(error=0.5, complexity=0.4)

    assert torch.equal(controller.thresholds, before)


def test_projected_thresholds_keep_order_and_margin() -> None:
    controller = EndToEndFeedback(
        thresholds=(0.12, 0.11, 0.1),
        target_band=(0.45, 0.55),
        momentum=0.0,
        step_size=1.0,
        minimum_gap=0.05,
    )

    controller.observe(error=1.0, complexity=0.5)
    tau32, tau16, tau8 = controller.thresholds.tolist()

    assert 0.1 <= tau8
    assert tau8 + 0.05 <= tau16 + 1e-7
    assert tau16 + 0.05 <= tau32 + 1e-7
    assert tau32 <= 0.9


def test_local_audit_runs_periodically_and_after_many_path_changes() -> None:
    audit = LocalAuditController(interval=20, path_change_limit=4)

    assert audit.should_audit(step=20, path_changes=0)
    assert audit.should_audit(step=3, path_changes=5)
    assert not audit.should_audit(step=3, path_changes=4)


def test_local_audit_normalizes_against_calibration_quantiles() -> None:
    audit = LocalAuditController(interval=20, path_change_limit=4, q05=0.1, q95=0.5)

    normalized = audit.observe(
        layer_idx=2,
        candidate=torch.tensor([[1.4, 0.0]]),
        reference_fp32=torch.tensor([[1.0, 0.0]]),
        path=PrecisionPath.INT8,
    )

    assert normalized == pytest.approx((0.4 - 0.1) / (0.5 - 0.1), abs=1e-5)
    assert audit.records[-1].layer_idx == 2
