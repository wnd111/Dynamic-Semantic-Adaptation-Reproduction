from __future__ import annotations

import torch

from dsa_repro.anchor import AnchorBank, ApproximationDecision
from dsa_repro.controller import LayerContext, RuntimeController
from dsa_repro.feedback import EndToEndFeedback
from dsa_repro.precision import PrecisionPath
from dsa_repro.pruning import MaskSelector, ProgressiveRefresh


class FixedApproximator:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted

    def try_approximate(
        self, layer_idx: int, current_hidden: torch.Tensor, bank: AnchorBank
    ) -> ApproximationDecision:
        del bank
        if self.accepted:
            return ApproximationDecision(
                accepted=True,
                hidden_state=current_hidden + 2.0,
                use_full_layer=False,
                corrected=False,
                similarity=0.99,
                drift=0.01,
                chain_depth=1,
                anchor_layer=max(0, layer_idx - 1),
                reason="accepted_pre_correction",
            )
        return ApproximationDecision(
            accepted=False,
            hidden_state=current_hidden,
            use_full_layer=True,
            corrected=False,
            similarity=0.2,
            drift=0.0,
            chain_depth=0,
            anchor_layer=None,
            reason="similarity_below_gate",
        )


def make_controller(accepted: bool = False) -> RuntimeController:
    return RuntimeController(
        num_layers=4,
        approximator=FixedApproximator(accepted),
        mask_selector=MaskSelector.default(),
        feedback=EndToEndFeedback.default(),
        refresh=ProgressiveRefresh(interval=20, precision_change_limit=4, error_sq_limit=0.1),
        e2e_interval=100,
    )


def make_context(**changes: object) -> LayerContext:
    values: dict[str, object] = {
        "layer_idx": 2,
        "hidden_state": torch.ones(1, 1, 4),
        "kv_length": 4,
        "importance": torch.tensor([0.4, 0.3, 0.2, 0.1]),
        "complexity_score": 0.2,
        "confidence": 0.9,
        "uncertainty": 0.1,
        "ambiguity": 0.1,
        "audit_error": None,
        "precision_changes": 0,
        "force_full": False,
    }
    values.update(changes)
    return LayerContext(**values)


def test_accepted_approximation_does_not_call_full_block() -> None:
    controller = make_controller(accepted=True)
    calls: list[str] = []

    plan = controller.enter_layer(make_context())
    result = plan.execute(lambda hidden, mask, precision: calls.append("full"))

    assert calls == []
    assert result.trace.path == "approximate"
    assert torch.equal(result.hidden_state, torch.full((1, 1, 4), 3.0))


def test_full_path_executes_block_with_selected_mask_and_precision() -> None:
    controller = make_controller(accepted=False)
    observed: dict[str, object] = {}

    def full_block(
        hidden: torch.Tensor, mask: torch.Tensor, precision: PrecisionPath
    ) -> torch.Tensor:
        observed["mask"] = mask
        observed["precision"] = precision
        return hidden + 1.0

    plan = controller.enter_layer(make_context())
    result = plan.execute(full_block)

    assert result.trace.path == "full"
    assert result.trace.mask_mode == "topk"
    assert observed["precision"] is PrecisionPath.INT4
    assert torch.equal(result.hidden_state, torch.full((1, 1, 4), 2.0))


def test_only_full_results_become_future_anchors() -> None:
    controller = make_controller(accepted=False)
    plan = controller.enter_layer(make_context())
    result = plan.execute(lambda hidden, mask, precision: hidden + 1.0)

    controller.finish_layer(result, kv=(torch.ones(1), torch.ones(1)))
    assert controller.anchor_bank.nearest_eligible(3) is None

    controller.finish_step()
    assert controller.anchor_bank.nearest_eligible(3).layer_idx == 2


def test_approximate_results_are_not_staged_as_anchors() -> None:
    controller = make_controller(accepted=True)
    result = controller.enter_layer(make_context()).execute(lambda hidden, mask, precision: hidden)

    controller.finish_layer(result, kv=None)
    controller.finish_step()

    assert controller.anchor_bank.nearest_eligible(3) is None


def test_common_fallback_forces_full_attention_and_fp32() -> None:
    controller = make_controller(accepted=True)

    plan = controller.enter_layer(make_context(force_full=True))

    assert plan.path == "full"
    assert plan.mask.mode == "full"
    assert plan.precision is PrecisionPath.FP32_ACC
    assert plan.reason == "common_fallback"


def test_periodic_replay_schedules_next_step_fallback() -> None:
    controller = make_controller(accepted=False)
    controller.state.step = 99

    controller.finish_step()

    assert controller.state.force_full_next_step
    assert controller.state.audit_count == 1


def test_begin_step_consumes_scheduled_fallback() -> None:
    controller = make_controller(accepted=False)
    controller.state.force_full_next_step = True

    controller.begin_step()

    assert controller.state.force_full_current_step
    assert not controller.state.force_full_next_step
    assert controller.enter_layer(make_context()).reason == "common_fallback"


def test_trace_contains_runtime_diagnostics() -> None:
    controller = make_controller(accepted=False)
    result = controller.enter_layer(make_context(complexity_score=0.8)).execute(
        lambda hidden, mask, precision: hidden
    )

    trace = result.trace.to_dict()

    assert trace["layer_idx"] == 2
    assert trace["path"] == "full"
    assert trace["complexity_score"] == 0.8
    assert trace["mask_mode"] == "full"
    assert trace["precision"] == "int4"


def test_ablation_switches_disable_each_runtime_decision() -> None:
    controller = make_controller(accepted=True)
    controller.approximation_enabled = False
    controller.pruning_enabled = False
    controller.precision_override = PrecisionPath.FP16

    plan = controller.enter_layer(make_context())

    assert plan.path == "full"
    assert plan.mask.mode == "full"
    assert plan.precision is PrecisionPath.FP16
