from __future__ import annotations

import pytest

from dsa_repro.anchor import AnchorApproximator
from dsa_repro.controller import RuntimeController
from dsa_repro.feedback import EndToEndFeedback
from dsa_repro.pipeline import configure_variant, extract_prompt_reference, summarize_metrics
from dsa_repro.precision import PrecisionPath
from dsa_repro.pruning import MaskSelector, ProgressiveRefresh


def test_hotpot_prompt_includes_context_and_question() -> None:
    row = {
        "question": "Where was Ada born?",
        "answer": "London",
        "context": {"title": ["Ada"], "sentences": [["Ada was born in London."]]},
    }

    prompt, reference = extract_prompt_reference("hotpotqa", row)

    assert "Ada was born in London" in prompt
    assert "Where was Ada born?" in prompt
    assert reference["answer"] == "London"


def test_alpaca_prompt_uses_instruction() -> None:
    prompt, reference = extract_prompt_reference("alpacaeval", {"instruction": "Write a haiku."})

    assert prompt == "Write a haiku."
    assert reference == {}


def test_metric_summary_averages_each_key() -> None:
    summary = summarize_metrics([{"f1": 0.5, "exact_match": 0.0}, {"f1": 1.0, "exact_match": 1.0}])

    assert summary == {"exact_match": pytest.approx(0.5), "f1": pytest.approx(0.75)}


def _controller() -> RuntimeController:
    return RuntimeController(
        num_layers=2,
        approximator=AnchorApproximator(hidden_size=4),
        mask_selector=MaskSelector.default(),
        feedback=EndToEndFeedback.default(),
        refresh=ProgressiveRefresh(interval=20, precision_change_limit=4, error_sq_limit=0.1),
    )


@pytest.mark.parametrize(
    ("variant", "approximation", "pruning", "precision"),
    [
        ("full", True, True, None),
        ("no-approximation", False, True, None),
        ("no-pruning", True, False, None),
        ("fp16-only", True, True, PrecisionPath.FP16),
    ],
)
def test_configure_variant_sets_exactly_one_ablation(
    variant: str,
    approximation: bool,
    pruning: bool,
    precision: PrecisionPath | None,
) -> None:
    controller = _controller()

    configure_variant(controller, variant)

    assert controller.approximation_enabled is approximation
    assert controller.pruning_enabled is pruning
    assert controller.precision_override is precision
