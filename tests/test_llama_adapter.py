from __future__ import annotations

import pytest
import torch

from dsa_repro.adapters.llama import DSALlamaAdapter
from dsa_repro.config import load_config
from dsa_repro.training import AuxiliaryModules


def test_tiny_llama_generation_is_deterministic() -> None:
    first = DSALlamaAdapter.from_random_tiny(seed=7).generate_ids([1, 2, 3], max_new_tokens=3)
    second = DSALlamaAdapter.from_random_tiny(seed=7).generate_ids([1, 2, 3], max_new_tokens=3)

    assert first.token_ids == second.token_ids
    assert len(first.token_ids) == 3
    assert len(first.traces) > 0
    assert first.result_kind == "synthetic_smoke"


def test_tiny_adapter_freezes_base_model() -> None:
    adapter = DSALlamaAdapter.from_random_tiny(seed=3)

    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert not adapter.model.training


def test_generation_trace_has_one_record_per_executed_layer() -> None:
    adapter = DSALlamaAdapter.from_random_tiny(seed=11, num_hidden_layers=2)

    record = adapter.generate_ids([1, 2], max_new_tokens=2)

    assert len(record.traces) == 4
    assert {trace["layer_idx"] for trace in record.traces} == {0, 1}
    assert all(trace["path"] in {"full", "approximate"} for trace in record.traces)


def test_from_pretrained_requires_hf_token_before_network_access() -> None:
    config = load_config("configs/paper_a100.yaml")

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        DSALlamaAdapter.from_pretrained(config, token=None)


def test_install_auxiliary_binds_trained_modules() -> None:
    adapter = DSALlamaAdapter.from_random_tiny(seed=3)
    auxiliary = AuxiliaryModules(hidden_size=16)

    adapter.install_auxiliary(auxiliary)

    assert adapter.controller.approximator.projector is auxiliary.projection
    assert adapter.controller.approximator.mapper is auxiliary.residual_mapper
    assert adapter.controller.approximator.corrector is auxiliary.residual_corrector
    assert adapter.controller.approximator.drift_detector is auxiliary.drift_detector
    complexity, confidence, uncertainty, ambiguity = adapter.signal_provider(
        0, torch.ones(1, 1, 16)
    )
    assert all(0.0 <= value <= 1.0 for value in (complexity, confidence, uncertainty, ambiguity))


def test_attention_mask_disables_unselected_keys() -> None:
    base = torch.zeros(1, 1, 1, 4)
    selected = torch.tensor([0, 3])

    masked = DSALlamaAdapter.apply_selected_key_mask(base, selected)

    assert masked[0, 0, 0, 0].item() == 0.0
    assert masked[0, 0, 0, 3].item() == 0.0
    assert torch.isneginf(masked[0, 0, 0, 1])
    assert torch.isneginf(masked[0, 0, 0, 2])
