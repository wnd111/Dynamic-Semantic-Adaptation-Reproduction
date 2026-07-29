from __future__ import annotations

import pytest
import torch

from dsa_repro.precision import (
    ConfidenceUncertaintyEstimator,
    PrecisionPath,
    PrecisionScheduler,
    ResidentPackStore,
    fake_quantize,
    precision_score,
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.8, PrecisionPath.FP32_ACC),
        (0.6, PrecisionPath.FP16),
        (0.4, PrecisionPath.INT8),
        (0.2, PrecisionPath.INT4),
    ],
)
def test_precision_threshold_order(score: float, expected: PrecisionPath) -> None:
    assert PrecisionScheduler.default().select(score) is expected


def test_precision_score_matches_equation_22() -> None:
    score = precision_score(
        uncertainty=torch.tensor(0.8),
        ambiguity=torch.tensor(0.4),
        confidence=torch.tensor(0.5),
    )
    assert score.item() == pytest.approx(0.4 * 0.8 + 0.3 * 0.4 + 0.3 * 0.5)


def test_int4_quantization_is_symmetric_and_bounded() -> None:
    q = fake_quantize(torch.tensor([-20.0, -1.0, 0.0, 1.0, 20.0]), bits=4)

    assert q.values.min().item() >= -7
    assert q.values.max().item() <= 7
    assert q.bits == 4
    assert torch.isfinite(q.dequantize()).all()


def test_zero_tensor_quantization_has_finite_scale() -> None:
    q = fake_quantize(torch.zeros(2, 4), bits=8)

    assert torch.all(q.scale > 0)
    assert torch.equal(q.dequantize(), torch.zeros(2, 4))


def test_unsupported_bit_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="4 or 8"):
        fake_quantize(torch.ones(2), bits=3)


def test_estimator_outputs_are_probabilities() -> None:
    estimator = ConfidenceUncertaintyEstimator(hidden_size=8)
    confidence, uncertainty = estimator(torch.randn(2, 3, 8))

    assert confidence.shape == (2, 3)
    assert uncertainty.shape == (2, 3)
    assert torch.all((0.0 <= confidence) & (confidence <= 1.0))
    assert torch.all((0.0 <= uncertainty) & (uncertainty <= 1.0))


def test_resident_pack_store_never_silently_substitutes() -> None:
    store = ResidentPackStore()
    store.register(layer_idx=2, path=PrecisionPath.INT8, packed=torch.ones(2))

    assert store.get(2, PrecisionPath.INT8).shape == (2,)
    with pytest.raises(KeyError, match="resident pack"):
        store.get(2, PrecisionPath.INT4)
