from __future__ import annotations

import pytest

from dsa_repro.statistics import aggregate_seeds, paired_bootstrap, paired_t_test


def test_paired_bootstrap_is_reproducible() -> None:
    first = paired_bootstrap([1, 2, 3], [1, 1, 1], seed=42, samples=100)
    second = paired_bootstrap([1, 2, 3], [1, 1, 1], seed=42, samples=100)

    assert first == second
    assert first.n == 3


def test_paired_bootstrap_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap([1, 2], [1], seed=42, samples=100)


def test_paired_t_test_reports_expected_direction() -> None:
    result = paired_t_test([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])

    assert result.n == 5
    assert result.mean_difference > 0
    assert result.pvalue < 0.05


def test_seed_aggregation_uses_sample_standard_deviation() -> None:
    result = aggregate_seeds([1.0, 2.0, 3.0])

    assert result.mean == pytest.approx(2.0)
    assert result.std == pytest.approx(1.0)
    assert result.n == 3
