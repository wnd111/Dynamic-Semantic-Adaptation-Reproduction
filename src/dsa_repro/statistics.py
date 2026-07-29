from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    lower: float
    upper: float
    confidence: float
    n: int
    samples: int


@dataclass(frozen=True)
class TTestResult:
    statistic: float
    pvalue: float
    mean_difference: float
    n: int


@dataclass(frozen=True)
class AggregateResult:
    mean: float
    std: float
    n: int


def _paired(ours: Sequence[float], control: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(ours) != len(control):
        raise ValueError("paired samples must have equal length")
    if not ours:
        raise ValueError("paired samples cannot be empty")
    return np.asarray(ours, dtype=np.float64), np.asarray(control, dtype=np.float64)


def paired_bootstrap(
    ours: Sequence[float],
    control: Sequence[float],
    *,
    seed: int,
    samples: int = 1000,
    confidence: float = 0.95,
) -> BootstrapResult:
    left, right = _paired(ours, control)
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), size=(samples, len(left)))
    differences = (left[indices] - right[indices]).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [alpha, 1.0 - alpha])
    return BootstrapResult(
        mean_difference=float((left - right).mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        n=len(left),
        samples=samples,
    )


def paired_t_test(ours: Sequence[float], control: Sequence[float]) -> TTestResult:
    left, right = _paired(ours, control)
    statistic, pvalue = stats.ttest_rel(left, right)
    return TTestResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        mean_difference=float((left - right).mean()),
        n=len(left),
    )


def aggregate_seeds(values: Sequence[float]) -> AggregateResult:
    if not values:
        raise ValueError("seed values cannot be empty")
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return AggregateResult(mean=float(array.mean()), std=standard_deviation, n=len(array))
