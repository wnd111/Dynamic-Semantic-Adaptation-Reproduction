from __future__ import annotations

from pathlib import Path

import pytest

from dsa_repro.data import deterministic_select, load_manifest, validate_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "count"),
    [
        ("alpacaeval", 805),
        ("vicuna80", 80),
        ("hotpotqa", 405),
        ("asqa", 948),
        ("sharegpt512", 512),
    ],
)
def test_manifests_match_paper_sample_counts(name: str, count: int) -> None:
    manifest = load_manifest(PROJECT_ROOT / "data" / "manifests" / f"{name}.json")

    assert manifest.name == name
    assert manifest.sample_count == count
    assert manifest.revision != "main"
    assert manifest.required_fields


def test_deterministic_selection_is_seeded_and_stable() -> None:
    rows = [{"id": str(index)} for index in range(20)]

    first = deterministic_select(rows, count=5, seed=42)
    second = deterministic_select(rows, count=5, seed=42)

    assert first == second
    assert len(first) == 5


def test_row_validation_reports_missing_fields() -> None:
    with pytest.raises(ValueError, match="answer"):
        validate_rows([{"question": "q"}], required_fields=("question", "answer"))
