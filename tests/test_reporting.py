from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dsa_repro.config import load_config
from dsa_repro.experiments import ExperimentRunner, preflight
from dsa_repro.provenance import write_manifest
from dsa_repro.reporting import build_report, validate_result_kinds

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gate_sweep_reference_matches_paper() -> None:
    frame = pd.read_csv(PROJECT_ROOT / "paper_reference" / "table8_gate_sweep.csv")
    default = frame.loc[frame["gate_threshold"] == 0.85].iloc[0]

    assert default["approximation_ratio"] == pytest.approx(41.2)
    assert default["latency_ms_per_step"] == pytest.approx(8.32)
    assert default["rouge_l"] == pytest.approx(42.1)
    assert default["judge_score"] == pytest.approx(6.87)


def test_result_kind_validation_refuses_mixed_inputs() -> None:
    with pytest.raises(ValueError, match="result_kind"):
        validate_result_kinds([{"result_kind": "measured"}, {"result_kind": "synthetic_smoke"}])


def test_report_builds_figure_and_summary(tmp_path: Path) -> None:
    run = tmp_path / "paper-run"
    write_manifest(
        run, load_config(PROJECT_ROOT / "configs" / "paper_a100.yaml"), "paper_reference"
    )

    outputs = build_report(
        run_dirs=[run],
        reference_dir=PROJECT_ROOT / "paper_reference",
        output_dir=tmp_path / "report",
    )

    assert outputs.figure2.is_file()
    assert outputs.summary_markdown.is_file()
    assert outputs.summary_json.is_file()
    assert (
        json.loads(outputs.summary_json.read_text(encoding="utf-8"))["result_kind"]
        == "paper_reference"
    )


def test_preflight_non_strict_reports_missing_a100_without_raising() -> None:
    report = preflight(load_config(PROJECT_ROOT / "configs" / "paper_a100.yaml"), strict=False)

    assert report.ok is False
    assert any("A100" in issue or "CUDA" in issue for issue in report.issues)


def test_experiment_runner_resumes_completed_stage(tmp_path: Path) -> None:
    runner = ExperimentRunner(tmp_path / "run")
    artifact = tmp_path / "run" / "artifact.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("stable", encoding="utf-8")
    runner.mark_completed("prepare", [artifact])

    reloaded = ExperimentRunner(tmp_path / "run")

    assert reloaded.is_completed("prepare")
    assert not reloaded.is_completed("train")
