from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

from .provenance import atomic_write_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ReportOutputs:
    figure2: Path
    summary_markdown: Path
    summary_json: Path


def validate_result_kinds(manifests: Sequence[dict[str, object]]) -> str:
    kinds = {str(manifest.get("result_kind")) for manifest in manifests}
    if len(kinds) != 1 or "None" in kinds:
        raise ValueError(f"result_kind mismatch: {sorted(kinds)}")
    return kinds.pop()


def _read_manifest(run_dir: Path) -> dict[str, object]:
    path = Path(run_dir) / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    *,
    run_dirs: Sequence[Path],
    reference_dir: Path,
    output_dir: Path,
) -> ReportOutputs:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    manifests = [_read_manifest(Path(run_dir)) for run_dir in run_dirs]
    result_kind = validate_result_kinds(manifests)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(reference_dir)
    gate_path = next(
        (
            Path(run_dir) / "gate_sweep.csv"
            for run_dir in run_dirs
            if (Path(run_dir) / "gate_sweep.csv").is_file()
        ),
        reference_dir / "table8_gate_sweep.csv",
    )
    gate = pd.read_csv(gate_path)

    figure2 = output_dir / "figure2_gate_sweep.png"
    figure, left = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    right = left.twinx()
    left.plot(
        gate["gate_threshold"],
        gate["approximation_ratio"],
        "o-",
        color="#0072B2",
        label="Approximation ratio",
    )
    right.plot(gate["gate_threshold"], gate["rouge_l"], "s-", color="#E69F00", label="ROUGE-L")
    left.axvline(0.85, color="#666666", linestyle="--", linewidth=1)
    left.set_xlabel("Gate threshold")
    left.set_ylabel("Accepted approximation ratio (%)", color="#0072B2")
    right.set_ylabel("HotpotQA ROUGE-L", color="#E69F00")
    left.grid(alpha=0.25)
    figure.savefig(figure2, dpi=220)
    plt.close(figure)

    tables = {path.stem: len(pd.read_csv(path)) for path in sorted(reference_dir.glob("*.csv"))}
    summary = {
        "schema_version": 1,
        "result_kind": result_kind,
        "run_dirs": [str(Path(item)) for item in run_dirs],
        "gate_sweep_source": str(gate_path),
        "reference_tables": tables,
        "figure2": str(figure2),
    }
    summary_json = atomic_write_json(output_dir / "summary.json", summary)
    summary_markdown = output_dir / "summary.md"
    summary_markdown.write_text(
        "# DSA Reproduction Report\n\n"
        f"- Result kind: `{result_kind}`\n"
        f"- Gate-sweep source: `{gate_path}`\n"
        f"- Figure: `{figure2.name}`\n"
        f"- Reference tables loaded: {len(tables)}\n",
        encoding="utf-8",
    )
    return ReportOutputs(
        figure2=figure2, summary_markdown=summary_markdown, summary_json=summary_json
    )
