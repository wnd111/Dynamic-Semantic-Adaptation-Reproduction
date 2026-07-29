from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dsa_repro.cli import app

runner = CliRunner()


def test_smoke_cli_writes_synthetic_manifest_and_generation(tmp_path: Path) -> None:
    output = tmp_path / "smoke"

    result = runner.invoke(app, ["smoke", "--output", str(output), "--max-new-tokens", "2"])

    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    generation = json.loads((output / "generation.json").read_text(encoding="utf-8"))
    assert manifest["result_kind"] == "synthetic_smoke"
    assert generation["result_kind"] == "synthetic_smoke"
    assert len(generation["token_ids"]) == 2


def test_smoke_cli_refuses_existing_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")

    result = runner.invoke(app, ["smoke", "--output", str(output)])

    assert result.exit_code != 0
    assert "not empty" in result.output
