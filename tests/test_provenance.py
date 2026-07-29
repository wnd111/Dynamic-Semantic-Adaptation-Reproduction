from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsa_repro.config import load_config
from dsa_repro.provenance import write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = PROJECT_ROOT / "configs" / "paper_a100.yaml"


def test_manifest_records_result_kind_and_resolved_config(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, load_config(PAPER_CONFIG), "synthetic_smoke")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["result_kind"] == "synthetic_smoke"
    assert payload["config"]["anchor"]["gate_threshold"] == pytest.approx(0.85)
    assert payload["environment"]["python"]
    assert "torch" in payload["packages"]


def test_manifest_rejects_unknown_result_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="result_kind"):
        write_manifest(tmp_path, load_config(PAPER_CONFIG), "unknown")


def test_manifest_does_not_serialize_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

    path = write_manifest(tmp_path, load_config(PAPER_CONFIG), "measured")
    text = path.read_text(encoding="utf-8")

    assert "hf_secret_value" not in text
    assert "sk-secret-value" not in text
    assert '"HF_TOKEN": "present"' in text
    assert '"OPENAI_API_KEY": "present"' in text
