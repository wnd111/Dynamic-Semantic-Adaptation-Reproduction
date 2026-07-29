from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsa_repro.judge import JudgeClient, JudgeParseError, parse_judge_response


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **request: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(output_text="Score: 8\nJustification: Relevant and complete.")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_judge_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        JudgeClient(model="gpt-4-1106-preview")


def test_parser_accepts_appendix_c_format() -> None:
    parsed = parse_judge_response("Score: 7\nJustification: Mostly correct.")

    assert parsed.score == 7
    assert parsed.justification == "Mostly correct."


def test_parser_rejects_out_of_range_score() -> None:
    with pytest.raises(JudgeParseError, match="1 to 10"):
        parse_judge_response("Score: 11\nJustification: invalid")


def test_dry_run_writes_request_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake = FakeClient()
    judge = JudgeClient(model="gpt-4-1106-preview", cache_dir=tmp_path, client=fake)

    request = judge.score("Explain gravity", "Gravity attracts masses.", dry_run=True)

    assert request["status"] == "dry_run"
    assert fake.responses.calls == 0
    assert Path(request["request_path"]).is_file()


def test_judge_cache_prevents_duplicate_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake = FakeClient()
    judge = JudgeClient(model="gpt-4-1106-preview", cache_dir=tmp_path, client=fake)

    first = judge.score("Explain gravity", "Gravity attracts masses.")
    second = judge.score("Explain gravity", "Gravity attracts masses.")

    assert first["score"] == 8
    assert second["score"] == 8
    assert fake.responses.calls == 1
    assert json.loads(Path(first["cache_path"]).read_text(encoding="utf-8"))["score"] == 8
