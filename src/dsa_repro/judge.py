from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json

SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the quality of AI assistant responses. "
    "Evaluate relevance, coherence, accuracy, and completeness. Return an integer "
    "score from 1 to 10 and a brief justification using exactly this format:\n"
    "Score: [integer]\nJustification: [2-3 sentences]"
)


class JudgeParseError(ValueError):
    pass


@dataclass(frozen=True)
class JudgeResult:
    score: int
    justification: str


def parse_judge_response(text: str) -> JudgeResult:
    score_match = re.search(r"(?im)^\s*Score\s*:\s*\[?(\d+)\]?\s*$", text)
    justification_match = re.search(r"(?ims)^\s*Justification\s*:\s*(.+?)\s*$", text)
    if not score_match or not justification_match:
        raise JudgeParseError("judge response must contain Score and Justification lines")
    score = int(score_match.group(1))
    if not 1 <= score <= 10:
        raise JudgeParseError("judge score must be an integer from 1 to 10")
    return JudgeResult(score=score, justification=justification_match.group(1).strip())


class JudgeClient:
    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path = Path("artifacts/judge-cache"),
        client: Any | None = None,
        max_output_tokens: int = 512,
        allow_missing_key_for_dry_run: bool = False,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and not allow_missing_key_for_dry_run:
            raise RuntimeError("OPENAI_API_KEY is required for LLM-as-a-judge evaluation")
        if client is None and api_key:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.max_output_tokens = int(max_output_tokens)

    @staticmethod
    def _prompt(instruction: str, response: str) -> str:
        return f"**Instruction:** {instruction}\n\n**Response:** {response}"

    def _request(self, instruction: str, response: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": self._prompt(instruction, response),
            "max_output_tokens": self.max_output_tokens,
        }
        if self.model.startswith("gpt-4"):
            request["temperature"] = 0.0
        return request

    @staticmethod
    def _hash(request: dict[str, Any]) -> str:
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def score(self, instruction: str, response: str, *, dry_run: bool = False) -> dict[str, Any]:
        request = self._request(instruction, response)
        request_hash = self._hash(request)
        request_path = self.cache_dir / "requests" / f"{request_hash}.json"
        cache_path = self.cache_dir / "responses" / f"{request_hash}.json"
        atomic_write_json(request_path, request)
        if dry_run:
            return {
                "status": "dry_run",
                "request_hash": request_hash,
                "request_path": str(request_path),
            }
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is required for a live judge request")
        response_object = self.client.responses.create(**request)
        raw_text = str(response_object.output_text)
        parsed = parse_judge_response(raw_text)
        payload = {
            "status": "completed",
            "model": self.model,
            "score": parsed.score,
            "justification": parsed.justification,
            "raw_response": raw_text,
            "request_hash": request_hash,
            "request_path": str(request_path),
            "cache_path": str(cache_path),
        }
        atomic_write_json(cache_path, payload)
        return payload
