from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from .controller import RuntimeController
from .precision import PrecisionPath


def _flatten_hotpot_context(context: Any) -> str:
    if not isinstance(context, Mapping):
        return ""
    titles = context.get("title", [])
    sentence_groups = context.get("sentences", [])
    blocks: list[str] = []
    for index, sentences in enumerate(sentence_groups):
        title = str(titles[index]) if index < len(titles) else ""
        if isinstance(sentences, (list, tuple)):
            body = " ".join(str(sentence) for sentence in sentences)
        else:
            body = str(sentences)
        blocks.append(f"{title}: {body}".strip(": "))
    return "\n".join(blocks)


def extract_prompt_reference(
    task: str,
    row: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Convert one prepared benchmark row into a stable prompt/reference pair."""
    normalized = task.lower().replace("-", "")
    if normalized == "alpacaeval":
        return str(row["instruction"]), {}
    if normalized == "hotpotqa":
        context = _flatten_hotpot_context(row.get("context"))
        question = str(row["question"])
        prompt = (
            "Answer the question using the supplied context. Give a concise answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return prompt, {"answer": str(row["answer"])}
    if normalized == "vicuna80":
        prompt = row.get(
            "instruction",
            row.get("question", row.get("prompt", row.get("text"))),
        )
        if prompt is None:
            raise ValueError("Vicuna-80 row is missing instruction/question/prompt/text")
        if isinstance(prompt, list):
            prompt = "\n".join(str(turn) for turn in prompt)
        answer = row.get("answer", row.get("reference", row.get("output", "")))
        return str(prompt), {"answer": str(answer)}
    if normalized == "asqa":
        prompt = row.get("ambiguous_question", row.get("question"))
        if prompt is None:
            raise ValueError("ASQA row is missing ambiguous_question/question")
        reference = {
            key: row[key] for key in ("short_answers", "qa_pairs", "long_answer") if key in row
        }
        return str(prompt), reference
    raise ValueError(f"unsupported task: {task}")


def summarize_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {}
    keys = sorted({key for row in materialized for key in row})
    return {key: fmean(float(row[key]) for row in materialized if key in row) for key in keys}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def configure_variant(controller: RuntimeController, variant: str) -> None:
    """Configure the full system or exactly one Table-7 style ablation."""
    controller.approximation_enabled = True
    controller.pruning_enabled = True
    controller.precision_override = None
    if variant == "full":
        return
    if variant == "no-approximation":
        controller.approximation_enabled = False
        return
    if variant == "no-pruning":
        controller.pruning_enabled = False
        return
    if variant == "fp16-only":
        controller.precision_override = PrecisionPath.FP16
        return
    raise ValueError(f"unsupported variant: {variant}")
