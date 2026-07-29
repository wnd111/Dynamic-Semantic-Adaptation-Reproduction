from __future__ import annotations

import collections
import re
import string
from collections.abc import Iterable, Mapping
from typing import Any


def normalize_answer(text: str) -> str:
    lowered = text.lower().replace("-", " ")
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    common = collections.Counter(predicted_tokens) & collections.Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    target = normalize_answer(reference).split()
    if not predicted or not target:
        return float(predicted == target)
    table = [[0] * (len(target) + 1) for _ in range(len(predicted) + 1)]
    for i, left in enumerate(predicted, start=1):
        for j, right in enumerate(target, start=1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if left == right else max(table[i - 1][j], table[i][j - 1])
            )
    lcs = table[-1][-1]
    precision = lcs / len(predicted)
    recall = lcs / len(target)
    return 2.0 * precision * recall / (precision + recall)


def _answers(reference: Mapping[str, Any]) -> list[str]:
    value = reference.get("answer", reference.get("answers", []))
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping) and "text" in value:
        value = value["text"]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise ValueError("reference does not contain a usable answer")


def _best(prediction: str, answers: list[str], metric: Any) -> float:
    if not answers:
        return 0.0
    return max(float(metric(prediction, answer)) for answer in answers)


def score_example(task: str, prediction: str, reference: Mapping[str, Any]) -> dict[str, float]:
    task = task.lower()
    if task in {"hotpotqa", "hotpot_qa"}:
        answers = _answers(reference)
        return {
            "exact_match": _best(prediction, answers, exact_match),
            "f1": _best(prediction, answers, token_f1),
            "rouge_l": _best(prediction, answers, rouge_l),
        }
    if task in {"vicuna80", "vicuna-80"}:
        answers = _answers(reference)
        return {
            "exact_match": _best(prediction, answers, exact_match),
            "f1": _best(prediction, answers, token_f1),
        }
    if task == "asqa":
        short_answers = [str(item) for item in reference.get("short_answers", [])]
        pairs = reference.get("qa_pairs", [])
        pair_scores = []
        for pair in pairs:
            pair_answers = [str(item) for item in pair.get("short_answers", [])]
            pair_scores.append(
                _best(
                    prediction,
                    pair_answers,
                    lambda p, a: float(normalize_answer(a) in normalize_answer(p)),
                )
            )
        long_answer = str(reference.get("long_answer", ""))
        return {
            "str_em": _best(
                prediction,
                short_answers,
                lambda p, a: float(normalize_answer(a) in normalize_answer(p)),
            ),
            "disambig_f1": sum(pair_scores) / len(pair_scores) if pair_scores else 0.0,
            "rouge_l": rouge_l(prediction, long_answer),
        }
    raise ValueError(f"task-specific metrics are not defined for {task}")
