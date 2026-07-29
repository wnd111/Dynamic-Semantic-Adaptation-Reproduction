from __future__ import annotations

import pytest

from dsa_repro.metrics import normalize_answer, rouge_l, score_example, token_f1


def test_answer_normalization_removes_articles_punctuation_and_case() -> None:
    assert normalize_answer("The Eiffel-Tower!") == "eiffel tower"


def test_token_f1_handles_partial_overlap() -> None:
    assert token_f1("red green blue", "red blue") == pytest.approx(0.8)


def test_rouge_l_uses_longest_common_subsequence() -> None:
    assert rouge_l("red b c", "red x c") == pytest.approx(2 / 3)


def test_hotpot_normalization_matches_expected() -> None:
    score = score_example("hotpotqa", "The Eiffel Tower.", {"answer": "eiffel tower"})

    assert score["exact_match"] == 1.0
    assert score["f1"] == 1.0
    assert score["rouge_l"] == 1.0


def test_vicuna_reports_em_and_f1_for_supplied_reference() -> None:
    score = score_example("vicuna80", "blue whale", {"answer": "the blue whale"})

    assert score == {"exact_match": 1.0, "f1": 1.0}


def test_asqa_scores_short_answers_and_disambiguation_pairs() -> None:
    reference = {
        "short_answers": ["Mercury"],
        "qa_pairs": [
            {"short_answers": ["Mercury"]},
            {"short_answers": ["Venus"]},
        ],
        "long_answer": "Mercury is closest to the Sun, followed by Venus.",
    }
    score = score_example(
        "asqa",
        "Mercury is closest to the Sun, followed by Venus.",
        reference,
    )

    assert score["str_em"] == 1.0
    assert score["disambig_f1"] == 1.0
    assert score["rouge_l"] == 1.0
