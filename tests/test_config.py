from pathlib import Path

import pytest

from dsa_repro.config import ConfigError, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = PROJECT_ROOT / "configs" / "paper_a100.yaml"
CURRENT_CONFIG = PROJECT_ROOT / "configs" / "current_compatible.yaml"


def test_paper_defaults_match_manuscript_tables() -> None:
    cfg = load_config(PAPER_CONFIG)

    assert cfg.model.model_id == "meta-llama/Llama-2-7b-hf"
    assert cfg.model.dtype == "float16"
    assert cfg.runtime.batch_size == 1
    assert cfg.runtime.context_length == 512
    assert cfg.anchor.gate_threshold == pytest.approx(0.85)
    assert cfg.anchor.drift_threshold == pytest.approx(0.15)
    assert cfg.anchor.max_chain == 3
    assert cfg.pruning.high_threshold == pytest.approx(0.7)
    assert cfg.pruning.low_threshold == pytest.approx(0.3)
    assert cfg.pruning.topk_fraction == pytest.approx(0.5)
    assert cfg.feedback.local_interval == 20
    assert cfg.feedback.e2e_interval == 100
    assert cfg.precision.thresholds == pytest.approx((0.7, 0.5, 0.3))
    assert cfg.training.calibration_sequences == 2048
    assert cfg.training.seed == 42
    assert cfg.evaluation.paired_seeds == (11, 22, 33, 44, 55)
    assert cfg.evaluation.judge_model == "gpt-4-1106-preview"
    assert cfg.evaluation.judge_comparability == "paper_exact"


def test_current_profile_is_explicitly_noncomparable() -> None:
    cfg = load_config(CURRENT_CONFIG)

    assert cfg.evaluation.judge_model == "gpt-5.6-sol"
    assert cfg.evaluation.judge_comparability == "current_noncomparable"


def test_invalid_precision_threshold_order_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        PAPER_CONFIG.read_text(encoding="utf-8").replace(
            "thresholds: [0.7, 0.5, 0.3]", "thresholds: [0.3, 0.5, 0.7]"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="strictly descending"):
        load_config(bad)


def test_batch_size_above_one_is_rejected_for_paper_protocol(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        PAPER_CONFIG.read_text(encoding="utf-8").replace("batch_size: 1", "batch_size: 2"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="batch_size=1"):
        load_config(bad)
