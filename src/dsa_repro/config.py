from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a reproduction configuration violates the paper protocol."""


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    dtype: str
    backend: str


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    context_length: int
    max_new_tokens: int
    warmup_steps: int
    measurement_steps: int
    require_a100: bool


@dataclass(frozen=True)
class AnchorConfig:
    gate_threshold: float
    drift_threshold: float
    max_chain: int
    approximation_error: float
    projection_dim: int
    gate_slope: float
    residual_blend: float


@dataclass(frozen=True)
class PruningConfig:
    high_threshold: float
    low_threshold: float
    window_fraction: float
    topk_fraction: float
    recency_weight: float
    deterministic: bool
    refresh_interval: int
    refresh_precision_changes: int
    refresh_error_sq: float


@dataclass(frozen=True)
class PrecisionConfig:
    thresholds: tuple[float, float, float]
    group_size: int
    activation_quantization: bool
    resident_packs: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackConfig:
    local_interval: int
    e2e_interval: int
    default_eout: float
    target_band: tuple[float, float]
    momentum: float
    step_size: float
    complexity_margin: float


@dataclass(frozen=True)
class TrainingConfig:
    calibration_dataset: str
    calibration_split: str
    calibration_sequences: int
    calibration_context_length: int
    auxiliary_dataset: str
    auxiliary_sequences: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    datasets: tuple[str, ...]
    samples: Mapping[str, int]
    paired_seeds: tuple[int, ...]
    judge_model: str
    judge_comparability: str
    judge_temperature: float
    judge_max_tokens: int


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    profile: str
    model: ModelConfig
    runtime: RuntimeConfig
    anchor: AnchorConfig
    pruning: PruningConfig
    precision: PrecisionConfig
    feedback: FeedbackConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing mapping section: {name}")
    return value


def _tuple3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{name} must contain exactly three values")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _tuple2(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{name} must contain exactly two values")
    return float(value[0]), float(value[1])


def load_config(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")

    model = _section(raw, "model")
    runtime = _section(raw, "runtime")
    anchor = _section(raw, "anchor")
    pruning = _section(raw, "pruning")
    precision = _section(raw, "precision")
    feedback = _section(raw, "feedback")
    training = _section(raw, "training")
    evaluation = _section(raw, "evaluation")

    cfg = ExperimentConfig(
        schema_version=int(raw.get("schema_version", 1)),
        profile=str(raw.get("profile", config_path.stem)),
        model=ModelConfig(
            model_id=str(model["model_id"]),
            dtype=str(model["dtype"]),
            backend=str(model["backend"]),
        ),
        runtime=RuntimeConfig(
            batch_size=int(runtime["batch_size"]),
            context_length=int(runtime["context_length"]),
            max_new_tokens=int(runtime["max_new_tokens"]),
            warmup_steps=int(runtime["warmup_steps"]),
            measurement_steps=int(runtime["measurement_steps"]),
            require_a100=bool(runtime["require_a100"]),
        ),
        anchor=AnchorConfig(
            gate_threshold=float(anchor["gate_threshold"]),
            drift_threshold=float(anchor["drift_threshold"]),
            max_chain=int(anchor["max_chain"]),
            approximation_error=float(anchor["approximation_error"]),
            projection_dim=int(anchor["projection_dim"]),
            gate_slope=float(anchor["gate_slope"]),
            residual_blend=float(anchor["residual_blend"]),
        ),
        pruning=PruningConfig(
            high_threshold=float(pruning["high_threshold"]),
            low_threshold=float(pruning["low_threshold"]),
            window_fraction=float(pruning["window_fraction"]),
            topk_fraction=float(pruning["topk_fraction"]),
            recency_weight=float(pruning["recency_weight"]),
            deterministic=bool(pruning["deterministic"]),
            refresh_interval=int(pruning["refresh_interval"]),
            refresh_precision_changes=int(pruning["refresh_precision_changes"]),
            refresh_error_sq=float(pruning["refresh_error_sq"]),
        ),
        precision=PrecisionConfig(
            thresholds=_tuple3(precision["thresholds"], "precision.thresholds"),
            group_size=int(precision["group_size"]),
            activation_quantization=bool(precision["activation_quantization"]),
            resident_packs=tuple(str(item) for item in precision["resident_packs"]),
        ),
        feedback=FeedbackConfig(
            local_interval=int(feedback["local_interval"]),
            e2e_interval=int(feedback["e2e_interval"]),
            default_eout=float(feedback["default_eout"]),
            target_band=_tuple2(feedback["target_band"], "feedback.target_band"),
            momentum=float(feedback["momentum"]),
            step_size=float(feedback["step_size"]),
            complexity_margin=float(feedback["complexity_margin"]),
        ),
        training=TrainingConfig(
            calibration_dataset=str(training["calibration_dataset"]),
            calibration_split=str(training["calibration_split"]),
            calibration_sequences=int(training["calibration_sequences"]),
            calibration_context_length=int(training["calibration_context_length"]),
            auxiliary_dataset=str(training["auxiliary_dataset"]),
            auxiliary_sequences=int(training["auxiliary_sequences"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            seed=int(training["seed"]),
        ),
        evaluation=EvaluationConfig(
            datasets=tuple(str(item) for item in evaluation["datasets"]),
            samples={str(k): int(v) for k, v in evaluation["samples"].items()},
            paired_seeds=tuple(int(item) for item in evaluation["paired_seeds"]),
            judge_model=str(evaluation["judge_model"]),
            judge_comparability=str(evaluation["judge_comparability"]),
            judge_temperature=float(evaluation["judge_temperature"]),
            judge_max_tokens=int(evaluation["judge_max_tokens"]),
        ),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
    if cfg.runtime.batch_size != 1:
        raise ConfigError("paper protocol requires batch_size=1")
    if cfg.runtime.context_length <= 0 or cfg.runtime.max_new_tokens <= 0:
        raise ConfigError("context_length and max_new_tokens must be positive")
    if not 0.0 < cfg.anchor.gate_threshold <= 1.0:
        raise ConfigError("anchor.gate_threshold must be in (0, 1]")
    if not 0.0 <= cfg.anchor.drift_threshold <= 1.0:
        raise ConfigError("anchor.drift_threshold must be in [0, 1]")
    if cfg.anchor.max_chain < 1:
        raise ConfigError("anchor.max_chain must be positive")
    if not 0.0 <= cfg.pruning.low_threshold < cfg.pruning.high_threshold <= 1.0:
        raise ConfigError("pruning thresholds must satisfy 0 <= low < high <= 1")
    if not (
        cfg.precision.thresholds[0] > cfg.precision.thresholds[1] > cfg.precision.thresholds[2]
    ):
        raise ConfigError("precision thresholds must be strictly descending")
    if cfg.precision.group_size <= 0:
        raise ConfigError("precision.group_size must be positive")
    low, high = cfg.feedback.target_band
    if not 0.0 <= low < high <= 1.0:
        raise ConfigError("feedback.target_band must satisfy 0 <= low < high <= 1")
    if cfg.evaluation.judge_comparability not in {"paper_exact", "current_noncomparable"}:
        raise ConfigError("unsupported judge_comparability")
