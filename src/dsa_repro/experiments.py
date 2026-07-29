from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers

from .config import ExperimentConfig
from .provenance import atomic_write_json

STAGES = (
    "preflight",
    "prepare",
    "calibrate",
    "train",
    "evaluate",
    "latency",
    "ablate",
    "sweep",
    "report",
)


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    environment: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def preflight(config: ExperimentConfig, *, strict: bool = True) -> PreflightReport:
    issues: list[str] = []
    warnings: list[str] = []
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    if config.runtime.require_a100:
        if not cuda_available:
            issues.append("CUDA is unavailable; the paper protocol requires one NVIDIA A100 80GB")
        elif "A100" not in str(device_name):
            issues.append(f"GPU is {device_name}; the paper protocol requires an NVIDIA A100 80GB")
        else:
            total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_gib < 75:
                issues.append(f"A100 memory is {total_gib:.1f} GiB; an 80GB device is required")
    if torch.__version__.split("+")[0] != "2.1.0":
        warnings.append(f"paper used torch 2.1.0; detected {torch.__version__}")
    if transformers.__version__ != "4.36.0":
        warnings.append(f"paper used transformers 4.36.0; detected {transformers.__version__}")
    if not os.getenv("HF_TOKEN"):
        issues.append("HF_TOKEN is absent; restricted LLaMA-2 weights cannot be loaded")
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append(
            "OPENAI_API_KEY is absent; generation can run but live judge scoring cannot"
        )
    report = PreflightReport(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warnings),
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": cuda_available,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": device_name,
        },
    )
    if strict and not report.ok:
        raise RuntimeError("preflight failed: " + "; ".join(report.issues))
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentRunner:
    """Checksum-based resumability without overwriting completed artifacts."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "stage_state.json"
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"schema_version": 1, "stages": {}}

    def mark_completed(self, stage: str, artifacts: Iterable[Path]) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        records = []
        for artifact in artifacts:
            path = Path(artifact).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append({"path": str(path), "sha256": _sha256(path)})
        self.state["stages"][stage] = {"status": "completed", "artifacts": records}
        atomic_write_json(self.state_path, self.state)

    def is_completed(self, stage: str) -> bool:
        record = self.state.get("stages", {}).get(stage)
        if not record or record.get("status") != "completed":
            return False
        for artifact in record.get("artifacts", []):
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                return False
        return True


def build_runbook(config_path: Path, run_dir: Path) -> dict[str, object]:
    config_path = Path(config_path)
    run_dir = Path(run_dir)
    commands = [
        f"dsa-repro preflight --config {config_path}",
        *[
            f"dsa-repro prepare-data {name} --output {run_dir / 'data'}"
            for name in ("alpacaeval", "vicuna80", "hotpotqa", "asqa")
        ],
        f"dsa-repro prepare-data sharegpt512 --output {run_dir / 'data'}",
        (
            f"dsa-repro calibrate --config {config_path} --auxiliary-data "
            f"{run_dir / 'data' / 'sharegpt512' / 'prepared.jsonl'} "
            f"--output {run_dir / 'calibration'}"
        ),
        (
            f"dsa-repro train --config {config_path} "
            f"--traces {run_dir / 'calibration'} --output {run_dir / 'checkpoints'}"
        ),
        (
            f"dsa-repro evaluate --config {config_path} "
            f"--checkpoint {run_dir / 'checkpoints' / 'auxiliary.pt'} "
            f"--data-root {run_dir / 'data'} --output {run_dir / 'evaluation'}"
        ),
        (
            f"dsa-repro latency --config {config_path} "
            f"--checkpoint {run_dir / 'checkpoints' / 'auxiliary.pt'} "
            f"--output {run_dir / 'latency'}"
        ),
        (
            f"dsa-repro ablate --config {config_path} "
            f"--checkpoint {run_dir / 'checkpoints' / 'auxiliary.pt'} "
            f"--data-root {run_dir / 'data'} --output {run_dir / 'ablation'}"
        ),
        (
            f"dsa-repro sweep --config {config_path} "
            f"--checkpoint {run_dir / 'checkpoints' / 'auxiliary.pt'} "
            f"--data-root {run_dir / 'data'} --output {run_dir / 'gate-sweep'}"
        ),
        (f"dsa-repro report --run-dir {run_dir / 'gate-sweep'} --output {run_dir / 'report'}"),
    ]
    return {
        "schema_version": 1,
        "config": str(config_path),
        "run_dir": str(run_dir),
        "commands": commands,
    }
