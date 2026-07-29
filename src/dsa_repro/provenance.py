from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ExperimentConfig

RESULT_KINDS = {"measured", "paper_reference", "synthetic_smoke"}
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "scipy",
    "pandas",
    "PyYAML",
    "matplotlib",
    "typer",
)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(project_dir: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(project_dir), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _cuda_state() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if available else None,
            "capability": list(torch.cuda.get_device_capability(0)) if available else None,
        }
    except (ImportError, RuntimeError):
        return {"available": False, "torch_cuda": None, "device": None, "capability": None}


def collect_environment(
    config: ExperimentConfig, result_kind: str, project_dir: Path
) -> dict[str, Any]:
    if result_kind not in RESULT_KINDS:
        raise ValueError("result_kind must be measured, paper_reference, or synthetic_smoke")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_kind": result_kind,
        "config": config.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "packages": {name: _version(name) for name in TRACKED_PACKAGES},
        "cuda": _cuda_state(),
        "git": _git_state(project_dir),
        "credentials": {
            "HF_TOKEN": "present" if os.getenv("HF_TOKEN") else "absent",
            "OPENAI_API_KEY": "present" if os.getenv("OPENAI_API_KEY") else "absent",
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)
    return path


def write_manifest(run_dir: Path, config: ExperimentConfig, result_kind: str) -> Path:
    project_dir = Path(__file__).resolve().parents[2]
    payload = collect_environment(config, result_kind, project_dir)
    return atomic_write_json(Path(run_dir) / "run_manifest.json", payload)
