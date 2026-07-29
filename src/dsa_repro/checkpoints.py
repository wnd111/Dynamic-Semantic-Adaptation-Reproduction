from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(path: Path, module: nn.Module, metadata: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "state_dict": module.state_dict(),
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(path: Path, module: nn.Module) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema_version in {path}")
    module.load_state_dict(payload["state_dict"], strict=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    return dict(metadata)
