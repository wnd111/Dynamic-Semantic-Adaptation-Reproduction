from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    name: str
    source_type: str
    source: str
    revision: str
    source_url: str
    config: str | None
    split: str
    sample_count: int
    selection_seed: int
    selection: str
    required_fields: tuple[str, ...]
    license: str
    notes: str


def load_manifest(path: Path) -> DatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported dataset manifest schema in {path}")
    payload["required_fields"] = tuple(payload["required_fields"])
    return DatasetManifest(**payload)


def deterministic_select(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError(f"count must be in [1, {len(rows)}]")
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def validate_rows(rows: Iterable[Mapping[str, Any]], required_fields: tuple[str, ...]) -> None:
    for index, row in enumerate(rows):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"row {index} is missing required fields: {', '.join(missing)}")


def _download_text(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "dsa-repro/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    destination.write_bytes(content)
    return destination


def _load_url_rows(manifest: DatasetManifest, cache_dir: Path) -> list[dict[str, Any]]:
    extension = ".jsonl" if manifest.source_type == "jsonl" else ".json"
    raw_path = _download_text(manifest.source_url, cache_dir / f"{manifest.name}{extension}")
    text = raw_path.read_text(encoding="utf-8")
    if manifest.source_type == "jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"{manifest.name} JSON source must contain a list")
    return [dict(item) for item in payload]


def _load_hf_rows(manifest: DatasetManifest, cache_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        manifest.source,
        manifest.config,
        split=manifest.split,
        revision=manifest.revision,
        cache_dir=str(cache_dir),
    )
    return [dict(row) for row in dataset]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(manifest_path: Path, output_dir: Path) -> Path:
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "download_cache"
    if manifest.source_type == "hf_dataset":
        rows = _load_hf_rows(manifest, cache_dir)
    elif manifest.source_type in {"json", "jsonl"}:
        rows = _load_url_rows(manifest, cache_dir)
    else:
        raise ValueError(f"unsupported source_type: {manifest.source_type}")
    validate_rows(rows, manifest.required_fields)
    if manifest.selection == "seeded_sample":
        rows = deterministic_select(
            rows,
            count=manifest.sample_count,
            seed=manifest.selection_seed,
        )
    elif len(rows) != manifest.sample_count:
        raise ValueError(
            f"{manifest.name} expected {manifest.sample_count} rows, received {len(rows)}"
        )

    prepared = output_dir / "prepared.jsonl"
    with prepared.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_write_json(
        output_dir / "dataset_record.json",
        {
            "name": manifest.name,
            "source": manifest.source,
            "revision": manifest.revision,
            "split": manifest.split,
            "samples": len(rows),
            "sha256": _sha256(prepared),
            "download_cache": [
                {"path": str(path.relative_to(output_dir)), "sha256": _sha256(path)}
                for path in sorted(cache_dir.rglob("*"))
                if path.is_file()
            ],
        },
    )
    return prepared
