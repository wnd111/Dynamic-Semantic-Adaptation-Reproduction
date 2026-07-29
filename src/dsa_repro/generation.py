from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationRecord:
    token_ids: list[int]
    traces: list[dict[str, object]]
    backend: str
    result_kind: str
    transformers_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
