from __future__ import annotations

import argparse
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REQUIRED = (
    "README.md",
    "README_EN.md",
    "pyproject.toml",
    "environment-a100.yml",
    "LICENSE",
    "configs/paper_a100.yaml",
    "src/dsa_repro/controller.py",
    "paper_reference/table10_quality.csv",
)
FORBIDDEN_PARTS = {".git", "artifacts", "__pycache__", ".pytest_cache", ".env"}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
)


@dataclass(frozen=True)
class ArchiveReport:
    ok: bool
    issues: tuple[str, ...]
    files: int


def verify_archive(path: Path, required: Sequence[str] = DEFAULT_REQUIRED) -> ArchiveReport:
    path = Path(path)
    issues: list[str] = []
    with zipfile.ZipFile(path) as bundle:
        corrupt = bundle.testzip()
        if corrupt:
            issues.append(f"corrupt member: {corrupt}")
        names = bundle.namelist()
        name_set = set(names)
        for item in required:
            if item not in name_set:
                issues.append(f"missing required member: {item}")
        for name in names:
            parts = set(Path(name).parts)
            if parts & FORBIDDEN_PARTS:
                issues.append(f"forbidden member: {name}")
            info = bundle.getinfo(name)
            if info.file_size <= 2_000_000 and Path(name).suffix.lower() in {
                ".py",
                ".md",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".sh",
                ".ps1",
            }:
                content = bundle.read(name)
                if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                    issues.append(f"possible credential in: {name}")
    return ArchiveReport(ok=not issues, issues=tuple(issues), files=len(names))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a DSA reproduction ZIP.")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    report = verify_archive(args.archive)
    print(f"ok={report.ok} files={report.files}")
    for issue in report.issues:
        print(issue)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
