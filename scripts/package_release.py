from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "checkpoints",
    "download_cache",
    "raw",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip", ".sha256"}
EXCLUDED_NAMES = {".env"}


def _included(path: Path, source: Path, output: Path) -> bool:
    relative = path.relative_to(source)
    if path.resolve() == output.resolve():
        return False
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if any("egg-info" in part for part in relative.parts):
        return False
    return True


def package_release(source: Path, output: Path) -> Path:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file() and _included(path, source, output):
                bundle.write(path, path.relative_to(source).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return checksum_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean DSA reproduction ZIP.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checksum = package_release(args.source, args.output)
    print(checksum)


if __name__ == "__main__":
    main()
