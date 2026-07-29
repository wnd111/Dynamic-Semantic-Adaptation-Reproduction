from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from scripts.package_release import package_release
from scripts.verify_archive import verify_archive


def test_release_archive_excludes_secrets_caches_and_git(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ["README.md", "pyproject.toml", "LICENSE"]:
        (source / name).write_text(name, encoding="utf-8")
    (source / "src").mkdir()
    (source / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("secret", encoding="utf-8")
    (source / "artifacts").mkdir()
    (source / "artifacts" / "token.txt").write_text("sk-secret", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "module.pyc").write_bytes(b"cache")
    (source / "src" / "demo.egg-info").mkdir()
    (source / "src" / "demo.egg-info" / "PKG-INFO").write_text("generated", encoding="utf-8")
    archive = tmp_path / "release.zip"

    checksum_path = package_release(source, archive)
    report = verify_archive(archive, required=("README.md", "pyproject.toml", "LICENSE"))

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert report.ok
    assert not any(
        ".git" in name or "artifacts" in name or "__pycache__" in name or "egg-info" in name
        for name in names
    )
    assert (
        checksum_path.read_text(encoding="ascii").split()[0]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
