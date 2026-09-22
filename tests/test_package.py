"""The submission must be reproducible and exclude local data and linked files."""
import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.package import build_archive


def put(root, name, content="example"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_only_source_formats_are_included_without_traversing_links(tmp_path):
    root = tmp_path / "project"
    expected = {"README.md", ".env.example", "seif/app.py", "config/policies.yaml",
                "deploy/ner/requirements-ner.txt", "docs/check.json", "web/app.js", "scripts/run.sh"}
    for name in expected:
        put(root, name)
    for name in (".env", "seif/.env", "seif/dump.rdb", "seif/__pycache__/cache.py",
                 "scripts/venv/secret.py", "local-data/requests.jsonl", "output/credentials.json",
                 "web/database.sqlite", "deploy/key.pem", "docs/private.zip"):
        put(root, name, "DO NOT INCLUDE")
    secret = put(tmp_path, "outside/secret.py", "LINKED PRIVATE DATA")
    (root / "seif/linked.py").symlink_to(secret)
    (root / "scripts/linked").symlink_to(secret.parent, target_is_directory=True)
    archive = build_archive(root, root / "output/source.zip")
    with ZipFile(archive) as packaged:
        assert set(packaged.namelist()) == expected
        assert all(b"PRIVATE" not in packaged.read(name) for name in packaged.namelist())


def test_archive_is_reproducible_across_file_creation_order_and_mtime(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    names = ["README.md", "seif/app.py", "scripts/run.sh"]
    for root, order in ((left, names), (right, list(reversed(names)))):
        for name in order:
            put(root, name, name)
    a = build_archive(left, left / "output/source.zip")
    b = build_archive(right, right / "output/source.zip")
    assert a.read_bytes() == b.read_bytes()
    with ZipFile(a) as archive:
        assert archive.getinfo("scripts/run.sh").external_attr >> 16 == 0o100755


def test_failed_build_preserves_previous_archive_and_removes_temporary(tmp_path, monkeypatch):
    put(tmp_path, "seif/app.py")
    target = put(tmp_path, "output/source.zip", "previous archive")

    def fail(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="simulated"):
        build_archive(tmp_path, target)
    assert target.read_text() == "previous archive"
    assert list(target.parent.iterdir()) == [target]


def test_output_cannot_replace_source_or_follow_symlink(tmp_path):
    source = put(tmp_path, "seif/app.py", "original source")
    with pytest.raises(ValueError):
        build_archive(tmp_path, tmp_path / "seif/../seif/app.py")
    link = tmp_path / "submission.zip"
    link.symlink_to(source)
    with pytest.raises(ValueError):
        build_archive(tmp_path, link)
    assert source.read_text() == "original source"


def test_import_does_not_create_or_replace_archive(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/package.py"
    copied = put(tmp_path, "scripts/package.py", source.read_text())
    runpy.run_path(str(copied), run_name="package_import_test")
    assert not (tmp_path / "output").exists()
