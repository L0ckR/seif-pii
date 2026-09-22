"""Validate source isolation and GitLab report semantics for the local analyzer."""
import shutil
import stat
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from scripts.code_quality import extract_python, run_profile, summarize, validate_report


def issue(path="seif/app.py", **overrides):
    return {
        "description": "A diagnostic", "check_name": "C901", "fingerprint": "unique",
        "severity": "major", "location": {"path": path, "lines": {"begin": 2}}, **overrides,
    }


def test_extraction_analyzes_source_without_opening_golden_or_logs(tmp_path):
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("seif/app.py", "x = 1\n")
        source.writestr("tests/test_app.py", "assert True\n")
        source.writestr("datasets/private.py", "not valid python, not source")
        source.writestr("local-data/capture.jsonl", "not source")
        source.writestr(".env", "SECRET=never_analyze")
    destination = tmp_path / "analysis"
    manifest = extract_python(archive, destination)
    assert set(manifest) == {"seif/app.py", "tests/test_app.py"}
    assert not (destination / "datasets").exists()
    assert manifest["seif/app.py"]["lines"] == 1


@pytest.mark.parametrize("name", ["../seif/app.py", "/seif/app.py", "seif/../app.py", "seif\\app.py"])
def test_rejects_archive_path_traversal(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as source:
        source.writestr(name, "x = 1")
    with pytest.raises(ValueError, match="non-canonical"):
        extract_python(archive, tmp_path / "analysis")


@pytest.mark.parametrize("override", [
    {"severity": "high"}, {"fingerprint": ""},
    {"severity": []}, {"location": None}, {"location": {"lines": 2}},
    {"location": {"path": [], "lines": {"begin": 2}}},
    {"location": {"path": "../app.py", "lines": {"begin": 2}}},
    {"location": {"path": "seif/app.py", "lines": {"begin": 0}}},
    {"location": {"path": "seif/app.py", "lines": {"begin": True}}},
])
def test_rejects_invalid_gitlab_findings(override):
    with pytest.raises(ValueError):
        validate_report([issue(**override)], {"seif/app.py"})


def test_accepts_native_ruff_positions_and_preserves_scope():
    native = issue(location={"path": "seif/app.py", "positions": {"begin": {"line": 4}}})
    tests = issue("tests/test_app.py", fingerprint="another")
    validate_report([native, tests], {"seif/app.py", "tests/test_app.py"})
    summary = summarize([native, tests])
    assert summary["total"] == 2
    assert summary["by_scope"] == {"seif": 1, "tests": 1}
    assert summary["by_rule"] == {"C901": 2}


def test_rejects_duplicate_fingerprints_instead_of_hiding_findings():
    with pytest.raises(ValueError, match="Duplicate fingerprint"):
        validate_report([issue(), issue(check_name="F821")], {"seif/app.py"})


def test_rejects_symlink_source_member(tmp_path):
    archive = tmp_path / "link.zip"
    link = ZipInfo("seif/app.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as source:
        source.writestr(link, "../../outside.py")
    with pytest.raises(ValueError, match="link or duplicate"):
        extract_python(archive, tmp_path / "analysis")


def test_rejects_duplicate_source_members(tmp_path):
    archive = tmp_path / "duplicate.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("seif/app.py", "x = 1")
        with pytest.warns(UserWarning, match="Duplicate name"):
            source.writestr("seif/app.py", "x = 2")
    with pytest.raises(ValueError, match="link or duplicate"):
        extract_python(archive, tmp_path / "analysis")


def test_ruff_does_not_silently_exclude_extracted_source(tmp_path):
    ruff = shutil.which("ruff") or str(Path(sys.executable).with_name("ruff"))
    if not Path(ruff).is_file():
        pytest.skip("Ruff is a development dependency")
    source = tmp_path / "seif" / "venv"
    source.mkdir(parents=True)
    (source / "hidden.py").write_text("missing_name()\n", encoding="utf-8")
    report, _ = run_profile(ruff, tmp_path, "F821")
    validate_report(report, {"seif/venv/hidden.py"})
    assert len(report) == 1
    assert report[0]["check_name"] == "F821"
