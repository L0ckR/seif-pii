"""Analyze a source ZIP locally and emit native Ruff GitLab Code Quality reports.

This is a reproducible linter comparison, not a reconstruction of the hackathon
scoring algorithm. No submitted code is executed or sent to an external service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

PROFILES = {
    "project": "E4,E7,E9,F,I,B",
    "maintainability": "C901,C4,SIM,PERF,PLR0911,PLR0912,PLR0913,PLR0915",
    "security": "S",
}
SOURCE_DIRECTORIES = {"seif", "scripts", "tests", "deploy"}
SEVERITIES = {"info", "minor", "major", "critical", "blocker"}
MAX_SOURCE_BYTES = 32 * 1024 * 1024


def extract_python(archive: Path, destination: Path) -> dict:
    """Copy Python files from known source directories; reject ambiguous paths."""
    manifest = {}
    total = 0
    with ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or str(path) != member.filename.rstrip("/")
            ):
                raise ValueError("Archive contains a non-canonical path")
            if member.is_dir() or path.suffix != ".py" or path.parts[0] not in SOURCE_DIRECTORIES:
                continue
            if stat.S_ISLNK(member.external_attr >> 16) or str(path) in manifest:
                raise ValueError("Archive contains a link or duplicate source path")
            total += member.file_size
            if total > MAX_SOURCE_BYTES:
                raise ValueError("Python source archive exceeds the analysis size limit")
            content = source.read(member)
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest[str(path)] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "lines": len(content.splitlines()),
            }
    if not manifest:
        raise ValueError("Archive contains no Python source files")
    return manifest


def validate_report(report: list, source_paths: set[str]) -> None:
    """Validate the subset required by GitLab, plus source-relative locations."""
    if not isinstance(report, list):
        raise ValueError("GitLab report must be a JSON array")
    fingerprints = set()
    for issue in report:
        if not isinstance(issue, dict):
            raise ValueError("GitLab findings must be objects")
        for field in ("description", "check_name", "fingerprint"):
            if not isinstance(issue.get(field), str) or not issue[field]:
                raise ValueError(f"Invalid GitLab finding field: {field}")
        if issue["fingerprint"] in fingerprints:
            raise ValueError("Duplicate fingerprint would hide a finding in GitLab")
        fingerprints.add(issue["fingerprint"])
        _validate_location(issue, source_paths)


def _validate_location(issue, source_paths):
    location = issue.get("location", {})
    try:
        line = location.get("lines", {}).get("begin")
        if line is None:
            line = location.get("positions", {}).get("begin", {}).get("line")
    except AttributeError:
        raise ValueError("Invalid GitLab location object") from None
    severity = issue.get("severity")
    if not isinstance(severity, str) or severity not in SEVERITIES or type(line) is not int or line < 1:
        raise ValueError("Invalid GitLab severity or line number")
    path = location.get("path")
    if not isinstance(path, str) or path not in source_paths:
        raise ValueError("GitLab finding does not reference an analyzed source file")


def summarize(report: list) -> dict:
    """Separate runtime, tooling, deployment and tests without dropping findings."""
    return {
        "total": len(report),
        "by_scope": dict(sorted(Counter(issue["location"]["path"].split("/", 1)[0] for issue in report).items())),
        "by_rule": dict(Counter(issue["check_name"] for issue in report).most_common()),
        "by_severity": dict(sorted(Counter(issue["severity"] for issue in report).items())),
        "top_files": dict(Counter(issue["location"]["path"] for issue in report).most_common(12)),
    }


def run_profile(ruff: str, source: Path, rules: str) -> tuple[list, list[str]]:
    arguments = [
        "check",
        "--isolated",
        "--no-cache",
        "--target-version",
        "py312",
        "--exclude",
        "",
        "--no-respect-gitignore",
        "--line-length",
        "120",
        "--select",
        rules,
        "--output-format",
        "gitlab",
        ".",
    ]
    result = subprocess.run([ruff, *arguments], cwd=source, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError(f"Ruff analysis failed (exit {result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout), arguments


def analyze_archive(archive: Path, output: Path, ruff: str) -> dict:
    """Keep each run immutable: an existing output directory is an error."""
    version = subprocess.run([ruff, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    output.mkdir(parents=True, exist_ok=False)
    with archive.open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "schema_version": 1,
        "archive": archive.name,
        "archive_sha256": archive_sha256,
        "analyzer": version,
        "profiles": {},
        "scope": "All Python files under seif/, scripts/, tests/, deploy/ in the source ZIP",
        "configuration": "Isolated fixed profiles; Ruff default thresholds; inline noqa is honored",
        "severity": "Native Ruff GitLab severity, unchanged; not Sonar or hackathon severity",
    }
    with tempfile.TemporaryDirectory(prefix="seif-code-quality-") as temporary:
        source = Path(temporary)
        metadata["source_files"] = extract_python(archive, source)
        for profile, rules in PROFILES.items():
            report, arguments = run_profile(ruff, source, rules)
            validate_report(report, set(metadata["source_files"]))
            filename = f"{profile}.gitlab.json"
            (output / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            metadata["profiles"][profile] = {
                "report": filename,
                "ruff_arguments": arguments,
                **summarize(report),
            }
    (output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ruff", default="ruff", help="Ruff executable (use the pinned project version)")
    args = parser.parse_args()
    executable = shutil.which(args.ruff)
    if executable is None:
        parser.error("Ruff executable not found")
    result = analyze_archive(args.archive.resolve(), args.output.resolve(), str(Path(executable).resolve()))
    print(
        json.dumps(
            {
                "archive": result["archive"],
                "analyzer": result["analyzer"],
                "profiles": {
                    name: {key: value for key, value in profile.items() if key in {"total", "by_scope", "by_rule"}}
                    for name, profile in result["profiles"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
