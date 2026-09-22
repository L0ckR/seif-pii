"""Build a reproducible source-only ZIP: no keys, data, dependencies or binaries."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRS = {"seif", "tests", "scripts", "config", "deploy", "web", ".github"}
ROOT_FILES = {"README.md", "pyproject.toml", "requirements.lock", "Dockerfile", ".dockerignore", ".gitignore", ".env.example", ".python-version", "Makefile", "process_api.yaml", "compose.yaml"}
EXCLUDED = {"__pycache__", ".venv", "node_modules", ".pytest_cache", "output"}
target = ROOT / "output" / "seif-pii-source.zip"
target.parent.mkdir(exist_ok=True)
with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(x in EXCLUDED for x in rel.parts):
            continue
        # Documentation is textual implementation evidence, not the supplied PDF.
        include = (len(rel.parts) == 1 and rel.name in ROOT_FILES) or rel.parts[0] in ALLOWED_DIRS
        include |= rel.parts[0] == "docs" and path.suffix in {".md", ".json"}
        if not include or path.suffix in {".pyc", ".log", ".zip"} or path.name.startswith(".env") and path.name != ".env.example":
            continue
        info = ZipInfo(rel.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
        archive.writestr(info, path.read_bytes())
print(target)
print(f"{target.stat().st_size:,} bytes; source only")
