"""Build a reproducible source ZIP without following links or traversing data."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {
    "seif": {".py"}, "tests": {".py"}, "scripts": {".py", ".sh"},
    "config": {".yaml", ".yml"},
    "deploy": {".yaml", ".yml", ".sh", ".py", ".txt"},
    "web": {".html", ".css", ".js", ".svg"}, ".github": {".yaml", ".yml"},
    "docs": {".md", ".json"},
}
ROOT_FILES = {
    "README.md", "pyproject.toml", "requirements.lock", "Dockerfile", "Dockerfile.ner",
    ".dockerignore", ".gitignore", ".env.example", ".python-version", "Makefile",
    "process_api.yaml", "compose.yaml", "compose.ner.yaml",
}
EXCLUDED = {
    "__pycache__", ".venv", "venv", "env", "node_modules", ".pytest_cache", ".ruff_cache",
    "output", "local-data", ".git", ".idea", "build", "dist", "target", "coverage",
}


def source_files(root: Path):
    """Allow source formats in known directories; never follow a symlink."""
    for name in sorted(ROOT_FILES):
        path = root / name
        if not path.is_symlink() and path.is_file():
            yield path
    for name, suffixes in sorted(SOURCE_SUFFIXES.items()):
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            continue
        for current, directories, files in os.walk(directory, followlinks=False):
            directories[:] = sorted(child for child in directories
                                    if child not in EXCLUDED and not (Path(current) / child).is_symlink())
            for filename in sorted(files):
                path = Path(current) / filename
                if (path.is_symlink() or not path.is_file() or path.suffix not in suffixes
                        or filename.startswith(".env")):
                    continue
                yield path


def build_archive(root: Path, target: Path) -> Path:
    """Replace an output archive only after a complete, successful build."""
    root = root.resolve()
    target = target.parent.resolve() / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    # The output is excluded even when a caller chooses an allowed source name.
    if target == root or target.is_symlink() or target in source_files(root):
        raise ValueError("Archive output must not overwrite a source file or symbolic link")
    descriptor, temporary = tempfile.mkstemp(prefix=".seif-source-", suffix=".zip", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream, ZipFile(stream, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(source_files(root)):
                rel = path.relative_to(root)
                info = ZipInfo(rel.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
                # Reject a leaf swapped for a link after enumeration on POSIX.
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as source:
                    archive.writestr(info, source.read(), compress_type=ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main() -> None:
    target = build_archive(ROOT, ROOT / "output" / "seif-pii-source.zip")
    print(target)
    print(f"{target.stat().st_size:,} bytes; source only")


if __name__ == "__main__":
    main()
