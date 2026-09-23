"""Build a reproducible, complete service ZIP; development tools stay in Git."""

from __future__ import annotations

import ast
import hashlib
import os
import shlex
import tempfile
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "seif/__init__.py",
    "seif/app.py",
    "scripts/serve.py",
    "scripts/ner_service.py",
    "config/policies.yaml",
    "requirements.lock",
    "deploy/ner/requirements-ner.txt",
    "Dockerfile",
    "Dockerfile.ner",
    "compose.yaml",
    "compose.ner.yaml",
    "pyproject.toml",
    ".python-version",
    "process_api.yaml",
    "third_party/pii-guard/LICENSE",
    "third_party/pii-guard/NOTICE",
    "third_party/pii-guard/ADAPTATION.md",
    "deploy/ner/requirements-rubert-tensorrt.txt",
)
TEMPLATES = {name: f"deploy/service/{name}" for name in ("README.md", ".env.example", ".dockerignore")}
UI_LINES = (
    "from pathlib import Path\n",
    "from fastapi.staticfiles import StaticFiles\n",
    '    web = Path(__file__).resolve().parent.parent / "web"\n',
    '    app.mount("/", StaticFiles(directory=web, html=True), name="web")\n',
)


def _read_source(root: Path, name: str) -> bytes:
    path = root / name
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
        raise ValueError(f"Service source must not be a symbolic link: {name}")
    if not path.is_file():
        raise ValueError(f"Missing required service file: {name}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as source:
        return source.read()


def source_files(root: Path):
    """Select runtime modules and explicit deployment files, never other tools."""
    names = set(REQUIRED_FILES) | set(TEMPLATES.values())
    for current, directories, files in os.walk(root / "seif", followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if not name.startswith(".") and name != "__pycache__" and not (Path(current) / name).is_symlink()
        )
        names.update(
            (Path(current) / name).relative_to(root).as_posix()
            for name in files
            if name.endswith(".py") and not name.startswith(".")
        )
    return [root / name for name in sorted(names)]


def _remove_exact(text: str, fragment: str, name: str) -> str:
    if text.count(fragment) != 1:
        raise ValueError(f"Service packaging recipe needs review: {name}")
    return text.replace(fragment, "")


def _service_app(data: bytes) -> bytes:
    text = data.decode("utf-8")
    for line in UI_LINES:
        text = _remove_exact(text, line, "seif/app.py static UI")
    # New uses must retain their imports or receive an updated packaging recipe.
    if any(isinstance(node, ast.Name) and node.id in {"Path", "StaticFiles"} for node in ast.walk(ast.parse(text))):
        raise ValueError("Service app still requires a removed UI import")
    return text.encode("utf-8")


def _runtime_metadata(data: bytes) -> bytes:
    text = data.decode("utf-8")
    marker = "[project.optional-dependencies]"
    if text.count(marker) != 1:
        raise ValueError("Service packaging recipe needs review: pyproject.toml")
    text = text.split(marker)[0] + '[tool.setuptools.packages.find]\ninclude = ["seif*"]\n'
    metadata = tomllib.loads(text)
    if not metadata["project"]["dependencies"] or not metadata["build-system"]["requires"]:
        raise ValueError("Runtime or build dependencies are missing")
    return text.encode("utf-8")


def _from_module(package: list[str], node: ast.ImportFrom) -> str:
    base = package[: len(package) - node.level + 1] if node.level else []
    return ".".join([*base, node.module] if node.module else base)


def _exported_names(source: bytes) -> set[str]:
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _package_imports(module: str, node: ast.ImportFrom, members: dict[str, bytes]):
    path = module.replace(".", "/")
    if module.split(".")[0] not in {"seif", "scripts"} or f"{path}.py" in members:
        return
    exports = _exported_names(members.get(f"{path}/__init__.py", b""))
    for alias in node.names:
        if alias.name == "*":
            raise ValueError("Service packaging requires explicit local package imports")
        if alias.name not in exports:
            yield f"{module}.{alias.name}"


def _local_imports(name: str, tree, members: dict[str, bytes]):
    package = name.removesuffix(".py").split("/")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _from_module(package, node)
            yield module
            yield from _package_imports(module, node, members)


def _validate_imports(members: dict[str, bytes]) -> None:
    for name, data in members.items():
        if not name.endswith(".py"):
            continue
        tree = ast.parse(data, filename=name)
        for module in _local_imports(name, tree, members):
            if module.split(".")[0] not in {"seif", "scripts"}:
                continue
            path = module.replace(".", "/")
            if f"{path}.py" not in members and not any(name.startswith(path + "/") for name in members):
                raise ValueError(f"Missing local import in service ZIP: {name} -> {module}")


def _validate_docker_files(members: dict[str, bytes]) -> None:
    for name in ("Dockerfile", "Dockerfile.ner"):
        for line in members[name].decode("utf-8").splitlines():
            instruction = line.split(maxsplit=1)
            if not instruction or instruction[0].upper() != "COPY":
                continue
            parts = shlex.split(line, comments=True)
            if len(parts) < 3 or any(part.startswith(("--", "[")) or "\\" in part for part in parts[1:]):
                raise ValueError(f"Service packaging recipe needs review: {name} COPY syntax")
            for source in parts[1:-1]:
                if source not in members and not any(path.startswith(source.rstrip("/") + "/") for path in members):
                    raise ValueError(f"Missing Docker COPY source in service ZIP: {name} -> {source}")


def service_members(root: Path) -> dict[str, bytes]:
    members = {
        path.relative_to(root).as_posix(): _read_source(root, path.relative_to(root).as_posix())
        for path in source_files(root)
    }
    for target, source in TEMPLATES.items():
        members[target] = members.pop(source)
    members["seif/app.py"] = _service_app(members["seif/app.py"])
    members["Dockerfile"] = _remove_exact(
        members["Dockerfile"].decode("utf-8"), "COPY web ./web\n", "Dockerfile static UI"
    ).encode("utf-8")
    members["pyproject.toml"] = _runtime_metadata(members["pyproject.toml"])
    _validate_imports(members)
    _validate_docker_files(members)
    members["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(members.items())
    ).encode("utf-8")
    return members


def build_archive(root: Path, target: Path) -> Path:
    """Validate all inputs before atomically replacing the previous archive."""
    root = root.resolve()
    target = target.parent.resolve() / target.name
    if target.suffix != ".zip" or target.is_symlink() or target in source_files(root):
        raise ValueError("Archive output must not overwrite a source file or symbolic link")
    members = service_members(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".seif-service-", suffix=".zip", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream, ZipFile(stream, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(members.items()):
                info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main() -> None:
    target = build_archive(ROOT, ROOT / "output" / "seif-pii-source.zip")
    print(target)
    print(f"{target.stat().st_size:,} bytes; API, NER and deployment files only; includes SHA256SUMS")


if __name__ == "__main__":
    main()
