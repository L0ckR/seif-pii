"""The submission must start independently and fail closed on missing files."""
import ast
import hashlib
import os
import runpy
import shutil
import subprocess
import sys
import tomllib
from zipfile import ZipFile

import pytest
import yaml

from scripts.package import REQUIRED_FILES, ROOT, TEMPLATES, build_archive


def put(root, name, content="example"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(ROOT / "seif", root / "seif", ignore=shutil.ignore_patterns("__pycache__"))
    for name in (*REQUIRED_FILES, *TEMPLATES.values()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return root


def test_service_only_with_complete_runtime_and_checksums(project):
    excluded = (".env", "seif/.env", "seif/dump.rdb", "seif/__pycache__/cache.py",
                "tests/test_app.py", "scripts/benchmark.py", "scripts/package.py",
                "local-data/requests.jsonl", "output/credentials.json", "web/app.js",
                "deploy/k8s/create-secrets.py", "docs/check.json", ".github/workflows/ci.yml")
    for name in excluded:
        put(project, name, "PRIVATE DEVELOPMENT FILE")
    archive = build_archive(project, project / "output/source.zip")
    with ZipFile(archive) as packaged:
        names = set(packaged.namelist())
        expected = {path.relative_to(ROOT).as_posix() for path in (ROOT / "seif").glob("*.py")}
        expected.update(REQUIRED_FILES)
        expected.update({"README.md", ".env.example", ".dockerignore", "SHA256SUMS"})
        assert names == expected
        assert names.isdisjoint(excluded)
        assert all(b"PRIVATE DEVELOPMENT FILE" not in packaged.read(name) for name in names)
        hashes = dict(line.split("  ")[::-1] for line in packaged.read("SHA256SUMS").decode().splitlines())
        assert set(hashes) == names - {"SHA256SUMS"}
        for name, digest in hashes.items():
            assert hashlib.sha256(packaged.read(name)).hexdigest() == digest
        metadata = tomllib.loads(packaged.read("pyproject.toml").decode())
        original = tomllib.loads((ROOT / "pyproject.toml").read_text())
        assert metadata["project"]["dependencies"] == original["project"]["dependencies"]
        assert "optional-dependencies" not in metadata["project"]
        assert "COPY web" not in packaged.read("Dockerfile").decode()
        assert not any(line and not line.startswith("#") and line.split("=")[1]
                       for line in packaged.read(".env.example").decode().splitlines()
                       if line.startswith(("SEIF_MASTER_KEY=", "SEIF_DEMO_API_KEY=", "SEIF_NER_TOKEN=")))


def test_runtime_unchanged_except_optional_static_ui(project):
    archive = build_archive(project, project / "output/source.zip")
    with ZipFile(archive) as packaged:
        for name in packaged.namelist():
            if name.endswith(".py") and name != "seif/app.py":
                assert packaged.read(name) == (project / name).read_bytes()
        original = ast.parse((project / "seif/app.py").read_text())
        original.body = [node for node in original.body if not (
            isinstance(node, ast.ImportFrom) and node.module in {"pathlib", "fastapi.staticfiles"})]
        factory = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "create_app")
        assert isinstance(factory.body[-3], ast.Assign) and factory.body[-3].targets[0].id == "web"
        assert isinstance(factory.body[-2], ast.Expr) and factory.body[-2].value.func.attr == "mount"
        del factory.body[-3:-1]
        assert ast.dump(original) == ast.dump(ast.parse(packaged.read("seif/app.py")))


def test_packaged_compose_mounts_and_build_inputs_are_valid(project):
    archive = build_archive(project, project / "output/source.zip")
    with ZipFile(archive) as packaged:
        for name in ("compose.yaml", "compose.ner.yaml"):
            services = yaml.safe_load(packaged.read(name))["services"]
            for service in services.values():
                assert all(mount.startswith("/") for mount in service.get("tmpfs", []))
                if "build" in service:
                    build = service["build"]
                    context = build["context"] if isinstance(build, dict) else build
                    dockerfile = build.get("dockerfile", "Dockerfile") if isinstance(build, dict) else "Dockerfile"
                    assert context == "." and dockerfile in packaged.namelist()
        ner = yaml.safe_load(packaged.read("compose.ner.yaml"))["services"]["ner"]
        assert len(ner["tmpfs"]) == 1
        assert set(ner["tmpfs"][0].split(":", 1)[1].split(",")) == {"size=64m", "mode=1777"}


@pytest.mark.parametrize("name", [*REQUIRED_FILES, *TEMPLATES.values(), "seif/detector.py", "seif/async_callbacks.py"])
def test_missing_runtime_dependency_preserves_previous_zip(project, name):
    target = put(project, "output/source.zip", "previous archive")
    (project / name).unlink()
    with pytest.raises(ValueError, match="Missing"):
        build_archive(project, target)
    assert target.read_text() == "previous archive"
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("name", ["seif/app.py", "seif/detector.py", "config/policies.yaml", "deploy/service/README.md"])
def test_linked_runtime_file_is_rejected(project, name, tmp_path):
    path = project / name
    outside = tmp_path / "outside.py"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        build_archive(project, project / "output/source.zip")


def test_linked_required_directory_is_rejected(project, tmp_path):
    outside = tmp_path / "outside"
    (project / "config").rename(outside)
    (project / "config").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        build_archive(project, project / "output/source.zip")


@pytest.mark.parametrize(("file", "extra", "message"), [
    ("seif/detector.py", "\nfrom .missing import detect\n", "Missing local import"),
    ("seif/detector.py", "\nimport scripts.evaluate\n", "Missing local import"),
    ("seif/detector.py", "\nfrom seif import missing\n", "Missing local import"),
    ("seif/detector.py", "\nfrom . import missing\n", "Missing local import"),
    ("Dockerfile.ner", "\nCOPY models ./models\n", "Missing Docker COPY"),
    ("Dockerfile.ner", "\n  copy models ./models\n", "Missing Docker COPY"),
    ("Dockerfile.ner", '\nCOPY ["models", "./models"]\n', "recipe needs review"),
    ("seif/app.py", "\nRESOURCE = Path('some-resource')\n", "removed UI import"),
])
def test_new_unpackaged_dependency_rejected(project, file, extra, message):
    path = project / file
    path.write_text(path.read_text() + extra)
    with pytest.raises(ValueError, match=message):
        build_archive(project, project / "output/source.zip")


def test_changed_ui_mount_requires_packaging_review(project):
    path = project / "seif/app.py"
    path.write_text(path.read_text().replace('name="web")', 'name="demo")'))
    with pytest.raises(ValueError, match="recipe needs review"):
        build_archive(project, project / "output/source.zip")


def test_package_imports_support_existing_submodules_and_exports(project):
    path = project / "seif/detector.py"
    path.write_text(path.read_text() + "\nfrom seif import person_fields, VERSION\nfrom scripts import serve\n")
    package = project / "seif/__init__.py"
    package.write_text(package.read_text() + '\nVERSION = "example"\n')
    assert build_archive(project, project / "output/source.zip").is_file()


def test_archive_is_reproducible_across_mtime_changes(project):
    first = build_archive(project, project / "output/first.zip")
    for path in project.rglob("*"):
        if path.is_file():
            os.utime(path, (1_700_000_000, 1_700_000_000))
    second = build_archive(project, project / "output/second.zip")
    assert first.read_bytes() == second.read_bytes()


def test_failed_build_preserves_previous_archive_and_removes_temporary(project, monkeypatch):
    target = put(project, "output/source.zip", "previous archive")

    def fail(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="simulated"):
        build_archive(project, target)
    assert target.read_text() == "previous archive"
    assert list(target.parent.iterdir()) == [target]


def test_output_cannot_replace_source_or_follow_symlink(project):
    source = project / "seif/app.py"
    original = source.read_bytes()
    with pytest.raises(ValueError):
        build_archive(project, project / "seif/../seif/app.py")
    link = project / "submission.zip"
    link.symlink_to(source)
    with pytest.raises(ValueError):
        build_archive(project, link)
    assert source.read_bytes() == original


def test_import_does_not_create_or_replace_archive(tmp_path):
    copied = put(tmp_path, "scripts/package.py", (ROOT / "scripts/package.py").read_text())
    runpy.run_path(str(copied), run_name="package_import_test")
    assert not (tmp_path / "output").exists()


def test_extracted_service_starts_without_checkout_or_environment(project, tmp_path):
    archive = build_archive(project, project / "output/source.zip")
    isolated = tmp_path / "standalone"
    with ZipFile(archive) as packaged:
        packaged.extractall(isolated)
    shutil.rmtree(project)
    program = '''
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import seif.app
from scripts.ner_service import create_app as ner_app, NerSettings
from fastapi.testclient import TestClient
assert Path(seif.app.__file__).resolve().parent == Path.cwd() / 'seif'
os.environ['SEIF_DEMO'] = '1'
os.environ['SEIF_DEMO_API_KEY'] = ''
with TestClient(seif.app.create_app()) as client:
    assert client.get('/health').status_code == 200
    assert client.get('/').status_code == 404
    assert client.get('/v1/types').status_code == 200
    text = 'Email: package@example.com'
    response = client.post('/process', json={'payload': text, 'payload_id': 'isolated'})
    assert response.status_code == 200 and response.json()['result'] != text
    restored = client.post('/process', json={'payload': response.json()['result'], 'payload_id': 'isolated'})
    assert restored.status_code == 200 and restored.json()['result'] == text
with TestClient(ner_app(NerSettings(demo=True), analyzer_factory=lambda: object())) as client:
    assert client.get('/health').status_code == 200
    assert client.post('/analyze', json={'text': ''}).json() == {'entities': []}
    assert client.post('/analyze', json={'text': 1}).status_code == 422
'''
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SEIF_", "PYTHON")) and key != "PROMETHEUS_MULTIPROC_DIR"}
    result = subprocess.run([sys.executable, "-I", "-c", program], cwd=isolated, env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
