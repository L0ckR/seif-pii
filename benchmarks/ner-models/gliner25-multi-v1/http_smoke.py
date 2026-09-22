"""Isolated native GLiNER/API HTTP smoke; uses only synthetic local input.

Run after model benchmarks finish. This starts one worker per service on
reserved loopback sockets, bypasses .env loading, and terminates only children
created here. It is a correctness smoke, not an RPS/capacity measurement.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import secrets
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

MODEL_NAME = "fastino/gliner2.5-multi-v1"
NAME = "Иван Иванов"
EMAIL = "smoke@example.invalid"


class SmokeFailure(RuntimeError):
    """A fixed check name, never a response body, input, or credential."""


def require(condition, check):
    if not condition:
        raise SmokeFailure(check)


def request(url, *, body=None, headers=None, timeout=90):
    parsed = urlsplit(url)
    require(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "loopback_http_only")
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    outgoing = {"Content-Type": "application/json", **(headers or {})}
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request("POST" if data is not None else "GET", parsed.path, body=data, headers=outgoing)
        response = connection.getresponse()
        raw = response.read(256 * 1024 + 1)
        require(len(raw) <= 256 * 1024, "bounded_http_response")
        return response.status, json.loads(raw)
    finally:
        connection.close()


def clean_environment():
    ignored = {"PROMETHEUS_MULTIPROC_DIR", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
               "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("SEIF_", "UVICORN_")) and name not in ignored}
    env.update(PYTHON_DOTENV_DISABLED="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def start_service(service, root, env, log, children):
    python, module, label = service
    require(python.is_file(), "local_interpreter_exists")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        argv = [str(python), "-m", "uvicorn", module, "--factory", "--fd", str(listener.fileno()),
                "--workers", "1", "--no-access-log", "--log-level", "warning"]
        # Fixed module arguments and a validated local interpreter; no shell.
        process = subprocess.Popen(  # noqa: S603
            argv, cwd=root, env={**env, "PYTHONPATH": str(root)}, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, pass_fds=(listener.fileno(),), start_new_session=True,
        )
        children.append((label, process))
    return process, f"http://127.0.0.1:{port}"


def await_ready(process, base_url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, "child_survives_startup")
        try:
            status, body = request(base_url + "/health", timeout=2)
            if status == 200:
                return body
        except (OSError, TimeoutError):
            pass
        time.sleep(0.2)
    raise SmokeFailure("startup_readiness_deadline")


def terminate_children(children):
    result = []
    for label, process in reversed(children):
        if process.poll() is None:
            process.terminate()
        forced = False
        try:
            code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            forced = True
            process.kill()
            code = process.wait(timeout=10)
        result.append({"service": label, "exited": process.poll() is not None,
                       "return_code": code, "forced_kill": forced})
    return result


def verify_person_offsets(ner_url, token, text):
    status, body = request(ner_url + "/analyze", body={"text": text},
                           headers={"Authorization": "Bearer " + token})
    require(status == 200, "ner_analyze_200")
    entities = body.get("entities")
    require(isinstance(entities, list), "ner_entities_list")
    found = False
    for span in entities:
        start, end, score = span.get("start"), span.get("end"), span.get("score")
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), "unicode_offsets_valid")
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, "model_score_valid")
        if span.get("entity_type") == "PERSON" and text[start:end] == NAME:
            found = True
    require(found, "person_exact_original_text_offsets")
    return {"entity_count": len(entities), "exact_person_found": found,
            "name_start": text.index(NAME), "characters": len(text)}


def verify_roundtrip(api_url, headers, text):
    payload_id = "gliner-native-smoke-" + secrets.token_hex(12)
    started = time.perf_counter()
    status, masked = request(api_url + "/process", body={"payload": text, "payload_id": payload_id}, headers=headers)
    require(status == 200 and set(masked) == {"result"}, "process_mask_contract")
    require(isinstance(masked["result"], str) and masked["result"] != text, "mask_changes_input")
    require(NAME not in masked["result"] and EMAIL not in masked["result"], "person_and_email_masked")
    mask_ms = (time.perf_counter() - started) * 1000
    status, details = request(api_url + "/v1/mask", body={"payload": text, "payload_id": payload_id}, headers=headers)
    require(status == 200 and {"PERSON", "EMAIL"}.issubset(details.get("types", [])), "mask_has_person_and_email_types")
    status, restored = request(api_url + "/process", body={"payload": masked["result"], "payload_id": payload_id},
                               headers=headers)
    require(status == 200 and restored == {"result": text}, "exact_process_roundtrip")
    return {"mask_ms": round(mask_ms, 3), "person_and_email_masked": True, "roundtrip_exact": True,
            "characters": len(text)}


def verify_http_boundaries(ner_url, api_url, token, headers):
    require(request(ner_url + "/analyze", body={"text": "synthetic"})[0] == 401, "ner_auth_401")
    require(request(ner_url + "/analyze", body={"text": 42},
                    headers={"Authorization": "Bearer " + token})[0] == 422, "ner_validation_422")
    require(request(api_url + "/process", body={"payload": "synthetic", "payload_id": "auth-check"},
                    headers={"X-System-ID": "smoke"})[0] == 401, "api_auth_401")
    require(request(api_url + "/process", body={"payload": 42, "payload_id": "validation-check"},
                    headers=headers)[0] == 422, "api_validation_422")
    return {"ner_401": True, "ner_422": True, "api_401": True, "api_422": True}


def run_services(args, temporary, children, report):
    root = args.repo.resolve()
    token, api_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    base_env = clean_environment()
    config = temporary / "smoke-policies.json"
    config.write_text(json.dumps({"systems": {"smoke": {
        "enabled": True, "api_key_env": "SEIF_SMOKE_API_KEY", "mode": "mask", "allow_unmask": True,
    }}}), encoding="utf-8")
    ner_env = {**base_env, "SEIF_NER_TOKEN": token, "SEIF_NER_BACKEND": "gliner", "SEIF_NER_DEMO": "0",
               "SEIF_GLINER_MODEL_PATH": str(args.model_path.resolve()), "SEIF_GLINER_DEVICE": args.device,
               "SEIF_GLINER_SCHEMA": args.schema, "SEIF_GLINER_THRESHOLD": str(args.threshold),
               "SEIF_NER_MAX_MODEL_JOBS": "1"}
    with (temporary / "ner.log").open("wb") as ner_log, (temporary / "api.log").open("wb") as api_log:
        ner_process, ner_url = start_service((args.ner_python, "scripts.ner_service:create_app", "ner"),
                                             root, ner_env, ner_log, children)
        ner_health = await_ready(ner_process, ner_url, args.startup_timeout)
        require(ner_health.get("model") == MODEL_NAME, "actual_gliner_model_health")
        api_env = {**base_env, "SEIF_DEMO": "0", "SEIF_CONFIG": str(config), "SEIF_SMOKE_API_KEY": api_key,
                   "SEIF_MASTER_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
                   "SEIF_NER_URL": ner_url, "SEIF_NER_TOKEN": token, "SEIF_NER_TIMEOUT_SECONDS": "90",
                   "SEIF_CPU_WORKERS": "1"}
        api_process, api_url = start_service((args.api_python, "seif.app:create_app", "api"),
                                             root, api_env, api_log, children)
        api_health = await_ready(api_process, api_url, args.startup_timeout)
        require(api_health.get("mode") == "restricted" and api_health.get("storage") == "memory"
                and api_health.get("detector_profile") == "hybrid", "isolated_restricted_hybrid_api")
        report["health"] = {"ner": ner_health, "api": api_health}
        headers = {"X-System-ID": "smoke", "X-API-Key": api_key}
        report["boundaries"] = verify_http_boundaries(ner_url, api_url, token, headers)
        short = f"😀 Клиент {NAME}. Электронная почта: {EMAIL}."
        long = "Документ содержит описание работы системы. " * 90 + short
        require(len(long.split()) > 384 and long.index(NAME) > len(long) * 0.8, "long_text_tail_fixture")
        for label, text in (("short", short), ("long", long)):
            report[label] = {"ner": verify_person_offsets(ner_url, token, text),
                             "api": verify_roundtrip(api_url, headers, text), "words": len(text.split())}


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root)
    parser.add_argument("--ner-python", type=Path, default=root / ".venv/bin/python")
    parser.add_argument("--api-python", type=Path, default=root.parent / "seif-pii/.venv/bin/python")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--schema", choices=("person-location", "presidio-labels", "described-names"),
                        default="described-names")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.model_path.is_dir(), "installed_local_checkpoint")
    require(math.isfinite(args.threshold) and 0 < args.threshold < 1, "valid_threshold")
    require(math.isfinite(args.startup_timeout) and args.startup_timeout > 0, "valid_startup_timeout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Native loopback correctness smoke; synthetic data, memory vault, one worker per service; no RPS claim.",
              "configuration": {"backend": "gliner", "schema": args.schema, "threshold": args.threshold,
                                "device": args.device, "dotenv_disabled": True, "inherited_seif_settings": False},
              "source_sha256": {}}
    for name in ("seif/gliner_ner.py", "scripts/ner_service.py", "seif/app.py"):
        with (args.repo / name).open("rb") as source:
            report["source_sha256"][name] = hashlib.file_digest(source, "sha256").hexdigest()
    children = []
    # Reserve the report path exclusively before starting any child process.
    with (
        args.output.open("x", encoding="utf-8") as output,
        tempfile.TemporaryDirectory(prefix="seif-gliner-http-smoke-") as temporary,
    ):
        try:
            run_services(args, Path(temporary), children, report)
            report["status"] = "PASS"
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            report["cleanup"] = terminate_children(children)
            report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
