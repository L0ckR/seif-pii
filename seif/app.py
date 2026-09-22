"""HTTP boundary: authenticated, retry-safe and fail-closed."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import sys
import sysconfig
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .detector import TYPES, detect, validate_extra_rule
from .transform import mask, restore_exact, restore_tokens
from .vault import Vault, VaultFull

LOG = logging.getLogger("seif.audit")


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: StrictStr
    payload_id: StrictStr = Field(min_length=1, max_length=256)


class MaskRequest(ProcessRequest):
    mode: str | None = None


def fail(status: int, code: str, message: str):
    raise HTTPException(status, detail={"code": code, "message": message},
                        headers={"Retry-After": "1"} if status == 429 else None)


def error(status, code, message, headers=None):
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class Boundary:
    """Pure ASGI middleware avoids per-request task overhead and raw access logs."""
    def __init__(self, app, settings, registry):
        self.app, self.settings, self.inflight = app, settings, 0
        self.buffered_bytes = 0
        self.count = Counter("seif_http_requests_total", "Requests by fixed route and status", ["route", "status"], registry=registry)
        self.latency = Histogram("seif_http_duration_seconds", "Full request duration", ["route"],
                                 buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10), registry=registry)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        route = path if path in {"/process", "/v1/mask", "/v1/unmask", "/health", "/metrics", "/v1/types"} else "other"
        start, request_id, status = time.perf_counter(), uuid.uuid4().hex, 500
        overloaded = self.inflight >= self.settings.max_inflight
        self.inflight += 1
        held_body_bytes = 0
        scope.setdefault("state", {})["request_id"] = request_id
        headers = dict(scope.get("headers", []))

        async def safe_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).extend([
                    (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"), (b"x-frame-options", b"DENY"),
                    (b"x-request-id", request_id.encode()),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")])
            await send(message)

        try:
            if overloaded:
                return await error(429, "busy", "Сервис занят; повторите запрос.", {"Retry-After": "1"})(scope, receive, safe_send)
            try:
                content_length = int(headers.get(b"content-length", b"0"))
            except ValueError:
                return await error(400, "invalid_request", "Некорректный размер запроса.")(scope, receive, safe_send)
            if content_length > self.settings.max_body_bytes:
                return await error(413, "too_large", "Превышен размер запроса.")(scope, receive, safe_send)
            if scope["method"] == "POST":
                if content_length > self.settings.max_inflight_body_bytes - self.buffered_bytes:
                    return await error(429, "body_budget", "Память обработки запросов занята; повторите запрос.", {"Retry-After": "1"})(scope, receive, safe_send)
                # Bound even chunked bodies before JSON parsing.
                body = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if held_body_bytes + len(chunk) > self.settings.max_body_bytes:
                        return await error(413, "too_large", "Превышен размер запроса.")(scope, receive, safe_send)
                    if self.buffered_bytes + len(chunk) > self.settings.max_inflight_body_bytes:
                        return await error(429, "body_budget", "Память обработки запросов занята; повторите запрос.", {"Retry-After": "1"})(scope, receive, safe_send)
                    self.buffered_bytes += len(chunk)
                    held_body_bytes += len(chunk)
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
                delivered = False

                async def bounded_receive():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, bounded_receive, safe_send)
            else:
                await self.app(scope, receive, safe_send)
        finally:
            self.buffered_bytes -= held_body_bytes
            self.inflight -= 1
            self.count.labels(route, str(status)).inc()
            self.latency.labels(route).observe(time.perf_counter() - start)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.ttl_seconds < 1 or settings.max_records < 1 or len(settings.encryption_key) != 32:
        raise ValueError("Invalid vault settings")
    vault, registry = Vault(settings), CollectorRegistry()
    cpu_pool = ThreadPoolExecutor(max_workers=settings.cpu_workers, thread_name_prefix="seif-detect")
    large_inflight = 0
    # Detect invalid plugin rules on startup, before any sensitive request arrives.
    for policy in settings.policies.values():
        if policy.mode not in policy.allowed_modes or set(policy.allowed_modes) - {"mask", "token", "synthetic"}:
            raise ValueError("Invalid allowed modes")
        if len(policy.extra_rules) > 32:
            raise ValueError("At most 32 custom rules are supported")
        for rule in policy.extra_rules:
            validate_extra_rule(rule)
        known_types = set(TYPES) | {rule["type"] for rule in policy.extra_rules}
        if (set(policy.types) | set(policy.required_types)) - known_types:
            raise ValueError("Unknown entity type in system policy")
    detected = Counter("seif_entities_total", "Detected entity types, never values", ["type"], registry=registry)
    chars = Counter("seif_input_characters_total", "Processed input Unicode characters", registry=registry)
    tokens = Counter("seif_input_tokens_estimated_total", "Estimated tokens = characters / 4, not a model tokenizer", registry=registry)
    stages = Histogram("seif_stage_duration_seconds", "Processing stages", ["stage"], registry=registry)
    limits = {}

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(level=os.getenv("SEIF_LOG_LEVEL", "INFO"), format="%(message)s")
        try:
            if os.getenv("SEIF_REQUIRE_FREE_THREADING", "0") == "1" and (
                not sysconfig.get_config_var("Py_GIL_DISABLED") or getattr(sys, "_is_gil_enabled", lambda: True)()
            ):
                raise RuntimeError("This deployment requires a free-threaded Python with GIL disabled")
            await vault.ping()
            yield
        finally:
            try:
                # Running detector jobs cannot be killed safely. Drain them without
                # blocking the event loop, including jobs whose HTTP task was cancelled.
                await asyncio.to_thread(cpu_pool.shutdown, wait=True, cancel_futures=True)
            finally:
                await vault.close()

    app = FastAPI(title="СЕЙФ · Personal Data Gateway", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.state.vault, app.state.settings = vault, settings
    app.add_middleware(Boundary, settings=settings, registry=registry)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default errors include the input. Never echo them.
        return error(422, "invalid_request", "Ожидаются строковые payload и непустой payload_id (до 256 символов).")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": "Запрос не выполнен."}
        return error(exc.status_code, detail["code"], detail["message"], exc.headers)

    async def authorize(request: Request):
        tenant = request.headers.get("X-System-ID", "demo" if settings.demo else "")
        policy = settings.policies.get(tenant)
        if policy is None or not policy.enabled:
            fail(403, "access_denied", "Система не допущена к сервису.")
        supplied = request.headers.get("X-API-Key", "")
        anonymous_demo = settings.demo and tenant == "demo" and not policy.api_key
        if not anonymous_demo and (not policy.api_key or not hmac.compare_digest(supplied.encode(), policy.api_key.encode())):
            fail(401, "unauthorized", "Требуется ключ системы.")
        # Fixed bounded keyspace: rate state exists only for configured tenants.
        if vault.redis:
            limiter_key = "seif:rate:" + vault.digest(tenant)
            allowed = await vault.redis.eval("local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],1) end; return n", 1, limiter_key)
            if allowed > policy.rps:
                fail(429, "rate_limit", "Превышена частота запросов системы.")
        else:
            now = time.monotonic()
            balance, last = limits.get(tenant, (float(policy.rps), now))
            balance = min(float(policy.rps), balance + (now - last) * policy.rps)
            if balance < 1:
                limits[tenant] = (balance, now)
                fail(429, "rate_limit", "Превышена частота запросов системы.")
            limits[tenant] = (balance - 1, now)
        return tenant, policy

    async def execute(body: ProcessRequest, request: Request, operation: str):
        nonlocal large_inflight
        start = time.perf_counter()
        stage_times = {}

        @contextmanager
        def stage(name):
            stage_start = time.perf_counter()
            try:
                yield
            finally:
                seconds = time.perf_counter() - stage_start
                stages.labels(name).observe(seconds)
                stage_times[name] = round(seconds * 1000, 3)

        try:
            with stage("authorize"):
                tenant, policy = await authorize(request)
            if len(body.payload) > settings.max_payload_chars:
                fail(413, "too_large", "Превышен лимит текста.")
            # Reject lone UTF-16 surrogates before UTF-8 encoding, with no input echo.
            try:
                body.payload.encode("utf-8")
                body.payload_id.encode("utf-8")
            except UnicodeEncodeError:
                fail(422, "invalid_unicode", "Некорректный Unicode.")
            mode = getattr(body, "mode", None) or policy.mode
            if mode not in policy.allowed_modes:
                fail(422, "invalid_mode", "Этот режим не разрешён политикой системы.")
            with stage("fingerprint"):
                key, fingerprint = vault.key(tenant, body.payload_id), vault.digest(body.payload)
            with stage("vault_read"):
                record = await vault.get(key)
            direction = "mask"
            if record is not None:
                if operation == "unmask" or (operation == "process" and fingerprint == record["masked_hash"] and fingerprint != record["original_hash"]):
                    if not policy.allow_unmask:
                        fail(403, "unmask_disabled", "Демаскирование отключено для системы.")
                    direction = "unmask"
                    with stage("restore"):
                        if fingerprint == record["masked_hash"]:
                            result = restore_exact(record)
                        elif operation == "unmask" and record["mode"] == "token":
                            try:
                                result = restore_tokens(body.payload, record)
                            except ValueError:
                                fail(409, "unknown_token", "Токен не принадлежит этому запросу или отсутствует.")
                        else:
                            fail(409, "payload_conflict", "Текст не соответствует сохранённой маске.")
                elif fingerprint == record["original_hash"]:
                    if operation == "mask" and mode != record["mode"]:
                        fail(409, "mode_conflict", "Для нового режима используйте новый payload_id.")
                    result = record["masked"]
                else:
                    fail(409, "payload_conflict", "payload_id уже связан с другим текстом.")
            elif operation == "unmask":
                fail(410, "mapping_expired", "Соответствие отсутствует или срок хранения истёк.")
            else:
                with stage("detect"):
                    if len(body.payload) > 16000:
                        if large_inflight >= settings.cpu_workers:
                            fail(429, "cpu_busy", "Все обработчики больших текстов заняты; повторите запрос.")
                        large_inflight += 1
                        loop = asyncio.get_running_loop()
                        pending = loop.create_future()
                        pending.add_done_callback(lambda done: None if done.cancelled() else done.exception())

                        def release_cpu_slot():
                            nonlocal large_inflight
                            large_inflight -= 1

                        def complete_cpu_work(done):
                            release_cpu_slot()
                            try:
                                result = done.result()
                            except BaseException as exc:
                                if not pending.done():
                                    pending.set_exception(exc)
                            else:
                                if not pending.done():
                                    pending.set_result(result)

                        try:
                            work = cpu_pool.submit(partial(detect, body.payload, extra_rules=list(policy.extra_rules)))
                        except BaseException:
                            release_cpu_slot()
                            raise
                        # Cancellation affects only our waiter, never the CPU job.
                        # Its real completion releases capacity on the owning loop.
                        # Avoid shield: Python 3.14 reports detached shield failures
                        # to the loop exception hook, potentially exposing input.
                        work.add_done_callback(lambda done: loop.call_soon_threadsafe(complete_cpu_work, done))
                        spans = await pending
                    else:
                        spans = detect(body.payload, extra_rules=list(policy.extra_rules))
                selected = [s for s in spans if not policy.types or s.type in policy.types]
                type_set = {s.type for s in selected}
                if len(type_set) < policy.min_types or not set(policy.required_types).issubset(type_set):
                    selected = []
                with stage("transform"):
                    result, replacements = mask(body.payload, selected, mode)
                record = {"original_hash": fingerprint, "masked_hash": vault.digest(result),
                          "masked": result, "replacements": replacements if policy.allow_unmask else [],
                          "entities": [asdict(s) for s in selected], "mode": mode}
                with stage("vault_write"):
                    record = await vault.put_if_absent(key, record)
                if record["original_hash"] != fingerprint:
                    fail(409, "payload_conflict", "payload_id уже связан с другим текстом.")
                if operation == "mask" and record["mode"] != mode:
                    fail(409, "mode_conflict", "Для нового режима используйте новый payload_id.")
                result = record["masked"]
            elapsed = (time.perf_counter() - start) * 1000
            kinds = sorted({s["type"] for s in record["entities"]})
            for kind in kinds:
                detected.labels(kind if kind in TYPES else "CUSTOM").inc()
            chars.inc(len(body.payload))
            tokens.inc(len(body.payload) / 4)
            LOG.info(json.dumps({"event": "processed", "request_id": request.state.request_id,
                                 "system": tenant, "operation": direction, "types": kinds,
                                 "mode": record["mode"], "latency_ms": round(elapsed, 3),
                                 "stages_ms": stage_times}, ensure_ascii=False))
            if operation == "process":
                return {"result": result}
            return {"result": result, "payload_id": body.payload_id, "entities": record["entities"],
                    "types": kinds, "mode": record["mode"], "latency_ms": round(elapsed, 3)}
        except HTTPException:
            raise
        except VaultFull:
            fail(429, "vault_full", "Хранилище заполнено; повторите запрос позднее.")
        except Exception:
            # Never downgrade a failed protection operation to returning raw input.
            LOG.error(json.dumps({"event": "processing_failed", "request_id": request.state.request_id,
                                  "stages_ms": stage_times}))
            fail(503, "temporarily_unavailable", "Защита временно недоступна; исходный текст не отправлен дальше.")

    @app.post("/process", response_model=dict[str, str])
    async def process(body: ProcessRequest, request: Request):
        return await execute(body, request, "process")

    @app.post("/v1/mask")
    async def mask_endpoint(body: MaskRequest, request: Request):
        return await execute(body, request, "mask")

    @app.post("/v1/unmask")
    async def unmask_endpoint(body: ProcessRequest, request: Request):
        return await execute(body, request, "unmask")

    @app.get("/health")
    async def health():
        try:
            await vault.ping()
        except Exception:
            return error(503, "storage_unavailable", "Хранилище недоступно.")
        return {"status": "ok", "mode": "demo" if settings.demo else "restricted",
                "storage": "sentinel" if vault.sentinel else "redis" if vault.redis else "memory", "python": sys.version.split()[0],
                "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
                "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)()}

    @app.get("/v1/types")
    async def types():
        return {"types": TYPES}

    @app.get("/metrics")
    async def metrics(request: Request):
        await authorize(request)
        if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
            from prometheus_client import multiprocess
            combined = CollectorRegistry()
            multiprocess.MultiProcessCollector(combined)
            content = generate_latest(combined)
        else:
            content = generate_latest(registry)
        return Response(content, media_type="text/plain; version=0.0.4")

    web = Path(__file__).resolve().parent.parent / "web"
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
