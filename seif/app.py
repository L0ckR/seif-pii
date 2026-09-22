"""HTTP boundary: authenticated, retry-safe and fail-closed."""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import sys
import sysconfig
import time
import uuid
from concurrent.futures import CancelledError as WorkerCancelledError
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Policy, Settings
from .detector import TYPES, detect, merge_ner_candidates, validate_extra_rule
from .ner import NerClient, validate_ner_settings
from .request_capture import RequestCapture
from .transform import RestorationTooLarge, mask, restore_exact, restore_tokens
from .vault import Vault, VaultFull

LOG = logging.getLogger("seif.audit")
PROCESS_ROUTE = "/process"
MASK_ROUTE = "/v1/mask"
UNMASK_ROUTE = "/v1/unmask"


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


def capture_origin(scope, headers):
    """Record selected provenance fields, never credentials or arbitrary headers.

    ASGI client may already reflect Uvicorn's trusted proxy handling. Preserve
    the Cloudflare address separately; neither is an organizer identity.
    """
    def address(value):
        try:
            return str(ipaddress.ip_address(value.strip()))
        except ValueError:
            return None

    def header(name, limit=512):
        return headers.get(name, b"")[:limit].decode("latin-1")

    client = scope.get("client")
    client_ip = address(client[0]) if client else None
    cf_ip = address(header(b"cf-connecting-ip", 64))
    forwarded = header(b"x-forwarded-for", 1024)
    chain = [address(value) for value in forwarded.split(",")] if forwarded else []
    return {
        "client_ip": client_ip,
        "cf_connecting_ip": cf_ip,
        "source_ip": cf_ip or client_ip,
        "ip_source": "cf_connecting_ip_header" if cf_ip else "asgi_client",
        "forwarded_for": ", ".join(chain) if chain and all(chain) else None,
        "cf_ray": header(b"cf-ray", 128),
        "user_agent": header(b"user-agent"),
        "capture_probe": header(b"x-seif-capture-probe", 128),
        "organizer_identity_verified": False,
    }


@dataclass(slots=True)
class BufferedRequestBody:
    """Request-owned bytes remain reserved until the complete ASGI call ends."""

    held_bytes: int = 0
    content: bytes = b""
    complete: bool = False


class Boundary:
    """Pure ASGI middleware avoids per-request task overhead and raw access logs."""
    def __init__(self, app, settings, registry, capture=None):
        self.app, self.settings, self.inflight = app, settings, 0
        self.capture = capture
        self.buffered_bytes = 0
        self.count = Counter("seif_http_requests_total", "Requests by fixed route and status", ["route", "status"], registry=registry)
        self.latency = Histogram("seif_http_duration_seconds", "Full request duration", ["route"],
                                 buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10), registry=registry)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        route = path if path in {PROCESS_ROUTE, MASK_ROUTE, UNMASK_ROUTE, "/health", "/metrics", "/v1/types"} else "other"
        start, request_id, status = time.perf_counter(), uuid.uuid4().hex, 500
        overloaded = self.inflight >= self.settings.max_inflight
        self.inflight += 1
        body = BufferedRequestBody()
        scope.setdefault("state", {})["request_id"] = request_id
        headers = dict(scope.get("headers", []))
        capture_this = self.capture is not None and scope["method"] == "POST" and path in {
            PROCESS_ROUTE, MASK_ROUTE, UNMASK_ROUTE,
        }
        received_at = datetime.now(timezone.utc).isoformat() if capture_this else None

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
            await self._dispatch_http(scope, receive, safe_send, headers, body)
        finally:
            self.buffered_bytes -= body.held_bytes
            self.inflight -= 1
            self.count.labels(route, str(status)).inc()
            self.latency.labels(route).observe(time.perf_counter() - start)
            if capture_this:
                self.capture.submit({
                    "received_at": received_at,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "request_id": request_id, "route": route, "method": scope["method"],
                    "status_code": status, "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
                    "body_complete": body.complete,
                    "operation": scope["state"].get("capture_operation"),
                    **capture_origin(scope, headers),
                }, body.content)

    def _check_body_size(self, headers, scope):
        try:
            content_length = int(headers.get(b"content-length", b"0"))
            if content_length < 0:
                raise ValueError
        except ValueError:
            return error(400, "invalid_request", "Некорректный размер запроса.")
        if content_length > self.settings.max_body_bytes:
            return error(413, "too_large", "Превышен размер запроса.")
        if scope["method"] == "POST" and content_length > self.settings.max_inflight_body_bytes - self.buffered_bytes:
            return error(429, "body_budget", "Память обработки запросов занята; повторите запрос.", {"Retry-After": "1"})
        return None

    async def _read_post_body(self, receive, body):
        """Collect within one upload deadline; leave response sending to dispatch."""
        chunks = bytearray()
        try:
            async with asyncio.timeout(self.settings.request_body_timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return None
                    chunk = message.get("body", b"")
                    if body.held_bytes + len(chunk) > self.settings.max_body_bytes:
                        return error(413, "too_large", "Превышен размер запроса.")
                    if self.buffered_bytes + len(chunk) > self.settings.max_inflight_body_bytes:
                        return error(429, "body_budget", "Память обработки запросов занята; повторите запрос.", {"Retry-After": "1"})
                    self.buffered_bytes += len(chunk)
                    body.held_bytes += len(chunk)
                    chunks.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return error(408, "request_timeout", "Истекло время получения запроса.")
        body.content, body.complete = bytes(chunks), True
        return None

    async def _dispatch_http(self, scope, receive, send, headers, body):
        rejected = self._check_body_size(headers, scope)
        if rejected is not None:
            return await rejected(scope, receive, send)
        if scope["method"] != "POST":
            return await self.app(scope, receive, send)
        rejected = await self._read_post_body(receive, body)
        if rejected is not None:
            return await rejected(scope, receive, send)
        if not body.complete:
            return
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body.content, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def _validate_policies(settings: Settings) -> None:
    for policy in settings.policies.values():
        if type(policy.masking_enabled) is not bool:
            raise ValueError("masking_enabled must be a boolean")
        if policy.mode not in policy.allowed_modes or set(policy.allowed_modes) - {"mask", "token", "synthetic"}:
            raise ValueError("Invalid allowed modes")
        if len(policy.extra_rules) > 32:
            raise ValueError("At most 32 custom rules are supported")
        for rule in policy.extra_rules:
            validate_extra_rule(rule)
        known_types = set(TYPES) | {rule["type"] for rule in policy.extra_rules}
        if (set(policy.types) | set(policy.required_types)) - known_types:
            raise ValueError("Unknown entity type in system policy")


@dataclass(slots=True)
class _OperationContext:
    """Validated request policy and correlation data for one operation."""

    body: ProcessRequest
    policy: Policy
    operation: str
    mode: str
    key: str
    fingerprint: str


class _AppContext:
    """Shared request-processing state and handlers for the FastAPI app."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ner = NerClient(settings.ner_url, settings.ner_token, settings.ner_timeout_seconds) if settings.ner_url else None
        self.vault = Vault(settings)
        self.registry = CollectorRegistry()
        self.cpu_pool = ThreadPoolExecutor(max_workers=settings.cpu_workers, thread_name_prefix="seif-detect")
        self.capture = RequestCapture.from_env()
        self.large_inflight = 0
        self.limits = {}
        self.detected = Counter("seif_entities_total", "Detected entity types, never values", ["type"], registry=self.registry)
        self.chars = Counter("seif_input_characters_total", "Processed input Unicode characters", registry=self.registry)
        self.tokens = Counter("seif_input_tokens_estimated_total", "Estimated tokens = characters / 4, not a model tokenizer", registry=self.registry)
        self.stages = Histogram("seif_stage_duration_seconds", "Processing stages", ["stage"], registry=self.registry)

    @asynccontextmanager
    async def lifespan(self, app):
        logging.basicConfig(level=os.getenv("SEIF_LOG_LEVEL", "INFO"), format="%(message)s")
        try:
            if os.getenv("SEIF_REQUIRE_FREE_THREADING", "0") == "1" and (
                not sysconfig.get_config_var("Py_GIL_DISABLED") or getattr(sys, "_is_gil_enabled", lambda: True)()
            ):
                raise RuntimeError("This deployment requires a free-threaded Python with GIL disabled")
            await self.vault.ping()
            if self.ner:
                await self.ner.health()
            yield
        finally:
            try:
                # Running detector jobs cannot be killed safely. Drain them without
                # blocking the event loop, including jobs whose HTTP task was cancelled.
                await asyncio.to_thread(self.cpu_pool.shutdown, wait=True, cancel_futures=True)
            finally:
                try:
                    if self.ner:
                        await self.ner.close()
                finally:
                    try:
                        await self.vault.close()
                    finally:
                        if self.capture:
                            await asyncio.to_thread(self.capture.close)

    async def authorize(self, request):
        settings = self.settings
        tenant = request.headers.get("X-System-ID", "demo" if settings.demo else "")
        policy = settings.policies.get(tenant)
        if policy is None or not policy.enabled:
            fail(403, "access_denied", "Система не допущена к сервису.")
        supplied = request.headers.get("X-API-Key", "")
        anonymous_demo = settings.demo and tenant == "demo" and not policy.api_key
        if not anonymous_demo and (not policy.api_key or not hmac.compare_digest(supplied.encode(), policy.api_key.encode())):
            fail(401, "unauthorized", "Требуется ключ системы.")
        # Fixed bounded keyspace: rate state exists only for configured tenants.
        if self.vault.redis:
            limiter_key = "seif:rate:" + self.vault.digest(tenant)
            allowed = await self.vault.redis.eval("local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],1) end; return n", 1, limiter_key)
            if allowed > policy.rps:
                fail(429, "rate_limit", "Превышена частота запросов системы.")
        else:
            now = time.monotonic()
            balance, last = self.limits.get(tenant, (float(policy.rps), now))
            balance = min(float(policy.rps), balance + (now - last) * policy.rps)
            if balance < 1:
                self.limits[tenant] = (balance, now)
                fail(429, "rate_limit", "Превышена частота запросов системы.")
            self.limits[tenant] = (balance - 1, now)
        return tenant, policy

    def _restore_record(self, context, record, stage):
        policy, body = context.policy, context.body
        operation, fingerprint = context.operation, context.fingerprint
        if not policy.allow_unmask:
            fail(403, "unmask_disabled", "Демаскирование отключено для системы.")
        with stage("restore"):
            if fingerprint == record["masked_hash"]:
                return restore_exact(record)
            if operation == "unmask" and record["mode"] == "token":
                try:
                    return restore_tokens(body.payload, record, max_output_chars=self.settings.max_payload_chars)
                except RestorationTooLarge:
                    fail(413, "too_large", "Превышен лимит восстановленного текста.")
                except ValueError:
                    fail(409, "unknown_token", "Токен не принадлежит этому запросу или отсутствует.")
            fail(409, "payload_conflict", "Текст не соответствует сохранённой маске.")

    async def _detect_large(self, policy, body):
        if self.large_inflight >= self.settings.cpu_workers:
            fail(429, "cpu_busy", "Все обработчики больших текстов заняты; повторите запрос.")
        self.large_inflight += 1
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        pending.add_done_callback(lambda done: None if done.cancelled() else done.exception())

        def release_cpu_slot():
            self.large_inflight -= 1

        def complete_cpu_work(done):
            release_cpu_slot()
            if pending.done():
                return
            # Worker failures belong to the awaiting request, including custom
            # BaseException subclasses. Reading exception() transfers them without
            # raising on the event loop or leaking their possibly sensitive text.
            exception = WorkerCancelledError() if done.cancelled() else done.exception()
            if exception is not None:
                pending.set_exception(exception)
            else:
                pending.set_result(done.result())

        try:
            work = self.cpu_pool.submit(partial(detect, body.payload, extra_rules=list(policy.extra_rules)))
        except BaseException:
            release_cpu_slot()
            raise
        # Cancellation affects only our waiter, never the CPU job.
        # Its real completion releases capacity on the owning loop.
        # Avoid shield: Python 3.14 reports detached shield failures
        # to the loop exception hook, potentially exposing input.
        work.add_done_callback(lambda done: loop.call_soon_threadsafe(complete_cpu_work, done))
        return await pending

    def _select_spans(self, spans, policy):
        selected = [s for s in spans if not policy.types or s.type in policy.types]
        type_set = {s.type for s in selected}
        if len(type_set) < policy.min_types or not set(policy.required_types).issubset(type_set):
            selected = []
        if not policy.masking_enabled:
            # Recognition and NER failures still fail closed. Only the
            # transformation selection changes under this explicit policy.
            selected = []
        return selected, type_set

    async def _detect_and_transform(self, context, stage):
        policy, body = context.policy, context.body
        operation, mode = context.operation, context.mode
        key, fingerprint = context.key, context.fingerprint
        with stage("detect"):
            if len(body.payload) > 16000:
                spans = await self._detect_large(policy, body)
            else:
                spans = detect(body.payload, extra_rules=list(policy.extra_rules))
        if self.ner:
            with stage("ner"):
                candidates = await self.ner.detect(body.payload)
                spans = merge_ner_candidates(body.payload, spans, candidates)
        selected, type_set = self._select_spans(spans, policy)
        with stage("transform"):
            result, replacements = mask(body.payload, selected, mode)
        record = {"original_hash": fingerprint, "masked_hash": self.vault.digest(result),
                  "masked": result, "replacements": replacements if policy.allow_unmask else [],
                  "entities": [asdict(s) for s in selected], "mode": mode,
                  "detected_types": sorted(type_set),
                  "masking_enabled": policy.masking_enabled,
                  "detector_profile": "hybrid" if self.ner else "rules"}
        with stage("vault_write"):
            record = await self.vault.put_if_absent(key, record)
        if record["original_hash"] != fingerprint:
            fail(409, "payload_conflict", "payload_id уже связан с другим текстом.")
        if operation == "mask" and record["mode"] != mode:
            fail(409, "mode_conflict", "Для нового режима используйте новый payload_id.")
        return record

    async def _resolve_record(self, context, record, stage):
        operation, fingerprint, mode = context.operation, context.fingerprint, context.mode
        direction = "mask"
        if record is not None:
            if operation == "unmask" or (operation == "process" and fingerprint == record["masked_hash"] and fingerprint != record["original_hash"]):
                direction = "unmask"
                result = self._restore_record(context, record, stage)
            elif fingerprint == record["original_hash"]:
                if operation == "mask" and mode != record["mode"]:
                    fail(409, "mode_conflict", "Для нового режима используйте новый payload_id.")
                result = record["masked"]
            else:
                fail(409, "payload_conflict", "payload_id уже связан с другим текстом.")
        elif operation == "unmask":
            fail(410, "mapping_expired", "Соответствие отсутствует или срок хранения истёк.")
        else:
            record = await self._detect_and_transform(context, stage)
            result = record["masked"]
        return direction, result, record

    async def execute(self, body: ProcessRequest, request: Request, operation: str):
        start = time.perf_counter()
        stage_times = {}

        @contextmanager
        def stage(name):
            stage_start = time.perf_counter()
            try:
                yield
            finally:
                seconds = time.perf_counter() - stage_start
                self.stages.labels(name).observe(seconds)
                stage_times[name] = round(seconds * 1000, 3)

        try:
            with stage("authorize"):
                tenant, policy = await self.authorize(request)
            if len(body.payload) > self.settings.max_payload_chars:
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
                key, fingerprint = self.vault.key(tenant, body.payload_id), self.vault.digest(body.payload)
            with stage("vault_read"):
                record = await self.vault.get(key)
            context = _OperationContext(body, policy, operation, mode, key, fingerprint)
            direction, result, record = await self._resolve_record(context, record, stage)
            elapsed = (time.perf_counter() - start) * 1000
            kinds = sorted({s["type"] for s in record["entities"]})
            for kind in kinds:
                self.detected.labels(kind if kind in TYPES else "CUSTOM").inc()
            self.chars.inc(len(body.payload))
            self.tokens.inc(len(body.payload) / 4)
            request.state.capture_operation = direction
            LOG.info(json.dumps({"event": "processed", "request_id": request.state.request_id,
                                 "system": tenant, "operation": direction, "types": kinds,
                                 "detected_types": record.get("detected_types", kinds),
                                 "mode": record["mode"], "latency_ms": round(elapsed, 3),
                                 "masking_enabled": record.get("masking_enabled", True),
                                 "detector_profile": record.get("detector_profile", "rules"),
                                 "stages_ms": stage_times}, ensure_ascii=False))
            if operation == "process":
                return {"result": result}
            return {"result": result, "payload_id": body.payload_id, "entities": record["entities"],
                    "types": kinds, "mode": record["mode"], "latency_ms": round(elapsed, 3),
                    "masking_enabled": record.get("masking_enabled", True)}
        except HTTPException:
            raise
        except VaultFull:
            fail(429, "vault_full", "Хранилище заполнено; повторите запрос позднее.")
        except Exception:
            # Never downgrade a failed protection operation to returning raw input.
            LOG.error(json.dumps({"event": "processing_failed", "request_id": request.state.request_id,
                                  "stages_ms": stage_times}))
            fail(503, "temporarily_unavailable", "Защита временно недоступна; исходный текст не отправлен дальше.")

    async def process(self, body: ProcessRequest, request: Request):
        return await self.execute(body, request, "process")

    async def mask(self, body: MaskRequest, request: Request):
        return await self.execute(body, request, "mask")

    async def unmask(self, body: ProcessRequest, request: Request):
        return await self.execute(body, request, "unmask")

    async def health(self):
        try:
            await self.vault.ping()
            if self.ner:
                await self.ner.health()
        except Exception:
            return error(503, "storage_unavailable", "Хранилище недоступно.")
        if self.vault.sentinel:
            storage = "sentinel"
        elif self.vault.redis:
            storage = "redis"
        else:
            storage = "memory"
        return {"status": "ok", "mode": "demo" if self.settings.demo else "restricted",
                "storage": storage, "python": sys.version.split()[0],
                "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
                "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
                "detector_profile": "hybrid" if self.ner else "rules"}

    def types(self):
        """Return fixed type metadata as a bounded synchronous endpoint."""
        return {"types": TYPES}

    async def metrics(self, request: Request):
        await self.authorize(request)
        if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
            from prometheus_client import multiprocess
            combined = CollectorRegistry()
            multiprocess.MultiProcessCollector(combined)
            content = generate_latest(combined)
        else:
            content = generate_latest(self.registry)
        return Response(content, media_type="text/plain; version=0.0.4")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    validate_ner_settings(settings.ner_url, settings.ner_token, settings.ner_timeout_seconds)
    if settings.ttl_seconds < 1 or settings.max_records < 1 or len(settings.encryption_key) != 32:
        raise ValueError("Invalid vault settings")
    # Detect invalid plugin rules on startup, before any sensitive request arrives.
    _validate_policies(settings)
    # Validation can fail synchronously, before lifespan cleanup exists.
    # Allocate clients and the worker pool only after all policies are valid.
    ctx = _AppContext(settings)

    app = FastAPI(title="СЕЙФ · Personal Data Gateway", version="1.0.0", lifespan=ctx.lifespan,
                  docs_url=None, redoc_url=None)
    app.state.vault, app.state.settings = ctx.vault, settings
    app.state.ner = ctx.ner
    app.state.capture = ctx.capture
    app.add_middleware(Boundary, settings=settings, registry=ctx.registry, capture=ctx.capture)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default errors include the input. Never echo them.
        return error(422, "invalid_request", "Ожидаются строковые payload и непустой payload_id (до 256 символов).")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": "Запрос не выполнен."}
        return error(exc.status_code, detail["code"], detail["message"], exc.headers)

    app.post(PROCESS_ROUTE, response_model=dict[str, str])(ctx.process)
    app.post(MASK_ROUTE)(ctx.mask)
    app.post(UNMASK_ROUTE)(ctx.unmask)
    app.get("/health")(ctx.health)
    app.get("/v1/types")(ctx.types)
    app.get("/metrics")(ctx.metrics)

    web = Path(__file__).resolve().parent.parent / "web"
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
