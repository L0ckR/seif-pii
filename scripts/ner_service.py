#!/usr/bin/env python3
"""Optional private NER service with explicit model capabilities.

Install deploy/ner/requirements-ner.txt in a separate environment. The model
must already be installed: this service never downloads a model at startup.
Run with SEIF_NER_TOKEN set; SEIF_NER_DEMO=1 is for loopback demos only.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import math
import os
from concurrent.futures import CancelledError as WorkerCancelledError
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.exceptions import HTTPException

from seif.async_callbacks import immediate_response
from seif.ner_contract import MAX_ENTITIES, analyzer_entities, validate_entity

LOG = logging.getLogger("seif.ner")
MAX_TEXT_CHARS = 20_000
MAX_BODY_BYTES = 128 * 1024
BODY_READ_TIMEOUT_SECONDS = 5.0
NER_UNAVAILABLE = "NER temporarily unavailable."
INVALID_REQUEST = "Invalid request."


@dataclass(frozen=True)
class NerSettings:
    token: str = field(default="", repr=False)
    demo: bool = False
    max_http_inflight: int = 16
    max_model_jobs: int = 4
    batch_size: int = 1
    batch_wait_ms: float = 1.0

    @classmethod
    def from_env(cls):
        return cls(token=os.getenv("SEIF_NER_TOKEN", ""), demo=os.getenv("SEIF_NER_DEMO") == "1",
                   max_model_jobs=int(os.getenv("SEIF_NER_MAX_MODEL_JOBS", "4")),
                   max_http_inflight=int(os.getenv("SEIF_NER_MAX_HTTP_INFLIGHT", "16")),
                   batch_size=int(os.getenv("SEIF_NER_BATCH_SIZE", "1")),
                   batch_wait_ms=float(os.getenv("SEIF_NER_BATCH_WAIT_MS", "1")))


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(max_length=MAX_TEXT_CHARS)


def error(status, code, message):
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    if status == 429:
        headers["Retry-After"] = "1"
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class Boundary:
    """Authenticate before reading, cap buffered bodies and active HTTP work."""

    def __init__(self, app, settings):
        self.app, self.settings, self.inflight = app, settings, 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        # Kubelet may probe anonymously; a gateway's supplied credential must
        # still be validated so readiness detects a mismatched shared secret.
        anonymous_probe = scope.get("path") == "/health" and scope.get("method") == "GET"
        if (not anonymous_probe or b"authorization" in headers) and not self.settings.demo:
            expected = b"Bearer " + self.settings.token.encode("utf-8")
            if not self.settings.token or not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                return await error(401, "unauthorized", "Authentication required.")(scope, receive, send)
        if self.inflight >= self.settings.max_http_inflight:
            return await error(429, "busy", "Service capacity reached.")(scope, receive, send)
        self.inflight += 1
        response_started = False

        async def safe_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._dispatch_http(scope, receive, send, safe_send, headers)
        except Exception:
            # ServerErrorMiddleware re-raises handled exceptions for server logs.
            # Catch here so a third-party exception cannot leak input to Uvicorn.
            LOG.warning("ner_request_failed")
            if not response_started:
                return await error(503, "unavailable", NER_UNAVAILABLE)(scope, receive, send)
        finally:
            self.inflight -= 1

    @staticmethod
    def _check_body_size(headers):
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return error(400, "invalid_request", INVALID_REQUEST)
        if content_length < 0:
            return error(400, "invalid_request", INVALID_REQUEST)
        if content_length > MAX_BODY_BYTES:
            return error(413, "too_large", "Request body is too large.")
        return None

    @staticmethod
    async def _read_post_body(receive):
        """Return the buffered body or an error after the upload timeout exits."""
        body = bytearray()
        try:
            async with asyncio.timeout(BODY_READ_TIMEOUT_SECONDS):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return None, None
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        return None, error(413, "too_large", "Request body is too large.")
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return None, error(408, "request_timeout", "Request body timed out.")
        return body, None

    async def _dispatch_http(self, scope, receive, send, safe_send, headers):
        rejected = self._check_body_size(headers)
        if rejected is not None:
            return await rejected(scope, receive, send)
        if scope["method"] != "POST":
            return await self.app(scope, receive, safe_send)
        body, rejected = await self._read_post_body(receive)
        if rejected is not None:
            return await rejected(scope, receive, send)
        if body is None:
            return
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, safe_send)


def build_analyzer():
    # Libraries are intentionally lazy: core API/test environments need no NLP.
    # Set before importing NumPy, avoiding hidden per-worker BLAS thread pools.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    backend = os.getenv("SEIF_NER_BACKEND", "presidio")
    if backend == "rubert":
        from seif.rubert_ner import RubertAnalyzer

        return RubertAnalyzer.from_env()
    if backend == "gliner":
        from seif.gliner_ner import GlinerAnalyzer

        return GlinerAnalyzer.from_env()
    if backend != "presidio":
        raise RuntimeError("SEIF_NER_BACKEND must be presidio, gliner or rubert.")
    import spacy.util
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import SpacyRecognizer

    if not spacy.util.is_package("ru_core_news_sm"):
        raise RuntimeError("Required local Russian NLP model is missing; install the locked environment.")
    engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ru", "model_name": "ru_core_news_sm"}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION"},
            "labels_to_ignore": [],
        },
    }).create_engine()
    # This service requests PERSON and LOCATION only and has no lemma/context recognizers.
    # Keep the pretrained token encoder and NER weights; omit unrelated syntax,
    # morphology and lemmatization work (76-case equivalence checked separately).
    engine.nlp["ru"].disable_pipes("morphologizer", "parser", "attribute_ruler", "lemmatizer")
    registry = RecognizerRegistry(supported_languages=["ru"])
    registry.add_recognizer(SpacyRecognizer(supported_language="ru", supported_entities=["PERSON", "LOCATION"]))
    return AnalyzerEngine(registry=registry, nlp_engine=engine, supported_languages=["ru"], default_score_threshold=0.0)


def infer(analyzer, text):
    requested = analyzer_entities(analyzer)
    results = analyzer.analyze(text=text, language="ru", entities=requested, score_threshold=0.0)
    return _format_results(text, results, requested)


def infer_batch(analyzer, texts):
    requested = analyzer_entities(analyzer)
    results = analyzer.analyze_batch(texts=texts, language="ru", entities=requested, score_threshold=0.0)
    if not isinstance(results, list) or len(results) != len(texts):
        raise ValueError("Invalid model batch count.")
    return [_format_results(text, result, requested) for text, result in zip(texts, results, strict=True)]


def _format_results(text, results, requested):
    if not isinstance(results, list) or len(results) > MAX_ENTITIES:
        raise ValueError("Invalid model result count.")
    entities = []
    for item in results:
        if item.entity_type not in requested:
            continue
        start, end, score = item.start, item.end, item.score
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid model output.")
        entity = {"start": start, "end": end, "score": float(score), "entity_type": item.entity_type}
        validate_entity(entity, len(text))
        entities.append(entity)
        if len(entities) > MAX_ENTITIES:
            raise ValueError("NER output exceeds entity count limit.")
    entities.sort(key=lambda item: (item["start"], item["end"]))
    return {"entities": entities}


def _lifespan(settings, analyzer_factory):
    @asynccontextmanager
    async def lifespan(app):
        if not settings.demo and not settings.token:
            raise RuntimeError("SEIF_NER_TOKEN is required outside demo mode.")
        if settings.max_http_inflight < 1 or not 1 <= settings.max_model_jobs <= settings.max_http_inflight:
            raise RuntimeError("NER model capacity must be positive and bounded by HTTP capacity.")
        if (type(settings.batch_size) is not int or settings.batch_size not in (1, 2, 4, 8, 16, 32)
                or settings.batch_size > settings.max_model_jobs
                or not math.isfinite(settings.batch_wait_ms) or not 0 <= settings.batch_wait_ms <= 20):
            raise RuntimeError("Invalid NER batch size, delay or model capacity.")
        # Preload before readiness; avoid logging third-party exception content.
        try:
            app.state.analyzer = analyzer_factory()
            analyzer_entities(app.state.analyzer)
        except Exception:
            raise RuntimeError("NER model initialization failed.") from None
        app.state.batched = settings.batch_size > 1
        if app.state.batched:
            from seif.ner_batching import BatchExecutor

            if not callable(getattr(app.state.analyzer, "analyze_batch", None)):
                raise RuntimeError("Configured analyzer does not support batching.")
            pool = BatchExecutor(lambda texts: infer_batch(app.state.analyzer, texts),
                                 batch_size=settings.batch_size, wait_ms=settings.batch_wait_ms,
                                 capacity=settings.max_model_jobs)
        else:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seif-ner")
        app.state.pool = pool
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
            app.state.analyzer = None
    return lifespan


# Bounded response construction stays on the request's event loop. Successful
# model operations retain their regular asynchronous implementation.
@immediate_response
def _validation_error(_request, _exc):
    return error(422, "invalid_request", "Expected one text string of at most 20000 characters.")


@immediate_response
def _http_error(_request, exc):
    return error(exc.status_code, "invalid_request", INVALID_REQUEST)


@immediate_response
def _unexpected_error(_request, _exc):
    LOG.warning("ner_request_failed")
    return error(503, "unavailable", NER_UNAVAILABLE)


def _consume_model_exception(future):
    if not future.cancelled():
        future.exception()


def _complete_model_job(state, waiter, job):
    # Runs on the owner loop after real CPU completion. Future.exception()
    # transfers even BaseException failures without raising on the loop, where
    # a third-party failure could otherwise expose private input in a traceback.
    state.model_inflight -= 1
    exception = WorkerCancelledError() if job.cancelled() else job.exception()
    if waiter.done():
        return
    if exception is not None:
        waiter.set_exception(exception)
    else:
        waiter.set_result(job.result())


async def _run_model(state, text):
    state.model_inflight += 1
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()
    waiter.add_done_callback(_consume_model_exception)
    try:
        job = (state.pool.submit(text) if getattr(state, "batched", False)
               else state.pool.submit(infer, state.analyzer, text))
    except Exception:
        state.model_inflight -= 1
        return error(503, "unavailable", NER_UNAVAILABLE)
    job.add_done_callback(lambda future: loop.call_soon_threadsafe(_complete_model_job, state, waiter, future))
    try:
        result = await waiter
    except asyncio.CancelledError:
        # Queued jobs can be cancelled. Running CPU work owns capacity until
        # the concurrent future's completion callback, including after timeout.
        job.cancel()
        raise
    except Exception:
        LOG.warning("ner_inference_failed")
        return error(503, "unavailable", NER_UNAVAILABLE)
    return JSONResponse(result, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


def create_app(settings=None, analyzer_factory=None):
    settings = settings or NerSettings.from_env()
    lifespan = _lifespan(settings, analyzer_factory or build_analyzer)
    app = FastAPI(title="СЕЙФ private PII NER", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ready = False
    app.state.model_inflight = 0
    app.add_middleware(Boundary, settings=settings)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)

    @app.get("/health")
    async def health():
        if not app.state.ready:
            return error(503, "not_ready", "NER not ready.")
        model_name = getattr(app.state.analyzer, "model_name", "ru_core_news_sm")
        return JSONResponse({"status": "ok", "model": model_name, "entities": analyzer_entities(app.state.analyzer)},
                            headers={"Cache-Control": "no-store"})

    @app.post("/analyze")
    async def analyze(body: AnalyzeRequest):
        if not app.state.ready:
            return error(503, "not_ready", "NER not ready.")
        if app.state.model_inflight >= settings.max_model_jobs:
            return error(429, "busy", "Model capacity reached.")
        if not body.text:
            return JSONResponse({"entities": []}, headers={"Cache-Control": "no-store"})
        return await _run_model(app.state, body.text)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("SEIF_NER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SEIF_NER_PORT", "8770")))
    default_workers = "1" if os.getenv("SEIF_NER_BACKEND") == "rubert" else "4"
    parser.add_argument("--workers", type=int, default=int(os.getenv("SEIF_NER_WORKERS", default_workers)))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if os.getenv("SEIF_NER_DEMO") == "1" and args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("demo mode must bind to loopback")
    import uvicorn

    uvicorn.run("scripts.ner_service:create_app", factory=True, host=args.host, port=args.port,
                workers=args.workers, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
