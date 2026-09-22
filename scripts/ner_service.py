#!/usr/bin/env python3
"""Optional, private PERSON/LOCATION NER service for ordinary CPython 3.13.

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

    @classmethod
    def from_env(cls):
        return cls(token=os.getenv("SEIF_NER_TOKEN", ""), demo=os.getenv("SEIF_NER_DEMO") == "1",
                   max_model_jobs=int(os.getenv("SEIF_NER_MAX_MODEL_JOBS", "4")))


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
    results = analyzer.analyze(text=text, language="ru", entities=["PERSON", "LOCATION"], score_threshold=0.0)
    entities = []
    for item in results:
        if item.entity_type not in {"PERSON", "LOCATION"}:
            continue
        start, end, score = item.start, item.end, item.score
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid model output.")
        entities.append({"start": start, "end": end, "score": float(score), "entity_type": item.entity_type})
    entities.sort(key=lambda item: (item["start"], item["end"]))
    return {"entities": entities}


def _lifespan(settings, analyzer_factory):
    @asynccontextmanager
    async def lifespan(app):
        if not settings.demo and not settings.token:
            raise RuntimeError("SEIF_NER_TOKEN is required outside demo mode.")
        if settings.max_http_inflight < 1 or not 1 <= settings.max_model_jobs <= settings.max_http_inflight:
            raise RuntimeError("NER model capacity must be positive and bounded by HTTP capacity.")
        # Preload before readiness; avoid logging third-party exception content.
        try:
            app.state.analyzer = analyzer_factory()
        except Exception:
            raise RuntimeError("NER model initialization failed.") from None
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


# Starlette dispatches synchronous exception handlers through a worker thread.
# These handlers perform only constant-time response construction; keeping the
# async callback contract avoids an extra thread-pool hop under invalid traffic.
# No artificial checkpoint is required or useful for this bounded work.
async def _validation_error(_request, _exc):
    return error(422, "invalid_request", "Expected one text string of at most 20000 characters.")


async def _http_error(_request, exc):
    return error(exc.status_code, "invalid_request", INVALID_REQUEST)


async def _unexpected_error(_request, _exc):
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
        job = state.pool.submit(infer, state.analyzer, text)
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
    app = FastAPI(title="СЕЙФ private PERSON/LOCATION NER", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
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
        return JSONResponse({"status": "ok", "model": "ru_core_news_sm", "entities": ["PERSON", "LOCATION"]},
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
    parser.add_argument("--workers", type=int, default=int(os.getenv("SEIF_NER_WORKERS", "4")))
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
