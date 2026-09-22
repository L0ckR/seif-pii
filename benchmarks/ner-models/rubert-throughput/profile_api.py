"""Diagnostic event-loop CPU profile; profiled throughput is not a capacity result."""
import cProfile
import importlib
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def create_app():
    from seif.app import create_app as factory

    app = factory()
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original(application):
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                yield
            finally:
                profiler.disable()
                profiler.dump_stats(os.environ["SEIF_API_CPU_PROFILE"])

    app.router.lifespan_context = lifespan
    return app


def main():
    scaling = importlib.import_module("benchmarks.ner-models.rubert-throughput.http_scaling")
    launch = scaling.smoke.start_service

    def profiled(spec, root, env, log, children):
        python, module, label = spec
        if label.startswith("api"):
            module = "benchmarks.ner-models.rubert-throughput.profile_api:create_app"
            env = {**env, "SEIF_API_CPU_PROFILE": str(ROOT / f"local-data/rubert-throughput/{label}.pstats")}
        return launch((python, module, label), root, env, log, children)

    scaling.smoke.start_service = profiled
    return scaling.main()


if __name__ == "__main__":
    raise SystemExit(main())
