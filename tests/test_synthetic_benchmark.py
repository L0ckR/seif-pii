"""Bad load parameters must fail before opening connections or scheduling work."""
import argparse
import asyncio

import pytest
from aiohttp import web

from scripts.benchmark import benchmark, validate_options


def options(**updates):
    return argparse.Namespace(**({"rps": 1000, "duration": 300, "concurrency": 256, "timeout": 10} | updates))


@pytest.mark.parametrize("name", ["rps", "duration", "timeout"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 0, -1])
def test_nonfinite_and_nonpositive_rates_are_rejected_without_network(name, value):
    with pytest.raises(ValueError, match="finite and positive"):
        asyncio.run(benchmark(options(**{name: value})))


@pytest.mark.parametrize("changes", [
    {"rps": 1e300}, {"duration": 3601}, {"timeout": 121}, {"concurrency": 0},
    {"concurrency": 8193}, {"concurrency": True}, {"concurrency": 2.5},
])
def test_excessive_work_is_rejected(changes):
    with pytest.raises(ValueError, match="bounded"):
        validate_options(options(**changes))


def test_normal_five_minute_target_is_valid():
    validate_options(options())


def test_redirect_does_not_forward_api_credentials_or_benchmark_load():
    async def scenario():
        forwarded = []

        async def destination(request):
            forwarded.append(dict(request.headers))
            return web.json_response({"result": "unexpected"})

        destination_app = web.Application()
        destination_app.router.add_route("*", "/process", destination)
        destination_runner = web.AppRunner(destination_app)
        await destination_runner.setup()
        destination_site = web.TCPSite(destination_runner, "127.0.0.1", 0)
        await destination_site.start()
        port = destination_runner.addresses[0][1]

        async def redirect(_request):
            raise web.HTTPTemporaryRedirect(f"http://127.0.0.1:{port}/process")

        source_app = web.Application()
        source_app.router.add_post("/process", redirect)
        source_runner = web.AppRunner(source_app)
        try:
            await source_runner.setup()
            await web.TCPSite(source_runner, "127.0.0.1", 0).start()
            args = options(rps=1000, duration=.002)
            args.url = f"http://127.0.0.1:{source_runner.addresses[0][1]}"
            args.system, args.api_key = "test-system", "test-key"
            report = await benchmark(args)
            assert report["errors"] == {"http_307": 1}
            assert report["counts"]["pairs_mask_failed"] == 1
            assert forwarded == []
        finally:
            await source_runner.cleanup()
            await destination_runner.cleanup()

    asyncio.run(scenario())
