"""Reproduce inherited-fd HTTP transport behavior without loading any model.

Runs four isolated loopback Uvicorn configurations. The two-message ASGI
response exposes header/body delay, while each protocol records actual accepted
socket TCP_NODELAY. This is a transport diagnostic, not service throughput.
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOCKET_INFO = {}


async def application(scope, receive, send):
    if scope["type"] != "http":
        return
    content = json.dumps(SOCKET_INFO).encode()
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(content)).encode())]})
    await send({"type": "http.response.body", "body": content})


def serve(args):
    import uvicorn

    if args.http == "h11":
        from uvicorn.protocols.http.h11_impl import H11Protocol as Protocol
    else:
        from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol as Protocol

    class ObservedProtocol(Protocol):
        def connection_made(self, transport):
            super().connection_made(transport)
            sock = transport.get_extra_info("socket")
            SOCKET_INFO.update(family=int(sock.family), proto=sock.proto,
                               tcp_nodelay=sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))

    uvicorn.run(application, fd=args.fd, http=ObservedProtocol, loop=args.loop,
                lifespan="off", access_log=False, log_level="warning")


def sample(connection):
    started = time.perf_counter()
    connection.request("GET", "/")
    response = connection.getresponse()
    headers_at = time.perf_counter()
    info = json.loads(response.read())
    ended = time.perf_counter()
    if response.status != 200:
        raise RuntimeError("Diagnostic HTTP request failed")
    return {"headers_ms": (headers_at - started) * 1000, "body_ms": (ended - headers_at) * 1000,
            "total_ms": (ended - started) * 1000, "server_socket": info,
            "client_tcp_nodelay": connection.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)}


def run_one(args, loop, protocol):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener, tempfile.TemporaryFile() as log:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        argv = [str(args.python), str(Path(__file__).resolve()), "--fd", str(listener.fileno()),
                "--loop", loop, "--http", protocol]
        process = subprocess.Popen(  # noqa: S603
            argv, cwd=ROOT, pass_fds=(listener.fileno(),), stdout=log, stderr=log,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        connection = HTTPConnection("127.0.0.1", listener.getsockname()[1], timeout=5)
        rows = []
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Diagnostic child failed to start")
                try:
                    rows.append(sample(connection))
                    break
                except (OSError, TimeoutError):
                    connection.close()
                    time.sleep(.1)
            if not rows:
                raise RuntimeError("Diagnostic readiness deadline")
            rows.extend(sample(connection) for _ in range(5))
        finally:
            connection.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return {"loop": loop, "http_protocol": protocol, "first_request": rows[0], "reused_samples": rows[1:],
                "reused_median_ms": {key: statistics.median(row[key] for row in rows[1:])
                                      for key in ("headers_ms", "body_ms", "total_ms")},
                "child_stopped": process.poll() is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--loop", choices=("asyncio", "uvloop"))
    parser.add_argument("--http", choices=("h11", "httptools"))
    args = parser.parse_args()
    if args.fd is not None:
        return serve(args)
    if args.output is None:
        parser.error("--output is required for the diagnostic")
    args.python = args.python.expanduser().absolute()
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": "Synthetic two-message ASGI response, inherited TCP listener fd; no model/PII/service load.",
              "python": str(args.python), "configurations": []}
    with args.output.open("x") as stream:
        for loop in ("asyncio", "uvloop"):
            for protocol in ("h11", "httptools"):
                result = run_one(args, loop, protocol)
                report["configurations"].append(result)
                print(json.dumps({"loop": loop, "http": protocol, "median": result["reused_median_ms"],
                                  "server_socket": result["reused_samples"][-1]["server_socket"]}), flush=True)
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
