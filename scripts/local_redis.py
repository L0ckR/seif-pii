#!/usr/bin/env python3
"""Manage one isolated local 3-Redis/3-Sentinel demonstration topology.

Requires the project's redis-py dependency and a separately installed Redis 7+
server. No server binary or generated credentials belong in the source archive.
A fresh start is deliberately one-shot: existing Sentinel/AOF state must not be
silently bootstrapped back to the original primary after a promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
REDIS_PORTS = (6386, 6387, 6388)
SENTINEL_PORTS = (26386, 26387, 26388)
MASTER = "seif-master"


def write_private(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(text)


def write_state(directory: Path, state: dict) -> None:
    temporary = directory / "state.next.json"
    if temporary.exists():
        temporary.unlink()
    write_private(temporary, json.dumps(state, indent=2) + "\n")
    temporary.replace(directory / "state.json")


def process_start(pid: int) -> str:
    # Linux field 22; splitting after the final ')' handles spaces in comm.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def process_owned(node: dict, executable: str) -> bool:
    try:
        base = Path(f"/proc/{node['pid']}")
        return (
            process_start(node["pid"]) == node["start_ticks"]
            and str((base / "exe").resolve(strict=True)) == executable
            and str((base / "cwd").resolve(strict=True)) == node["directory"]
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return False


def load_state(directory: Path) -> tuple[dict, dict]:
    state = json.loads((directory / "state.json").read_text())
    credentials = json.loads((directory / "credentials.json").read_text())
    return state, credentials


def client(port: int, password: str) -> Redis:
    return Redis(
        host="127.0.0.1",
        port=port,
        password=password,
        socket_timeout=0.4,
        socket_connect_timeout=0.4,
        decode_responses=True,
        retry=Retry(NoBackoff(), 0),
    )


def _node_snapshot(node, executable, credentials):
    vote = None
    owned = process_owned(node, executable)
    item = {"port": node["port"], "owned_process_running": owned}
    password = credentials["redis_password" if node["kind"] == "redis" else "sentinel_password"]
    try:
        if not owned:
            raise ConnectionError("Owned process unavailable")
        with client(node["port"], password) as connection:
            if node["kind"] == "redis":
                replication = connection.info("replication")
                item.update(role=replication["role"], connected_replicas=replication.get("connected_slaves", 0))
                if replication["role"] == "slave":
                    item["primary_port"] = replication["master_port"]
                    item["replication_link"] = replication["master_link_status"]
            else:
                info = connection.sentinel_master(MASTER)
                item.update(
                    primary_host=info["ip"],
                    primary_port=info["port"],
                    known_sentinels=info["num-other-sentinels"] + 1,
                )
                vote = (info["ip"], int(info["port"]))
    except (RedisError, OSError):
        item["available"] = False
    else:
        item["available"] = True
    return item, vote


def snapshot(directory: Path) -> dict:
    state, credentials = load_state(directory)
    result = {"ready": False, "same_host_only": True, "redis": [], "sentinels": []}
    primary_votes = []
    for node in state["nodes"]:
        item, vote = _node_snapshot(node, state["executable"], credentials)
        if vote is not None:
            primary_votes.append(vote)
        result["redis" if node["kind"] == "redis" else "sentinels"].append(item)
    agreed = len(primary_votes) == 3 and len(set(primary_votes)) == 1
    if agreed:
        primary_port = primary_votes[0][1]
        primaries = [node for node in result["redis"] if node.get("role") == "master"]
        replicas = [
            node
            for node in result["redis"]
            if node.get("role") == "slave"
            and node.get("primary_port") == primary_port
            and node.get("replication_link") == "up"
        ]
        result["primary_port"] = primary_port
        result["ready"] = bool(
            len(primaries) == 1
            and primaries[0]["port"] == primary_port
            and primaries[0]["connected_replicas"] >= 1
            and len(replicas) == 2
            and all(node.get("known_sentinels") == 3 for node in result["sentinels"])
        )
    return result


def stop(directory: Path) -> dict:
    state, _ = load_state(directory)
    stopped, not_owned = [], []
    # Close discovery first so an intentional stop cannot trigger a promotion.
    for node in sorted(state["nodes"], key=lambda item: item["kind"] != "sentinel"):
        if process_owned(node, state["executable"]):
            os.kill(node["pid"], signal.SIGTERM)
            stopped.append(node["port"])
        else:
            not_owned.append(node["port"])
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if not any(process_owned(node, state["executable"]) for node in state["nodes"]):
            break
        time.sleep(0.1)
    remaining = [node["port"] for node in state["nodes"] if process_owned(node, state["executable"])]
    state["stopped"] = not remaining
    write_state(directory, state)
    return {
        "stopped_ports": stopped,
        "absent_or_identity_mismatch_ports": not_owned,
        "still_running_ports": remaining,
        "persistent_data_kept": True,
    }


def _check_available_ports():
    # Refuse occupied ports before creating credentials or touching any service.
    reserved = []
    try:
        for port in REDIS_PORTS + SENTINEL_PORTS:
            sock = socket.socket()
            sock.bind(("127.0.0.1", port))
            reserved.append(sock)
    except OSError:
        raise ValueError("A required local topology port is already occupied") from None
    finally:
        for sock in reserved:
            sock.close()


def _start_node(directory, executable, environment, credentials, identity):
    kind, index, port = identity
    node_directory = directory / f"{kind}-{port}"
    node_directory.mkdir(mode=0o700)
    config = node_directory / f"{kind}.conf"
    common = f"bind 127.0.0.1\nprotected-mode yes\nport {port}\ndir {node_directory}\ndaemonize no\n"
    if kind == "redis":
        body = (
            f"requirepass {credentials['redis_password']}\nmasterauth {credentials['redis_password']}\n"
            'appendonly yes\nappendfsync everysec\nsave ""\nmaxmemory 1gb\nmaxmemory-policy noeviction\n'
            "min-replicas-to-write 1\nmin-replicas-max-lag 5\n"
            + (f"replicaof 127.0.0.1 {REDIS_PORTS[0]}\n" if index else "")
        )
    else:
        body = (
            f"requirepass {credentials['sentinel_password']}\n"
            f"sentinel sentinel-pass {credentials['sentinel_password']}\n"
            f"sentinel monitor {MASTER} 127.0.0.1 {REDIS_PORTS[0]} 2\n"
            f"sentinel auth-pass {MASTER} {credentials['redis_password']}\n"
            f"sentinel down-after-milliseconds {MASTER} 5000\n"
            f"sentinel failover-timeout {MASTER} 15000\nsentinel parallel-syncs {MASTER} 1\n"
        )
    write_private(config, common + body)
    log_path = node_directory / "server.log"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    command = [str(executable), str(config)] + (["--sentinel"] if kind == "sentinel" else [])
    with os.fdopen(descriptor, "wb") as log:
        process = subprocess.Popen(
            command,
            cwd=node_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            umask=0o077,
        )
    node = {
        "kind": kind,
        "port": port,
        "pid": process.pid,
        "directory": str(node_directory),
        "start_ticks": process_start(process.pid),
    }
    return node


def start(directory: Path, executable: Path, library_dir: Path | None) -> dict:
    if directory.exists():
        raise ValueError(
            "Directory already exists. Use status/stop; existing AOF and Sentinel state must not be re-bootstrapped."
        )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Provide an executable Redis server with --redis-server")
    _check_available_ports()
    directory.mkdir(mode=0o700, parents=True)
    credentials = {"redis_password": secrets.token_urlsafe(32), "sentinel_password": secrets.token_urlsafe(32)}
    write_private(directory / "credentials.json", json.dumps(credentials, indent=2) + "\n")
    environment = os.environ.copy()
    if library_dir:
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = str(library_dir) + (":" + previous if previous else "")
    state = {"version": 1, "executable": str(executable.resolve()), "nodes": [], "stopped": False}
    write_state(directory, state)
    try:
        for kind, ports in (("redis", REDIS_PORTS), ("sentinel", SENTINEL_PORTS)):
            for index, port in enumerate(ports):
                node = _start_node(directory, executable, environment, credentials, (kind, index, port))
                state["nodes"].append(node)
                write_state(directory, state)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = snapshot(directory)
            if result["ready"]:
                return result
            time.sleep(0.25)
        raise RuntimeError("The isolated topology did not become ready; inspect its private server logs")
    except BaseException:
        stop(directory)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--directory", type=Path, default=ROOT / "output" / "redis-live")
    parser.add_argument(
        "--redis-server", type=Path, default=Path(shutil.which("redis-server") or "/nonexistent/redis-server")
    )
    parser.add_argument(
        "--library-dir", type=Path, help="Optional directory for separately installed Redis shared libraries"
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    try:
        if args.action == "start":
            result = start(
                directory, args.redis_server.absolute(), args.library_dir.resolve() if args.library_dir else None
            )
        elif args.action == "stop":
            result = stop(directory)
        else:
            result = snapshot(directory)
    except json.JSONDecodeError:
        print("Local topology state is unavailable or invalid; no unrelated process was changed.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, KeyError):
        print("Local topology state is unavailable or invalid; no unrelated process was changed.", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action != "stop":
        print("Same-host demonstration only: this does not provide node-level high availability.")
    return 0 if args.action == "stop" or result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
