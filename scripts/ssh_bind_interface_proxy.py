#!/usr/bin/env python3
"""Connect SSH through a chosen local interface, with a default-route fallback.

OpenSSH's BindInterface selects a source address but does not bypass a local
transparent VPN route. SO_BINDTODEVICE pins the socket to the physical link.
This program is meant for SSH's ProxyCommand and keeps SSH encryption intact.
"""

import os
import selectors
import socket
import sys


def connect(interface: str, host: str, port: int) -> socket.socket:
    last_error = None
    for device in (interface, None):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.settimeout(3)
        try:
            if device is not None:
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                                      device.encode() + b"\0")
            connection.connect((host, port))
            connection.settimeout(None)
            if device is None:
                print("Direct interface unavailable; using default route", file=sys.stderr)
            return connection
        except OSError as error:
            last_error = error
            connection.close()
    raise OSError(f"Could not connect to {host}:{port}") from last_error


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def forward(connection: socket.socket) -> None:
    selector = selectors.DefaultSelector()
    selector.register(sys.stdin.buffer, selectors.EVENT_READ, "stdin")
    selector.register(connection, selectors.EVENT_READ, "socket")
    while selector.get_map():
        for key, _ in selector.select():
            if key.data == "stdin":
                data = os.read(sys.stdin.fileno(), 65536)
                if data:
                    connection.sendall(data)
                else:
                    selector.unregister(sys.stdin.buffer)
                    connection.shutdown(socket.SHUT_WR)
            else:
                data = connection.recv(65536)
                if data:
                    write_all(sys.stdout.fileno(), data)
                else:
                    selector.unregister(connection)
                    return


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: ssh_bind_interface_proxy.py INTERFACE HOST PORT")
    interface, host, raw_port = sys.argv[1:]
    with connect(interface, host, int(raw_port)) as connection:
        forward(connection)


if __name__ == "__main__":
    main()
