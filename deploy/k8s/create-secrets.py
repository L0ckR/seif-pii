#!/usr/bin/env python3
"""Create one Kubernetes Secret from temporary 0600 files; print no credentials.

The namespace must already exist. Creation intentionally fails if the Secret
exists: rerunning this command must not rotate the key for stored ciphertext.
"""
from __future__ import annotations

import argparse
import base64
import os
import secrets
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="seif")
    args = parser.parse_args()
    os.umask(0o077)
    values = {
        "master-key": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
        "api-key": secrets.token_urlsafe(48),
        "redis-password": secrets.token_hex(32),
    }
    with tempfile.TemporaryDirectory(prefix="seif-k8s-secrets-") as temporary:
        command = ["kubectl", "--namespace", args.namespace, "create", "secret", "generic", "seif-secrets"]
        for name, value in values.items():
            path = Path(temporary) / name
            path.write_text(value, encoding="ascii")
            command.append(f"--from-file={name}={path}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
