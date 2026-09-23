"""Process supervisor with one fresh metrics directory shared by its workers."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env", override=False)
    workers = int(os.getenv("SEIF_WORKERS", "1"))
    if workers < 1:
        raise SystemExit("SEIF_WORKERS must be positive")
    if workers > 1 and not (os.getenv("SEIF_REDIS_URL") or os.getenv("SEIF_SENTINELS")):
        raise SystemExit("Multiple workers require Redis or Sentinel and a shared SEIF_MASTER_KEY")
    directory = tempfile.mkdtemp(prefix="seif-prometheus-")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = directory
    # Must happen after the environment is set; worker processes inherit it.
    import uvicorn

    try:
        uvicorn.run(
            "seif.app:create_app",
            factory=True,
            host=os.getenv("SEIF_HOST", "127.0.0.1"),
            port=int(os.getenv("SEIF_PORT", "8765")),
            workers=workers,
            access_log=False,
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
