"""CPU threading probe; not an HTTP benchmark and not a production capacity claim."""
import importlib
import json
import os
import platform
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seif.detector import detect

payloads = (
    "Клиент Иванов Иван Иванович, паспорт 4509 123456; email ivan@example.net.",
    "Телефон: +7 (999) 123-45-67. Дата рождения: 12 марта 1990 года.",
    "Карта: 4111 1111 1111 1111, CVV: 123, ПИН: 4321.",
    "Адрес проживания: г. Москва, ул. Тестовая, дом 17, квартира 8.",
)
imports = {}
for module in ("fastapi", "pydantic_core", "cryptography.hazmat.primitives.ciphers.aead", "regex", "yaml", "redis", "uvloop", "httptools"):
    importlib.import_module(module)
    imports[module] = getattr(sys, "_is_gil_enabled", lambda: True)()


def batch(count):
    for index in range(count):
        detect(payloads[index % len(payloads)])


batch(100)
results = []
for threads in (1, 4):
    total = 20_000
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        list(executor.map(batch, [total // threads] * threads))
    elapsed = time.perf_counter() - start
    results.append({"threads": threads, "calls": total, "seconds": round(elapsed, 3), "calls_per_second": round(total / elapsed)})
print(json.dumps({"python": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
                  "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
                  "gil_enabled_after_imports": imports, "detector_thread_benchmark": results,
                  "scope": "Single synthetic CPU microbenchmark; no HTTP, Redis or network latency"}, ensure_ascii=False, indent=2))
