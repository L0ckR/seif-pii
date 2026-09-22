# Remaining-findings runtime validation

The snapshot from `016a9c6` and the candidate were evaluated with sequential in-process HTTP requests through real FastAPI routing: 60 warmup and 1,200 measured requests per path, repeated twice per revision. `latency.py` reproduces the run; raw summaries and source hashes are in before/after JSON files.

Four no-await callbacks now follow the ordinary synchronous FastAPI/Starlette contract. `/v1/types` and NER error responses incur the expected threadpool hop: median overhead ranged 19–236 μs across paths/runs. Successful `/process` remains asynchronous; its observed local median changed +3.1% and +7.2%, while the unchanged invalid-request control changed −0.4% and −8.9%. These short sequential measurements are sensitive to system activity and do not establish a production RPS effect. No public endpoint, Redis, or NER model was involved.

TaskGroup task creation is an explicit loop preserving input order and sibling cancellation. `test_ner_batch.py` verifies four simultaneous chunks, failure cancellation before return, and the direct single-chunk path.
