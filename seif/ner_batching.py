"""Bounded micro-batching with Future ownership through actual completion."""
from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import Future
from threading import Condition, Thread


class BatchExecutor:
    """One model thread; unrelated requests share a bounded inference batch.

    Cancellation removes only queued work. A running Future stays owned until
    the model returns, matching ThreadPoolExecutor's completion semantics.
    """

    def __init__(self, call_batch, *, batch_size, wait_ms, capacity):
        if (type(batch_size) is not int or not 2 <= batch_size <= 32
                or type(capacity) is not int or capacity < batch_size
                or type(wait_ms) not in (int, float) or not math.isfinite(wait_ms)
                or not 0 <= wait_ms <= 20):
            raise ValueError("Invalid bounded NER batch settings.")
        self.call_batch = call_batch
        self.batch_size, self.delay, self.capacity = batch_size, wait_ms / 1000, capacity
        self.condition = Condition()
        self.pending = deque()
        self.stopping = False
        self.cancel_queued = False
        self.worker = Thread(target=self._work, name="seif-ner-batch", daemon=False)
        self.worker.start()

    def submit(self, text):
        future = Future()
        with self.condition:
            if self.stopping or len(self.pending) >= self.capacity:
                raise RuntimeError("NER batch executor is unavailable.")
            self.pending.append((future, text))
            self.condition.notify()
        return future

    def _take_batch(self):
        with self.condition:
            while not self.pending and not self.stopping:
                self.condition.wait()
            if not self.pending:
                return []
            batch = [self.pending.popleft()]
            deadline = time.monotonic() + self.delay
            while len(batch) < self.batch_size:
                if self.pending:
                    batch.append(self.pending.popleft())
                elif self.stopping or time.monotonic() >= deadline:
                    break
                else:
                    self.condition.wait(deadline - time.monotonic())
            return batch

    def _work(self):
        while batch := self._take_batch():
            self._process_batch(batch)
            # Drop raw text references before waiting for another request.
            batch.clear()

    def _process_batch(self, batch):
        with self.condition:
            if self.cancel_queued:
                for future, _ in batch:
                    future.cancel()
                return
            active = [(future, text) for future, text in batch if future.set_running_or_notify_cancel()]
        if not active:
            return
        try:
            results = self.call_batch([text for _, text in active])
            if not isinstance(results, list) or len(results) != len(active):
                raise ValueError("Invalid NER batch response count.")
        except BaseException as error:
            # Transfer exceptions to their owners; never log input-bearing
            # third-party tracebacks from the model thread.
            for future, _ in active:
                future.set_exception(error)
        else:
            for (future, _), result in zip(active, results, strict=True):
                future.set_result(result)

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self.condition:
            self.stopping = True
            self.cancel_queued |= cancel_futures
            if cancel_futures:
                while self.pending:
                    future, _ = self.pending.popleft()
                    future.cancel()
            self.condition.notify_all()
        if wait:
            self.worker.join()
