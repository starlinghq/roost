"""Load generation: closed loop (fixed concurrency) and open loop (Poisson session arrivals at a fixed rate)."""

from __future__ import annotations

import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .client import Endpoint, send
from .metrics import RequestRecord
from .workloads import Session


@dataclass
class LevelResult:
    """Records of one load level, and how far the client fell behind its open-loop schedule."""

    mode: str
    level: float
    records: list[RequestRecord] = field(default_factory=list)
    max_client_lag_s: float = 0.0


def run_session(ep: Endpoint, session: Session) -> list[RequestRecord]:
    """Run a session's steps: in order (carrying chat history when asked) or all at once when ``parallel``."""

    def one(i: int, step, history: list[dict] | None) -> tuple[RequestRecord, str]:
        body = dict(step.body)
        if step.api == "chat":
            body["messages"] = (history or []) + step.messages
        elif step.api == "completions":
            body["prompt"] = step.text
        else:
            body["input"] = step.text
        return send(ep, step.api, body, f"{session.session_id}.{i}", session.session_id)

    if session.parallel and len(session.steps) > 1:
        with ThreadPoolExecutor(max_workers=len(session.steps)) as pool:
            return [r for r, _ in pool.map(lambda a: one(a[0], a[1], None), enumerate(session.steps))]
    records: list[RequestRecord] = []
    history: list[dict] = []
    for i, step in enumerate(session.steps):
        record, reply = one(i, step, history if session.carry_history else None)
        records.append(record)
        if not record.ok:
            break  # later turns depend on this reply
        if session.carry_history:
            history = history + step.messages + [{"role": "assistant", "content": reply}]
    return records


def run_concurrency(ep: Endpoint, sessions: list[Session], concurrency: int) -> LevelResult:
    """Closed loop: ``concurrency`` workers each take the next session as soon as their previous one finishes."""
    todo: queue.SimpleQueue[Session] = queue.SimpleQueue()
    for s in sessions:
        todo.put(s)
    result = LevelResult("concurrency", concurrency)
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                s = todo.get_nowait()
            except queue.Empty:
                return
            recs = run_session(ep, s)
            with lock:
                result.records.extend(recs)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return result


def arrival_offsets(n: int, rate: float, seed: int) -> list[float]:
    """Poisson arrivals: cumulative exponential gaps with mean 1 / rate seconds, from a seeded generator."""
    rng = random.Random(f"arrivals:{seed}:{rate}")
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def run_rate(
    ep: Endpoint, sessions: list[Session], rate: float, seed: int = 0, max_inflight: int = 1024
) -> LevelResult:
    """Open loop: sessions arrive as a Poisson process at ``rate`` per second. Lag past the schedule is reported."""
    result = LevelResult("rate", rate)
    lock = threading.Lock()
    offsets = arrival_offsets(len(sessions), rate, seed)

    def job(s: Session, due: float) -> None:
        lag = time.perf_counter() - due
        recs = run_session(ep, s)
        with lock:
            result.records.extend(recs)
            result.max_client_lag_s = max(result.max_client_lag_s, lag)

    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        t0 = time.perf_counter()
        for s, off in zip(sessions, offsets, strict=True):
            due = t0 + off
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pool.submit(job, s, due)
    return result
