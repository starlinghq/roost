"""Client, runner and CLI against an in-process fake OpenAI-compatible server on loopback (no GPU, no network)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from roost_bench import cli
from roost_bench.client import Endpoint, send
from roost_bench.runner import arrival_offsets, run_concurrency, run_rate, run_session
from roost_bench.workloads import Session, Step

FIRST_DELAY = 0.05
TOKEN_GAP = 0.01
MAX_STREAM = 8  # the fake stops early so long workloads stay fast; usage reports what it sent


class FakeServer(BaseHTTPRequestHandler):
    """Streams ``min(max_tokens, MAX_STREAM)`` one-word chunks: the first after FIRST_DELAY, then every TOKEN_GAP."""

    requests: list[dict] = []
    lock = threading.Lock()

    def log_message(self, *args):  # keep pytest output quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            FakeServer.requests.append({"path": self.path, "body": body})
        if body.get("model") == "refuse":
            self._json(400, {"error": {"message": "bad", "type": "roost_error", "param": "model", "code": "x"}})
            return
        if self.path.endswith("/embeddings"):
            self._json(200, {"data": [{"embedding": [0.0, 1.0]}], "usage": {"prompt_tokens": 3, "total_tokens": 3}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        n = min(int(body.get("max_tokens", 4)), MAX_STREAM)
        chat = self.path.endswith("/chat/completions")
        if chat:
            self._event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})  # no output: not a token
        time.sleep(FIRST_DELAY)
        for i in range(n):
            if i:
                time.sleep(TOKEN_GAP)
            word = f"w{i} "
            self._event({"choices": [{"index": 0, "delta": {"content": word}} if chat else {"index": 0, "text": word}]})
        if body.get("stream_options", {}).get("include_usage"):
            self._event({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": 7 + n}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _event(self, doc: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(doc).encode() + b"\n\n")
        self.wfile.flush()

    def _json(self, status: int, doc: dict) -> None:
        data = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    FakeServer.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


def test_streamed_chat_records_ttft_chunks_and_usage(server):
    ep = Endpoint(server, "m")
    rec, reply = send(ep, "chat", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}, "r1")
    assert rec.ok and rec.error is None
    assert len(rec.chunk_times) == 5  # the role-only chunk is not counted
    assert rec.output_tokens == 5 and rec.prompt_tokens == 7
    assert rec.ttft >= FIRST_DELAY
    assert all(g >= TOKEN_GAP * 0.5 for g in rec.itl_samples())
    assert reply == "w0 w1 w2 w3 w4 "
    sent = FakeServer.requests[0]["body"]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True} and sent["model"] == "m"
    assert "ignore_eos" not in sent  # backend extensions only when asked


def test_completions_extra_body_and_no_usage(server):
    ep = Endpoint(server, "m", include_usage=False, extra_body={"ignore_eos": True})
    rec, _ = send(ep, "completions", {"prompt": "x", "max_tokens": 3}, "r1")
    assert rec.ok and len(rec.chunk_times) == 3 and rec.output_tokens is None and rec.tokens_out() == 3
    sent = FakeServer.requests[0]["body"]
    assert sent["ignore_eos"] is True and "stream_options" not in sent


def test_embeddings_and_http_errors(server):
    rec, _ = send(Endpoint(server, "m", embed_model="e"), "embeddings", {"input": "x"}, "r1")
    assert rec.ok and rec.kind == "embed" and rec.prompt_tokens == 3 and rec.ttft is None
    assert FakeServer.requests[0]["body"]["model"] == "e"
    bad, _ = send(Endpoint(server, "refuse"), "chat", {"messages": []}, "r2")
    assert not bad.ok and bad.error.startswith("HTTP 400")


def test_connection_refused_is_a_failed_record():
    rec, _ = send(Endpoint("http://127.0.0.1:9/v1", "m", timeout_s=2), "chat", {"messages": []}, "r")
    assert not rec.ok and rec.error


def test_multi_turn_session_sends_history_with_replies(server):
    steps = [Step("chat", {"max_tokens": 2}, messages=[{"role": "user", "content": f"q{i}"}]) for i in range(3)]
    recs = run_session(Endpoint(server, "m"), Session("s", steps, carry_history=True))
    assert [r.ok for r in recs] == [True, True, True]
    last = FakeServer.requests[-1]["body"]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "user", "assistant", "user"]
    assert last[1]["content"] == "w0 w1 "


def test_parallel_session_overlaps_requests(server):
    steps = [Step("chat", {"max_tokens": 1}, messages=[{"role": "user", "content": "q"}]) for _ in range(6)]
    t0 = time.perf_counter()
    recs = run_session(Endpoint(server, "m"), Session("s", steps, parallel=True))
    assert len(recs) == 6 and all(r.ok for r in recs)
    assert time.perf_counter() - t0 < 6 * FIRST_DELAY


def test_closed_and_open_loop_levels(server):
    sessions = [Session(f"s{i}", [Step("chat", {"max_tokens": 2}, messages=[])]) for i in range(8)]
    ep = Endpoint(server, "m")
    res = run_concurrency(ep, sessions, 4)
    assert len(res.records) == 8 and all(r.ok for r in res.records)
    res = run_rate(ep, sessions, rate=200.0, seed=3)
    assert len(res.records) == 8 and res.mode == "rate"


def test_arrival_offsets_are_seeded_and_increasing():
    a = arrival_offsets(100, 10.0, 1)
    assert a == arrival_offsets(100, 10.0, 1) and a != arrival_offsets(100, 10.0, 2)
    assert all(x < y for x, y in zip(a, a[1:], strict=False))
    assert 5 < a[-1] < 20  # mean gap 0.1 s


def test_cli_run_and_summarize_round_trip(server, tmp_path: Path, capsys):
    root = Path(__file__).resolve().parents[3] / "benchmarks"
    out = tmp_path / "run"
    rc = cli.main(
        [
            "run", "--workload", str(root / "workloads/w7-single-user.toml"), "--base-url", server, "--model", "m",
            "--label", "fake", "--out", str(out), "--slo", str(root / "slo.toml"), "--max-requests", "3",
            "--extra-body", '{"ignore_eos": true}',
        ]
    )  # fmt: skip
    assert rc == 0
    report = json.loads((out / "summary.json").read_text())
    assert report["header"]["format"] == "roost-bench-trace/0"
    assert report["header"]["label"] == "fake"
    level = report["levels"][0]
    assert level["level"] == 1 and level["completed"] == 3 and level["slo"]["status"] == "proposed"
    assert level["output_tokens"] == 3 * MAX_STREAM
    assert level["itl_s"]["p50"] >= TOKEN_GAP * 0.5
    capsys.readouterr()
    assert cli.main(["summarize", str(out / "trace.jsonl"), "--slo", str(root / "slo.toml")]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["levels"][0]["output_tokens"] == level["output_tokens"]
    assert again["levels"][0]["ttft_s"] == level["ttft_s"]


def test_cli_validate(capsys):
    root = Path(__file__).resolve().parents[3] / "benchmarks"
    assert cli.main(["validate", *map(str, sorted((root / "workloads").glob("*.toml")))]) == 0
