"""A minimal OpenAI-compatible streaming client on the standard library (``http.client``, no proxies, no retries).

It records the arrival time of every output-bearing SSE chunk on ``time.perf_counter`` and the server's ``usage``
counts. It speaks ``/v1/chat/completions``, ``/v1/completions`` and ``/v1/embeddings`` as Roost, vLLM, SGLang and
llama.cpp's server expose them. Backend-specific fields (``ignore_eos``, ``min_tokens``) are never sent unless the
caller passes them in ``extra_body``: Roost's server refuses unknown fields.
"""

from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from .metrics import RequestRecord

ENDPOINTS = {"chat": "/chat/completions", "completions": "/completions", "embeddings": "/embeddings"}


@dataclass(frozen=True)
class Endpoint:
    """Where and how to send requests. ``base_url`` ends at the API root, for example ``http://127.0.0.1:8000/v1``."""

    base_url: str
    model: str
    api_key: str | None = None
    embed_model: str | None = None
    timeout_s: float = 600.0
    include_usage: bool = True
    extra_body: dict | None = None


def _connect(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parts = urlsplit(url)
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"unsupported base URL {url!r}")
    return cls(parts.hostname, parts.port, timeout=timeout), parts.path.rstrip("/")


def _chunk_has_output(chunk: dict) -> bool:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True
        if choice.get("text") or choice.get("logprobs"):
            return True
    return False


def _chunk_text(chunk: dict) -> tuple[str, list]:
    text, tools = "", []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        text += delta.get("content") or choice.get("text") or ""
        tools.extend(delta.get("tool_calls") or [])
    return text, tools


def send(
    ep: Endpoint, api: str, body: dict, request_id: str, session_id: str | None = None
) -> tuple[RequestRecord, str]:
    """Send one request; returns the record and the reply text (for multi-turn history). Never raises on HTTP errors."""
    payload = dict(body)
    payload["model"] = ep.embed_model if api == "embeddings" and ep.embed_model else ep.model
    stream = api != "embeddings"
    if stream:
        payload["stream"] = True
        if ep.include_usage:
            payload["stream_options"] = {"include_usage": True}
        if ep.extra_body:
            payload.update(ep.extra_body)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}
    if ep.api_key:
        headers["Authorization"] = f"Bearer {ep.api_key}"
    data = json.dumps(payload).encode()
    kind = "embed" if api == "embeddings" else "generate"
    record = RequestRecord(request_id=request_id, kind=kind, start=0.0, end=0.0, ok=False, session_id=session_id)
    reply = ""
    conn, root = _connect(ep.base_url, ep.timeout_s)
    record.start = time.perf_counter()
    try:
        conn.request("POST", root + ENDPOINTS[api], body=data, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            record.error = f"HTTP {resp.status}: {resp.read(2048).decode(errors='replace')}"
            return record, reply
        if not stream:
            doc = json.loads(resp.read())
            record.prompt_tokens = (doc.get("usage") or {}).get("prompt_tokens")
            record.ok = "data" in doc
            if not record.ok:
                record.error = f"no data in embeddings response: {str(doc)[:200]}"
            return record, reply
        done = False
        while not done:
            raw = resp.readline()
            if not raw:
                break
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data_part = line[5:].strip()
            if data_part == b"[DONE]":
                done = True
                break
            now = time.perf_counter()
            chunk = json.loads(data_part)
            if "error" in chunk:
                record.error = json.dumps(chunk["error"])[:2048]
                return record, reply
            if _chunk_has_output(chunk):
                record.chunk_times.append(now)
                text, _ = _chunk_text(chunk)
                reply += text
            usage = chunk.get("usage")
            if usage:
                record.output_tokens = usage.get("completion_tokens", record.output_tokens)
                record.prompt_tokens = usage.get("prompt_tokens", record.prompt_tokens)
        record.ok = done
        if not done:
            record.error = "stream ended without [DONE]"
        return record, reply
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
        record.error = f"{type(e).__name__}: {e}"
        return record, reply
    finally:
        record.end = time.perf_counter()
        conn.close()
