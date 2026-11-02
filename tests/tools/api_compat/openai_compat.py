"""R-040: drive a running Roost server with the official openai Python package and check the responses.

The checks hold for any completion the model gives (they compare shapes, the whole reply with the streamed one, and
usage), so the script runs against `roost serve` on a GPU as well as against the scripted worker of
tests/Starlings.Roost.ApiCompat.Tests (PythonClientCompatTests starts that server and runs this script when
ROOST_API_COMPAT_PYTHON=1). Every unsupported option must be a 400 whose error body names it.

Run (the openai version is pinned; the script needs nothing else):

    uv run --no-project --with openai==3.35.1 -- python -I tests/tools/api_compat/openai_compat.py \
        --base-url http://127.0.0.1:8000/v1

Exit code 0 when every check passes; otherwise the first failure is printed and the exit code is 1.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import openai

PINNED_OPENAI = "3.35.1"


class CheckFailed(Exception):
    """A check that did not hold."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def expect_refused(call: Callable[[], Any], param: str) -> None:
    """The call must fail with HTTP 400, Roost's error body, code invalid_option and the given param."""
    try:
        call()
    except openai.BadRequestError as e:
        body = e.body if isinstance(e.body, dict) else {}
        error = body.get("error", body)
        expect(error.get("type") == "roost_error", f"{param}: error type {error.get('type')!r}, expected 'roost_error'")
        expect(error.get("code") == "invalid_option", f"{param}: error code {error.get('code')!r}")
        expect(error.get("param") == param, f"{param}: error param {error.get('param')!r}")
        expect(bool(error.get("message")), f"{param}: empty error message")
        return
    raise CheckFailed(f"{param}: the request was accepted; expected a 400 naming it")


def check_models(client: openai.OpenAI) -> str:
    models = list(client.models.list())
    expect(len(models) == 1, f"/v1/models lists {len(models)} models, expected the one served")
    expect(models[0].owned_by == "roost", f"owned_by {models[0].owned_by!r}")
    return models[0].id


def check_chat(client: openai.OpenAI, model: str) -> None:
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    whole = client.chat.completions.create(model=model, messages=messages, max_tokens=16)
    expect(whole.object == "chat.completion", f"object {whole.object!r}")
    expect(whole.id.startswith("chatcmpl-"), f"id {whole.id!r}")
    expect(len(whole.choices) == 1, "one choice")
    choice = whole.choices[0]
    expect(choice.message.role == "assistant", f"role {choice.message.role!r}")
    expect(choice.finish_reason in ("stop", "length"), f"finish_reason {choice.finish_reason!r}")
    expect(isinstance(choice.message.content, str), "content is a string")
    usage = whole.usage
    expect(usage is not None and usage.total_tokens == usage.prompt_tokens + usage.completion_tokens, "usage adds up")

    text, finish, stream_usage = [], None, None
    for chunk in client.chat.completions.create(
        model=model, messages=messages, max_tokens=16, stream=True, stream_options={"include_usage": True}
    ):
        expect(chunk.object == "chat.completion.chunk", f"chunk object {chunk.object!r}")
        if chunk.usage is not None:
            stream_usage = chunk.usage
        for c in chunk.choices:
            text.append(c.delta.content or "")
            finish = c.finish_reason or finish
    # Greedy decoding: the stream is the whole reply, delta by delta.
    expect("".join(text) == choice.message.content, "the streamed text equals the whole reply")
    expect(finish == choice.finish_reason, f"stream finish_reason {finish!r}")
    expect(stream_usage is not None and stream_usage.completion_tokens == usage.completion_tokens, "stream usage")


def check_completions(client: openai.OpenAI, model: str) -> None:
    whole = client.completions.create(model=model, prompt="two plus two is", max_tokens=8)
    expect(whole.object == "text_completion", f"object {whole.object!r}")
    expect(whole.id.startswith("cmpl-"), f"id {whole.id!r}")
    expect(whole.choices[0].finish_reason in ("stop", "length"), f"finish_reason {whole.choices[0].finish_reason!r}")
    streamed = "".join(
        c.choices[0].text
        for c in client.completions.create(model=model, prompt="two plus two is", max_tokens=8, stream=True)
        if c.choices
    )
    expect(streamed == whole.choices[0].text, "the streamed completion equals the whole one")


def check_responses(client: openai.OpenAI, model: str) -> None:
    created = client.responses.create(model=model, input="Hi", instructions="Be brief.", max_output_tokens=16)
    expect(created.object == "response", f"object {created.object!r}")
    expect(created.id.startswith("resp_"), f"id {created.id!r}")
    expect(created.status in ("completed", "incomplete"), f"status {created.status!r}")
    stored = client.responses.retrieve(created.id)
    expect(stored.output_text == created.output_text, "the stored response has the same text")

    events = list(
        client.responses.create(model=model, input="Hi", instructions="Be brief.", max_output_tokens=16, stream=True)
    )
    expect(events[0].type == "response.created", f"first event {events[0].type!r}")
    expect(events[-1].type in ("response.completed", "response.incomplete"), f"last event {events[-1].type!r}")
    expect([e.sequence_number for e in events] == list(range(len(events))), "sequence numbers count from 0")
    deltas = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    expect(deltas == created.output_text, "the streamed text equals the whole response")


def check_refusals(client: openai.OpenAI, model: str) -> None:
    hi = [{"role": "user", "content": "Hi"}]
    chat = client.chat.completions.create
    expect_refused(lambda: chat(model=model, messages=hi, temperature=0.7), "temperature")
    expect_refused(lambda: chat(model=model, messages=hi, top_p=0.9), "top_p")
    expect_refused(lambda: chat(model=model, messages=hi, n=2), "n")
    expect_refused(lambda: chat(model=model, messages=hi, logprobs=True), "logprobs")
    expect_refused(lambda: chat(model=model, messages=hi, response_format={"type": "json_object"}), "response_format")
    expect_refused(lambda: chat(model=model, messages=hi, reasoning_effort="low"), "reasoning_effort")
    expect_refused(lambda: chat(model=model, messages=hi, store=True), "store")
    expect_refused(lambda: chat(model=model, messages=hi, extra_body={"frobnicate": 1}), "frobnicate")
    expect_refused(lambda: chat(model="another-model", messages=hi), "model")

    completion = client.completions.create
    expect_refused(lambda: completion(model=model, prompt="x", echo=True), "echo")
    expect_refused(lambda: completion(model=model, prompt="x", best_of=2), "best_of")
    expect_refused(lambda: completion(model=model, prompt="x", logprobs=1), "logprobs")
    expect_refused(lambda: completion(model=model, prompt="x", suffix="y"), "suffix")

    response = client.responses.create
    expect_refused(lambda: response(model=model, input="Hi", temperature=0.7), "temperature")
    expect_refused(lambda: response(model=model, input="Hi", previous_response_id="resp_1"), "previous_response_id")
    expect_refused(lambda: response(model=model, input="Hi", reasoning={"effort": "low"}), "reasoning")
    expect_refused(lambda: response(model=model, input="Hi", truncation="auto"), "truncation")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="the server's /v1 URL, for example http://127.0.0.1:8000/v1")
    args = parser.parse_args(argv)

    if openai.__version__ != PINNED_OPENAI:
        print(f"openai {openai.__version__} is not the pinned {PINNED_OPENAI}", file=sys.stderr)
        return 1

    client = openai.OpenAI(base_url=args.base_url, api_key="unused", max_retries=0, timeout=60)
    checks = [check_chat, check_completions, check_responses, check_refusals]
    try:
        model = check_models(client)
        for check in checks:
            check(client, model)
            print(f"ok {check.__name__}")
    except (CheckFailed, openai.APIError) as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"openai {openai.__version__}: every check passed against {args.base_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
