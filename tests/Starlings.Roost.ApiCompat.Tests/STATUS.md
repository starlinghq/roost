# R-040 status: partial (CPU-tested; no GPU evidence)

`tests/Starlings.Roost.ApiCompat.Tests` drives `Starlings.Roost.Server` with the official OpenAI client libraries. The
server runs on real Kestrel (loopback, port 0, the listener `roost serve` uses) over `RoostChatClient` on the in-process
transport with the scripted worker of the server tests (`ScriptedExecutor`: the eager worker's plan reader and result writer,
no model). Every request and response crosses HTTP as a client sends it, so the suite checks that the clients serialise
requests Roost accepts and parse what Roost answers. It does not check model output: no request in it reaches a GPU, so
nothing here changes a route's `untested` status.

## Clients

| Client | Version | How it runs | Covered |
| --- | --- | --- | --- |
| OpenAI .NET (`OpenAI` NuGet package) | 2.14.0, exact pin in `Directory.Packages.props`, this project only | Every `dotnet test --solution Roost.slnx` (CI) | Chat completions whole and streamed (text, developer/system and assistant turns, multi-part user text, `max_completion_tokens`, `stop`, `seed`, `temperature` 0, `user`, usage on the last stream chunk, finish reasons `stop`, `length` and `tool_calls`); tool declarations, `tool_choice` `auto` and `none`, `parallel_tool_calls` true, parsed `tool_calls` whole and streamed, a call and its `tool` result sent back; `reasoning_content` (read from the raw body; the client has no field for it) with `chat_template_kwargs` added through the client's `JsonPatch`; the Responses API (create whole, streamed and in the background, retrieve, cancel, `max_output_tokens` ending `incomplete`, `store` false); `GET /v1/models`; `POST /v1/embeddings` through `EmbeddingClient` (one input, a batch, `dimensions`, `user`, usage) over a fake embedding model, and `profile_unsupported` when none is configured (R-052); error bodies (`prompt_too_long`, `not_found`, `invalid_request`, a body that is not an object, an unknown field). |
| openai (Python package) | 3.35.1, pinned in `tests/tools/api_compat/openai_compat.py` | Opt-in: `ROOST_API_COMPAT_PYTHON=1 dotnet test --project tests/Starlings.Roost.ApiCompat.Tests` starts the scripted server and runs the script with `uv run --no-project --with openai==3.35.1`. Against a real server: the command in the script's docstring. Not in CI: it needs `uv` and the network to fetch the package. | `GET /v1/models`; chat completions and legacy completions whole and streamed (the streamed text must equal the whole reply); the Responses API create, retrieve and stream; refusals of chat, completion and Responses options by name. Run on 2026-11-02 against the scripted server: every check passed. |

The OpenAI .NET client 2.14.0 has no legacy completions client, so `POST /v1/completions` is covered by the Python script
only.

## Unsupported options

Every option the official .NET client can set that Roost does not support is asserted through the client as a 400 with
`code` `invalid_option` and `param` naming it, whole and streamed, before any step reaches the worker:
`temperature` other than 0, `top_p`, `frequency_penalty`, `presence_penalty`, `logprobs`, `top_logprobs`, `logit_bias`,
`store` true, `reasoning_effort`, `response_format` (JSON object and JSON schema), `modalities` with audio, `metadata`,
`service_tier`, `safety_identifier`, `web_search_options`, `prediction`, and through `JsonPatch` `n` 2, `top_k` and
`chat_template_kwargs.add_generation_prompt`; `tool_choice` `required` or a named function, `parallel_tool_calls` false and
a strict tool (`tools[0].function.strict`); image and audio content parts and a participant name, named by path; another
model. On `/v1/responses`: `temperature`, `top_p`, `tools`, `tool_choice`, `reasoning`, `previous_response_id`,
`text.format`, `truncation` `auto`, `include`, `top_logprobs`, `max_tool_calls`, `service_tier`, `background` without
`store`, and an image input part.

Accepted at a value that changes nothing, and covered by the success tests: `seed`, `temperature` 0, `top_p` 1 on
`/v1/responses`, `user`.

## Found by the suite

- `GET /v1/models/{id}` (the client's `GetModelAsync`) is not routed: a 404 `not_found` with the one error body. The
  list route serves the model.

## Still open

- **GPU evidence.** No run of either client against `roost serve` on a GPU; the script is ready for one (see its
  docstring). Until a run is committed with its log, every route stays `untested`.
- **Anthropic client library** against `/v1/messages`: not in this suite (the acceptance criterion names the OpenAI
  libraries). The server tests cover Anthropic's bodies over raw HTTP.
- **vLLM's OpenAI-compatibility test suite** against Roost is R-075.
