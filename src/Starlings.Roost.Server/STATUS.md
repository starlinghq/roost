# R-037 status: partial (CPU-tested; no GPU evidence)

`Starlings.Roost.Server` is an ASP.NET Core minimal API host over `Starlings.Generation.Roost`'s `RoostChatClient`, and it builds
the `roost` executable. The routes, the error body, the option checks and loopback-only listening are built and tested on the
CPU path only: `tests/Starlings.Roost.Server.Tests` drives the real client, the C8 rings and the eager worker's plan reader and
result writer through an in-process scripted worker, behind ASP.NET Core's in-memory `TestServer`, and starts real Kestrel once
to check the loopback default. No request has reached a GPU through the server, so every route is `untested`, never
`supported` or `experimental`.

## Routes (PRD V1 set for R-037)

| Route | Status | Notes |
| --- | --- | --- |
| `POST /v1/chat/completions` | untested | Whole response, or SSE (`chat.completion.chunk`: a role chunk, content deltas, a finish chunk, a usage chunk when `stream_options.include_usage` is true, then `data: [DONE]`). |
| `POST /v1/completions` | untested | Raw prompt, tokenised as given with no chat template (new `RoostChatClient.GetStreamingCompletionAsync`, capability `raw_completion`, `untested`). `max_tokens` defaults to 16, as in OpenAI's API and vLLM. Whole or SSE, as above, without the role chunk. |
| `GET /v1/models` | untested | The one served model, `owned_by` `roost`. |

## Behaviour

- **One error body.** Every failure answers `{"error":{"message","type","param","code"}}` with `type` `roost_error`. Request
  and generation errors use `RoostError.WriteHttpBody` (C6 code names and HTTP statuses: `invalid_option` and
  `prompt_too_long` are 400, `internal` 500, and so on); routing failures use the same shape with `not_found` (404) or
  `method_not_allowed` (405). A stream that fails after its headers were sent ends with a `data:` event carrying the error body,
  then `data: [DONE]`. A streamed request refused on its first step (a prompt longer than the context) still gets its HTTP
  status, because the first update is awaited before the headers are sent.
- **Options.** Request bodies are read field by field (`OpenAiRequestReader`). Every field is mapped, accepted at a value that
  changes nothing (`n` 1, `logprobs` false, `tool_choice` `"none"`, `response_format` `{"type":"text"}`, `store` false, empty
  `logit_bias`, `best_of` 1, `echo` false, empty `suffix`, `user`), or refused with a 400 whose `param` names it; an unknown
  field is refused the same way. JSON `null` means unset. Sampling, penalties, `top_k`, `stop`, `max_tokens` and `model` are
  passed to `RoostChatClient`, whose refusals (greedy decoding only in M0) name the same OpenAI parameter. Message fields are
  named by path, for example `messages[0].content[0].type`. An absent `temperature` means greedy decoding, not OpenAI's 1.0.
- **Loopback by default.** `127.0.0.1:8000`. A non-loopback `--host` (or `RoostServerOptions.Host`) is refused unless
  `--allow-non-loopback` (`AllowNonLoopback`) is set: the server has no authentication. Kestrel gets only that one listener.
- **Cancellation.** The request's abort token is the generation's token: when the client disconnects, the worker drops the
  sequence from its next step and the next request runs at once (tested over a streaming request).
- **Native AOT.** `PublishAot` with the slim builder, `RequestDelegate` handlers and a source-generated `JsonSerializerContext`
  for every response body; requests are read with `JsonDocument`. No reflection-based serialization. A local
  `dotnet publish -c Release -r linux-x64` on 2026-11-01 (no GPU on the machine) produced an 11.2 MB `roost` with zero trim or
  AOT warnings, and the binary printed its usage and refused a missing tokenizer; that is a build check, not serving evidence.
- **`roost serve`.** `--model qwen3.5-0.8b --weights --weights-sha256 --tokenizer [--kernels] [--worker] [--manifest]
  [--allow-untested-sm] [--host] [--port] [--allow-non-loopback]`; unknown flags are refused by name. It starts the packaged
  worker in its own process (decision 21) with the capacities of the R-036 GPU run (32 attention blocks of 16 tokens, 64
  tokens and one sequence per step, so a 512-token context). Parsing and the refusal paths are tested; serving through it is
  not.

## Still open (R-037 is not done)

- **GPU end-to-end evidence.** No run of `roost serve` with Qwen3.5-0.8B on the 3090 or 4060 Ti through `tools/gpu-lease`;
  until one is committed with its log, every route stays `untested`. The S1 RSS and SSE figures (docs/m0/roost-s1-aot-cuda.md)
  are from the spike, not from this server.
- **OpenAI client libraries** against the server: the compatibility suite is R-040.
- **Metrics, traces and log hygiene** (`/metrics`, one Activity per request): R-038. The server logs no prompt or output text;
  the only log message is an unexpected exception.
- **Responses, Anthropic Messages, tokenize and detokenize, health and readiness routes**: R-074. `/v1/models` is here because
  the task scoped it in; R-074 lists it too.
- **Tool calling, structured output, sampling, log probabilities, `n` > 1, multimodal input**: refused with a 400 naming the
  option until R-053, R-051 and R-042 land.
- **vLLM's CLI flags and environment variables**: R-075.
- **Concurrency**: M0 runs one sequence per step, so concurrent requests take turns (`continuous_batching` `unsupported`).
