# R-037, R-074, R-038 and R-053 status: partial (CPU-tested; no GPU evidence)

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

## Routes added by R-074

| Route | Status | Notes |
| --- | --- | --- |
| `POST /v1/responses` | untested | OpenAI Responses API. `input` is a string or message items with text parts (`input_text`; `output_text` on assistant items); `instructions` is the system message; `max_output_tokens`, `temperature`, `top_p` pass to the client. Whole, streamed (`response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`s, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, then `response.completed`, `response.incomplete` or `response.failed`; each event carries `sequence_number`; no `[DONE]`), or `background` (answers `queued` at once and generates detached). The output limit or the end of the context gives `status` `incomplete` with `incomplete_details.reason` `max_output_tokens`. |
| `GET /v1/responses/{id}` | untested | The stored response as it stands (a background one shows its text so far). Any query parameter (`stream`, `include`, `starting_after`) is refused by name. |
| `POST /v1/responses/{id}/cancel` | untested | Cancels a background response (the worker drops the sequence from its next step) and answers it as `cancelled`; cancelling a cancelled one answers it again. As in OpenAI's API, a response that was not created with `background` true, or one already finished, is a 400 with `code` `invalid_request`. |
| `POST /v1/messages` | untested | Anthropic Messages API over `RoostChatClient`: `system` (string or text blocks, joined with newlines) is the system message, text content blocks only, `max_tokens` required, `stop_sequences`, `temperature`, `top_p` and `top_k` passed on. Whole, or streamed with Anthropic's events (`message_start`, `content_block_start`, `ping`, `content_block_delta`s with `text_delta`, `content_block_stop`, `message_delta`, `message_stop`; a failure after the headers is an `error` event). Errors use Anthropic's body (below). |
| `POST /v1/messages/count_tokens` | untested | `{"input_tokens"}`: the conversation rendered with the chat template and tokenised as the client does. No step runs. |
| `POST /tokenize` | untested | vLLM's two bodies: `prompt` (`add_special_tokens` defaults to true) or `messages` rendered with the model's chat template (`add_special_tokens` defaults to false; `add_generation_prompt` true and `continue_final_message` false only). `return_token_strs` gives `token_strs`. Answers `count`, `max_model_len` (the worker's context, 512 tokens for the `roost serve` profile), `tokens`, `token_strs`. |
| `POST /detokenize` | untested | vLLM's body `{"model","tokens"}`; answers `{"prompt"}` decoded with special tokens kept, as vLLM decodes. An id outside the vocabulary is refused naming its index (`tokens[1]`). |
| `GET /health`, `GET`/`POST /ping` | untested | Liveness, vLLM's names: 200 with an empty body while the process serves HTTP. `RoostChatClient` exposes no worker liveness, so a dead worker shows as failed requests, not a failed health check. |
| `GET /ready` | untested | Readiness: the server is built over a client whose worker has started (spawn, handshake, weight load), so it answers 200 `{"status":"ready","model"}` once listening, and 503 `stopping` once the host begins to stop. |
| `GET /version` | untested | vLLM's name: `{"version"}`, the `roost` assembly's informational version. |

## Metrics, traces and logs (R-038)

| Item | Status | Notes |
| --- | --- | --- |
| `GET /metrics` | untested | Prometheus text format 0.0.4 (`text/plain; version=0.0.4`), written by `Metrics/PrometheusExporter` from a `MeterListener` over the client's `RoostTelemetry.Meter`; no exporter package. Every series has vLLM's `model_name` label. |
| Recorded in `Starlings.Generation.Roost` | untested | `RoostChatClient.Telemetry` records for every request submitted to the worker, whichever route (or library caller) sent it: finish reason `stop`, `length`, `abort` (cancelled or abandoned) or `error`; prompt and generated tokens; time to first token, inter-token latency and end-to-end latency, measured in the host from the start of tokenisation; requests waiting (submitted, no token yet) and running (first token received); the share of the KV cache's attention blocks in use after the last planned step. A request refused while it is read records nothing. |
| One activity per request | untested | `roost.request` on `ActivitySource` `Starlings.Roost`, tagged `roost.request_id` (the id the route returns without its `chatcmpl-`, `cmpl-` or other prefix), `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, and `error.type` with the C6 code name on failure. Never prompt or output text. |
| Log hygiene | untested | The server's only own log message is an unexpected exception (Roost's exception messages carry no prompt text, C6). `TracesAndLogsTests` captures every category at `Trace` over every route, streamed and whole, and over refusals, and finds neither the prompt nor the reply. |

Metric families (Roost name, then vLLM's names, which are aliases of the same values):

| Roost | vLLM | Type |
| --- | --- | --- |
| `roost_requests_total{finish_reason}` | `vllm:request_success_total{finished_reason}` (stop, length, abort; `error` is Roost's only) | counter |
| `roost_prompt_tokens_total` | `vllm:prompt_tokens_total` | counter |
| `roost_generation_tokens_total` | `vllm:generation_tokens_total` | counter |
| `roost_request_prompt_tokens` | `vllm:request_prompt_tokens` | histogram |
| `roost_request_generation_tokens` | `vllm:request_generation_tokens` | histogram |
| `roost_time_to_first_token_seconds` | `vllm:time_to_first_token_seconds` | histogram |
| `roost_inter_token_latency_seconds` | `vllm:inter_token_latency_seconds`, `vllm:time_per_output_token_seconds` | histogram |
| `roost_e2e_request_latency_seconds` | `vllm:e2e_request_latency_seconds` | histogram |
| `roost_requests_running` | `vllm:num_requests_running` | gauge |
| `roost_requests_waiting` | `vllm:num_requests_waiting` | gauge |
| `roost_kv_cache_usage_ratio` | `vllm:kv_cache_usage_perc`, `vllm:gpu_cache_usage_perc` (a 0 to 1 fraction, as in vLLM) | gauge |

Not exported: prefix-cache queries and hits (the M0 driver does not use the prefix cache, R-030), preemptions, queue time,
prefill and decode time, and vLLM's `engine` label. Bucket boundaries span the ranges of vLLM's histograms but are Roost's
own. M0 runs one sequence per step, so `running` is at most 1 and concurrent requests show as `waiting`. No number from this
endpoint is a throughput claim (AGENTS.md); it has not been scraped on a GPU.

## Tool calls and reasoning (R-053)

| Item | Status | Notes |
| --- | --- | --- |
| Parsers in `Starlings.Generation.Roost` | untested | `Parsing/`: `ThinkBlockParser` (Qwen think blocks: with thinking enabled the completion is reasoning up to the first `</think>`, then the answer; a repeated leading `<think>` is dropped; line feeds trimmed as the Qwen3.5 template trims them when it reads a think block back), `HermesToolCallParser` (`<tool_call>{"name","arguments"}</tool_call>`, the Qwen2.5 and Qwen3 format) and `Qwen35ToolCallStream` over Starling's published `Qwen35ToolCallParser` (the Qwen3.5 `<function=...><parameter=...>` format, values cast with the tool's schema). All incremental: text is released as soon as it cannot start a tag, a call when its block closes, and malformed or unfinished blocks come back as text, verbatim. Calls are parsed only when the request rendered tools, reasoning only when it enabled thinking, so other requests and the other routes see the completion as before. |
| `RoostChatClient` | untested | `ChatOptions.Tools` (`AIFunction` declarations: name, description, JSON schema) and `AdditionalProperties["enable_thinking"]` (`RoostChatClient.EnableThinkingProperty`) reach `IRoostChatTemplate.Render`. Output: `TextReasoningContent`, `TextContent` and `FunctionCallContent` (id `call_<response id>_<index>`, arguments as `JsonElement` values, `RawRepresentation` the `RoostToolCall`); finish reason `ToolCalls` when a call was returned and the model stopped. History: assistant `FunctionCallContent` and `TextReasoningContent`, and tool messages of `FunctionResultContent` (one template tool message per result) or text. `ToolMode` none renders no tools and parses no calls; `RequireAny`, `RequireSpecific` and `AllowMultipleToolCalls` false are refused (nothing constrains decoding until R-051). The format comes from `RoostModelProfile.ToolCallFormat` and `ReasoningFormat`, by default from the template: Qwen3.5 for `Qwen35RoostChatTemplate`, Hermes and think tags for `Qwen3RoostChatTemplate`, whose published template still refuses tools and thinking (`InvalidOption` naming `tools` or `enable_thinking`). |
| `POST /v1/chat/completions` | untested | Request: `tools` (`type` `function`; `strict` true refused), `tool_choice` (`none`, `auto`; `required` and a named function are refused by the client), `parallel_tool_calls` (true; false refused), vLLM's `chat_template_kwargs` with `enable_thinking` only, assistant `tool_calls` (arguments a JSON object in a string) and `reasoning_content`, `tool` messages with `tool_call_id`. Response: `message.tool_calls` (`content` null when the message is only calls) and `message.reasoning_content` (omitted when empty); stream deltas carry `reasoning_content` text and one `tool_calls` delta per call with its whole arguments (OpenAI allows the arguments in one delta; vLLM splits them). |
| Corpus | untested | `tests/Starlings.Generation.Roost.Tests/Parsing/TemplateCorpus.cs`. P-029 (the template corpus the build plan names) is not in this repository, so the tests use a local corpus: Qwen3.5 assistant turns rendered by Starling's published template, which writes the think block and tool calls as it instructs the model to answer, parsed back into their reasoning, content and calls; hand-written Hermes completions in the upstream Qwen2.5/Qwen3 format (Starling's Qwen3 template has no tools); think-block edge cases. Every case is also parsed from single-character deltas and from every split in two. The corpus is not a fixture: no status flips on it. |

## Behaviour

- **One error body.** Every failure outside the Anthropic routes answers `{"error":{"message","type","param","code"}}` with `type` `roost_error`. Request
  and generation errors use `RoostError.WriteHttpBody` (C6 code names and HTTP statuses: `invalid_option` and
  `prompt_too_long` are 400, `internal` 500, and so on); routing failures use the same shape with `not_found` (404) or
  `method_not_allowed` (405). A stream that fails after its headers were sent ends with a `data:` event carrying the error body,
  then `data: [DONE]`. A streamed request refused on its first step (a prompt longer than the context) still gets its HTTP
  status, because the first update is awaited before the headers are sent.
- **Anthropic error body.** `/v1/messages` and `/v1/messages/count_tokens` answer `{"type":"error","error":{"type","message"}}`
  with Anthropic's type for the status (400 `invalid_request_error`, 404 `not_found_error`, 429 `rate_limit_error`, 500
  `api_error`, 503 `overloaded_error`). The body has no parameter field, so a refused field starts the message
  (`"tools: Tools are unsupported ..."`). Routing failures (404, 405) keep the one OpenAI-shaped body on every path.
- **Response store.** In memory, at most `RoostServerOptions.ResponseStoreCapacity` responses (1024 by default), each kept for
  `ResponseStoreTtl` (one hour) after creation; a running response is never dropped. A full store drops its oldest finished
  response; a store full of running responses refuses a new stored one with `queue_full` (429). `store` defaults to true, as
  in OpenAI's API; `store` false keeps nothing (retrieve is a 404 `not_found`). A whole response is stored once complete, a
  streamed one from `response.created`, a background one at creation. Nothing survives a restart. `background` needs `store`,
  and `background` with `stream` is refused.
- **Options.** Request bodies are read field by field (`OpenAiRequestReader`). Every field is mapped, accepted at a value that
  changes nothing (`n` 1, `logprobs` false, `response_format` `{"type":"text"}`, `store` false, empty
  `logit_bias`, `best_of` 1, `echo` false, empty `suffix`, `user`), or refused with a 400 whose `param` names it; an unknown
  field is refused the same way. JSON `null` means unset. Sampling, penalties, `top_k`, `stop`, `max_tokens` and `model` are
  passed to `RoostChatClient`, whose refusals (greedy decoding only in M0) name the same OpenAI parameter. Message fields are
  named by path, for example `messages[0].content[0].type`. An absent `temperature` means greedy decoding, not OpenAI's 1.0
  (nor Anthropic's 1.0 on `/v1/messages`). The Responses, Messages and tokenize readers follow the same rules: for example
  `tools` (unless empty), `reasoning`, `previous_response_id`, `conversation`, JSON `text.format`, `truncation` `auto`,
  non-empty `include`, `top_logprobs` and `stream_options` on `/v1/responses`; `tools` (unless empty), `tool_choice` other
  than `none`, `thinking` other than `disabled`, non-text blocks, a final assistant message (prefill) and unknown `metadata`
  keys on `/v1/messages`; `chat_template`, `tools` and `add_generation_prompt` false on `/tokenize`.
- **Prompt embeddings.** Refused: vLLM's `prompt_embeds` on `/v1/completions` is a 400 naming it. Contract C2 carries
  embeddings as a `MultimodalPart` of kind `Embeddings`, but `Starlings.Generation.Roost`'s `RoostChatClient` exposes no way to
  pass one, so the server cannot reach the C2 slot until that package gains an embedding input (open; R-074 was scoped to the
  server).
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
- **OpenAI client libraries** against the server: the R-040 suite (`tests/Starlings.Roost.ApiCompat.Tests/STATUS.md`) runs the
  official .NET client in CI and the Python package opt-in, over the scripted worker only. It found `GET /v1/models/{id}`
  unrouted (404).
- **R-038 gaps.** No GPU run has been scraped; prefix-cache, preemption and queue-time families wait on the engine (R-030,
  R-032); no OpenTelemetry exporter is wired (a host adds one over the `Starlings.Roost` meter and activity source).
- **R-074 gaps.** Prompt-embeddings input waits on an embedding input in `RoostChatClient`. `stop_reason`
  `stop_sequence` is never reported on `/v1/messages` (the client reports a stop string and a stop token alike, so both are
  `end_turn`), and a stream's `message_start` carries 0 input tokens (usage arrives with the last update; `message_delta` has
  both counts). Not built: `DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items`, streaming a stored response,
  and vLLM's other utility routes (`/load`, `/is_sleeping`, `/server_info` and the like). The R-040 suite covers the Responses
  routes with the official OpenAI clients; no Anthropic client library has run against `/v1/messages`.
- **R-053 gaps.** No model has run with tools or thinking on Roost (no GPU run; `tools` and `reasoning` stay `untested` until
  fixtures exist). Tools are on `/v1/chat/completions` only: `/v1/responses` (`function_call` items), `/v1/messages`
  (`tool_use` blocks) and `/tokenize` still refuse them, and thinking is not mapped to Anthropic's `thinking` or OpenAI's
  `reasoning_effort`. A tool without a description renders `"description": ""`, because `RoostToolDefinition.Description` (stream
  B1's type) is not nullable. Parsers for other families (Llama, Mistral, DeepSeek and so on) come per family with their
  templates.
- **Structured output, sampling, log probabilities, `n` > 1, multimodal input**: refused with a 400 naming the option until
  R-051 and R-042 land.
- **vLLM's CLI flags and environment variables**: R-075.
- **Concurrency**: M0 runs one sequence per step, so concurrent requests take turns (`continuous_batching` `unsupported`).
