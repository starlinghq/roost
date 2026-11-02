using System.Reflection;
using System.Text.Json;
using Starlings.Generation.Roost;
using Starlings.Roost.Server.OpenAi;
using Starlings.Roost.Server.Routing;
using static Starlings.Roost.Server.Routing.RequestFields;

namespace Starlings.Roost.Server.Utility;

/// <summary>
/// vLLM's utility routes (R-074): <c>POST /tokenize</c> and <c>POST /detokenize</c> with vLLM's bodies over the model's tokenizer,
/// <c>GET /health</c> (liveness), <c>GET</c> and <c>POST /ping</c> (vLLM's SageMaker alias of <c>/health</c>), <c>GET /version</c>,
/// and Roost's <c>GET /ready</c> (readiness). Errors use the OpenAI envelope (<see cref="ErrorBodies"/>), as vLLM's do.
/// </summary>
/// <remarks>
/// The server is built over a client whose worker has already started (spawn, handshake and weight load), so it is ready as soon
/// as it listens; <c>/ready</c> answers 503 once the host begins to stop. <c>/health</c> answers 200 while the process serves
/// HTTP: no worker liveness is exposed by <see cref="RoostChatClient"/>, so a dead worker shows up as failed requests, not as a
/// failed health check.
/// </remarks>
internal sealed class UtilityRoutes
{
    private readonly RoostChatClient _client;
    private readonly CancellationToken _stopping;
    private readonly RouteRunner _runner;
    private readonly byte[] _version;

    public UtilityRoutes(RoostChatClient client, ILogger logger, CancellationToken stopping)
    {
        _client = client;
        _stopping = stopping;
        _runner = new RouteRunner(logger, ErrorBodies.Shape);
        string version = typeof(UtilityRoutes).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";
        _version = RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteString("version", version);
            w.WriteEndObject();
        });
    }

    private RoostModelProfile Model => _client.Model;

    /// <summary><c>GET /health</c> and <c>/ping</c>: 200 with an empty body, as vLLM answers.</summary>
    public static Task HealthAsync(HttpContext context)
    {
        context.Response.StatusCode = StatusCodes.Status200OK;
        return Task.CompletedTask;
    }

    /// <summary><c>GET /ready</c>: 200 while the worker is started and the host is not stopping, else 503.</summary>
    public Task ReadyAsync(HttpContext context)
    {
        bool ready = !_stopping.IsCancellationRequested;
        context.Response.StatusCode = ready ? StatusCodes.Status200OK : StatusCodes.Status503ServiceUnavailable;
        context.Response.ContentType = "application/json";
        var body = RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteString("status", ready ? "ready" : "stopping");
            w.WriteString("model", Model.ModelId);
            w.WriteEndObject();
        });
        return context.Response.Body.WriteAsync(body, context.RequestAborted).AsTask();
    }

    /// <summary><c>GET /version</c>: <c>{"version":…}</c>, the <c>roost</c> assembly's informational version.</summary>
    public Task VersionAsync(HttpContext context) => RouteRunner.WriteJsonAsync(context, _version);

    /// <summary>
    /// <c>POST /tokenize</c>, vLLM's two bodies: <c>{"prompt", "add_special_tokens" (default true)}</c>, or
    /// <c>{"messages", "add_generation_prompt" (true only), "continue_final_message" (false only), "add_special_tokens" (default
    /// false)}</c> rendered with the model's chat template. Both take <c>model</c> and <c>return_token_strs</c>. Answers
    /// <c>{"count","max_model_len","tokens","token_strs"}</c>, where <c>max_model_len</c> is the worker's context.
    /// </summary>
    public Task TokenizeAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        int[] ids;
        bool tokenStrings = false;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            string? prompt = null;
            JsonElement? messages = null;
            bool? addSpecial = null;
            foreach (var field in Fields(body.RootElement))
            {
                var v = field.Value;
                switch (field.Name)
                {
                    case "model":
                        CheckModel(v);
                        break;
                    case "prompt":
                        prompt = StringValue(v, "prompt");
                        break;
                    case "messages":
                        messages = v;
                        break;
                    case "add_special_tokens":
                        addSpecial = Bool(v, field.Name);
                        break;
                    case "return_token_strs":
                        tokenStrings = Bool(v, field.Name);
                        break;
                    case "add_generation_prompt":
                        if (!Bool(v, field.Name))
                        {
                            throw Refuse(field.Name, "add_generation_prompt is unsupported except true: the chat template always ends with the generation prompt.");
                        }

                        break;
                    case "continue_final_message":
                        if (Bool(v, field.Name))
                        {
                            throw Refuse(field.Name, "continue_final_message is unsupported except false.");
                        }

                        break;
                    case "chat_template":
                    case "chat_template_kwargs":
                    case "tools":
                    case "mm_processor_kwargs":
                        throw Refuse(field.Name, $"'{field.Name}' is unsupported: the server renders with the model's own template, text only.");
                    default:
                        throw Unknown(field.Name);
                }
            }

            if (messages is { } m)
            {
                if (prompt is not null)
                {
                    throw Refuse("prompt", "Set prompt or messages, not both.");
                }

                var text = ChatPrompts.Render(Model, OpenAiRequestReader.Messages(m), null);
                ids = ChatPrompts.Encode(Model, text, addSpecial ?? false);
            }
            else
            {
                if (prompt is null)
                {
                    throw Refuse("prompt", "prompt or messages is required.");
                }

                ids = ChatPrompts.Encode(Model, prompt, addSpecial ?? true);
            }
        }

        var json = RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteNumber("count", ids.Length);
            w.WriteNumber("max_model_len", Model.ContextTokens);
            w.WriteStartArray("tokens");
            foreach (var id in ids)
            {
                w.WriteNumberValue(id);
            }

            w.WriteEndArray();
            if (tokenStrings)
            {
                w.WriteStartArray("token_strs");
                foreach (var id in ids)
                {
                    w.WriteStringValue(Model.Tokenizer.IdToToken(id));
                }

                w.WriteEndArray();
            }
            else
            {
                w.WriteNull("token_strs");
            }

            w.WriteEndObject();
        });
        await RouteRunner.WriteJsonAsync(context, json).ConfigureAwait(false);
    });

    /// <summary>
    /// <c>POST /detokenize</c>, vLLM's body <c>{"model", "tokens"}</c>: answers <c>{"prompt"}</c>, the ids decoded with special
    /// tokens kept (as vLLM decodes). An id outside the vocabulary is refused naming its index.
    /// </summary>
    public Task DetokenizeAsync(HttpContext context) => _runner.RunAsync(context, async ct =>
    {
        int[]? ids = null;
        using (var body = await RouteRunner.ReadBodyAsync(context, ct).ConfigureAwait(false))
        {
            foreach (var field in Fields(body.RootElement))
            {
                switch (field.Name)
                {
                    case "model":
                        CheckModel(field.Value);
                        break;
                    case "tokens":
                        ids = Tokens(field.Value);
                        break;
                    default:
                        throw Unknown(field.Name);
                }
            }
        }

        var text = Model.Tokenizer.Decode(ids ?? throw Refuse("tokens", "tokens is required."), false);
        await RouteRunner.WriteJsonAsync(context, RouteRunner.Json(w =>
        {
            w.WriteStartObject();
            w.WriteString("prompt", text);
            w.WriteEndObject();
        })).ConfigureAwait(false);
    });

    private void CheckModel(JsonElement v)
    {
        var model = StringValue(v, "model");
        if (!string.Equals(model, Model.ModelId, StringComparison.Ordinal))
        {
            throw Refuse("model", $"Model '{model}' is not served here; this server serves '{Model.ModelId}'.");
        }
    }

    private int[] Tokens(JsonElement v)
    {
        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse("tokens", "tokens must be an array of token ids.");
        }

        int vocab = Model.Tokenizer.GetVocabularySize(true);
        var ids = new int[v.GetArrayLength()];
        var i = 0;
        foreach (var item in v.EnumerateArray())
        {
            string param = Inv($"tokens[{i}]");
            int id = item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out var x) ? x : throw Refuse(param, "A token id must be an integer.");
            ids[i++] = id >= 0 && id < vocab ? id : throw Refuse(param, Inv($"Token id {id} is outside the vocabulary (0 to {vocab - 1})."));
        }

        return ids;
    }
}
