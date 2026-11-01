using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;
using static Starlings.Roost.Server.Routing.RequestFields;

namespace Starlings.Roost.Server.OpenAi;

/// <summary>A checked <c>POST /v1/chat/completions</c> body.</summary>
/// <param name="Messages">The conversation.</param>
/// <param name="Options">The options, for <see cref="RoostChatClient"/> to check against what the model runs.</param>
/// <param name="Stream">Whether to answer with server-sent events.</param>
/// <param name="IncludeUsage">Whether a stream ends with a usage chunk (<c>stream_options.include_usage</c>).</param>
internal sealed record ChatCompletionRequest(IReadOnlyList<ChatMessage> Messages, ChatOptions Options, bool Stream, bool IncludeUsage);

/// <summary>A checked <c>POST /v1/completions</c> body.</summary>
/// <param name="Prompt">The raw prompt.</param>
/// <param name="Options">The options, for <see cref="RoostChatClient"/> to check against what the model runs.</param>
/// <param name="Stream">Whether to answer with server-sent events.</param>
/// <param name="IncludeUsage">Whether a stream ends with a usage chunk (<c>stream_options.include_usage</c>).</param>
internal sealed record CompletionRequest(string Prompt, ChatOptions Options, bool Stream, bool IncludeUsage);

/// <summary>
/// Reads OpenAI request bodies field by field. Every field is either mapped, accepted at a value that changes nothing (for
/// example <c>n</c> = 1 or <c>logprobs</c> = false), or refused with a 400 whose <c>param</c> names it; an unknown field is
/// refused the same way, so nothing is silently ignored. A JSON <c>null</c> means the field is not set, as in OpenAI's API.
/// Options that map onto <see cref="ChatOptions"/> (sampling, penalties, the model) are checked by
/// <see cref="RoostChatClient"/>, which refuses what the model does not run, naming the same OpenAI parameter.
/// </summary>
internal static class OpenAiRequestReader
{
    /// <summary>The completions route's <c>max_tokens</c> default, as in OpenAI's API and vLLM.</summary>
    public const int CompletionsDefaultMaxTokens = 16;

    /// <summary>
    /// Why vLLM's <c>prompt_embeds</c> is refused: contract C2 carries embeddings as a <c>MultimodalPart</c> of kind
    /// <c>Embeddings</c>, but <see cref="RoostChatClient"/> exposes no way to pass one (R-074).
    /// </summary>
    public const string PromptEmbedsUnsupported =
        "Prompt-embeddings input is unsupported: Starlings.Generation.Roost's RoostChatClient exposes no embedding input (contract C2's Embeddings part).";

    public static ChatCompletionRequest ReadChat(JsonElement body)
    {
        var options = new ChatOptions();
        IReadOnlyList<ChatMessage>? messages = null;
        string? model = null;
        bool stream = false;
        JsonElement? streamOptions = null;
        int? maxTokens = null;
        foreach (var field in Fields(body))
        {
            var v = field.Value;
            switch (field.Name)
            {
                case "model":
                    model = StringValue(v, "model");
                    break;
                case "messages":
                    messages = Messages(v);
                    break;
                case "max_tokens":
                case "max_completion_tokens":
                    if (maxTokens is not null)
                    {
                        throw Refuse(field.Name, "Set max_tokens or max_completion_tokens, not both.");
                    }

                    maxTokens = PositiveInt(v, field.Name);
                    break;
                case "stream":
                    stream = Bool(v, "stream");
                    break;
                case "stream_options":
                    streamOptions = v;
                    break;
                case "tools":
                    throw Refuse("tools", "Tools are unsupported: tool calling is not built yet (R-053).");
                case "tool_choice":
                    if (v.ValueKind != JsonValueKind.String || v.GetString() != "none")
                    {
                        throw Refuse("tool_choice", "tool_choice is unsupported except \"none\": tools are unsupported.");
                    }

                    break;
                case "parallel_tool_calls":
                    throw Refuse("parallel_tool_calls", "parallel_tool_calls is unsupported: tools are unsupported.");
                case "response_format":
                    ResponseFormat(v);
                    break;
                case "logprobs":
                    if (Bool(v, "logprobs"))
                    {
                        throw Refuse("logprobs", "Log probabilities are unsupported.");
                    }

                    break;
                case "top_logprobs":
                    throw Refuse("top_logprobs", "Log probabilities are unsupported.");
                default:
                    if (!Common(field.Name, v, options))
                    {
                        throw Unknown(field.Name);
                    }

                    break;
            }
        }

        options.ModelId = model ?? throw Refuse("model", "model is required.");
        if (messages is null)
        {
            throw Refuse("messages", "messages is required.");
        }

        options.MaxOutputTokens = maxTokens;
        var includeUsage = StreamOptions(streamOptions, stream);
        return new ChatCompletionRequest(messages, options, stream, includeUsage);
    }

    public static CompletionRequest ReadCompletion(JsonElement body)
    {
        var options = new ChatOptions();
        string? prompt = null;
        string? model = null;
        bool stream = false;
        JsonElement? streamOptions = null;
        int maxTokens = CompletionsDefaultMaxTokens;
        foreach (var field in Fields(body))
        {
            var v = field.Value;
            switch (field.Name)
            {
                case "model":
                    model = StringValue(v, "model");
                    break;
                case "prompt":
                    prompt = Prompt(v);
                    break;
                case "max_tokens":
                    maxTokens = PositiveInt(v, "max_tokens");
                    break;
                case "stream":
                    stream = Bool(v, "stream");
                    break;
                case "stream_options":
                    streamOptions = v;
                    break;
                case "logprobs":
                    throw Refuse("logprobs", "Log probabilities are unsupported.");
                case "echo":
                    if (Bool(v, "echo"))
                    {
                        throw Refuse("echo", "echo is unsupported.");
                    }

                    break;
                case "best_of":
                    if (v.ValueKind != JsonValueKind.Number || !v.TryGetInt32(out var bestOf) || bestOf != 1)
                    {
                        throw Refuse("best_of", "best_of is unsupported except 1.");
                    }

                    break;
                case "prompt_embeds": // vLLM's extension
                    throw Refuse("prompt_embeds", PromptEmbedsUnsupported);
                case "suffix":
                    if (v.ValueKind != JsonValueKind.String || v.GetString()!.Length != 0)
                    {
                        throw Refuse("suffix", "suffix (insertion) is unsupported.");
                    }

                    break;
                default:
                    if (!Common(field.Name, v, options))
                    {
                        throw Unknown(field.Name);
                    }

                    break;
            }
        }

        options.ModelId = model ?? throw Refuse("model", "model is required.");
        options.MaxOutputTokens = maxTokens;
        var includeUsage = StreamOptions(streamOptions, stream);
        return new CompletionRequest(prompt ?? throw Refuse("prompt", "prompt is required."), options, stream, includeUsage);
    }

    /// <summary>Fields both routes share. Returns false for a name neither route knows.</summary>
    private static bool Common(string name, JsonElement v, ChatOptions options)
    {
        switch (name)
        {
            case "temperature":
                options.Temperature = Number(v, name);
                return true;
            case "top_p":
                options.TopP = Number(v, name);
                return true;
            case "top_k": // vLLM's extension
                options.TopK = Int(v, name);
                return true;
            case "frequency_penalty":
                options.FrequencyPenalty = Number(v, name);
                return true;
            case "presence_penalty":
                options.PresencePenalty = Number(v, name);
                return true;
            case "seed":
                options.Seed = Int(v, name);
                return true;
            case "stop":
                options.StopSequences = Stop(v);
                return true;
            case "n":
                if (Int(v, name) != 1)
                {
                    throw Refuse("n", "n is unsupported except 1: one choice per request.");
                }

                return true;
            case "user":
                // An end-user identifier for the caller's own records; it changes nothing about the output. Checked, not used.
                StringValue(v, name);
                return true;
            case "logit_bias":
                if (v.ValueKind != JsonValueKind.Object || v.EnumerateObject().Any())
                {
                    throw Refuse("logit_bias", "logit_bias is unsupported.");
                }

                return true;
            case "store":
                if (Bool(v, name))
                {
                    throw Refuse("store", "store is unsupported: Roost keeps no responses.");
                }

                return true;
            default:
                return false;
        }
    }

    internal static List<ChatMessage> Messages(JsonElement v)
    {
        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse("messages", "messages must be an array.");
        }

        var list = new List<ChatMessage>();
        foreach (var item in v.EnumerateArray())
        {
            list.Add(Message(item, Inv($"messages[{list.Count}]")));
        }

        return list;
    }

    private static ChatMessage Message(JsonElement v, string at)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse(at, "A message must be an object.");
        }

        ChatRole? role = null;
        var contents = new List<AIContent>();
        string? name = null;
        foreach (var field in Fields(v, at))
        {
            var f = field.Value;
            string param = at + "." + field.Name;
            switch (field.Name)
            {
                case "role":
                    role = StringValue(f, param) switch
                    {
                        "system" or "developer" => ChatRole.System,
                        "user" => ChatRole.User,
                        "assistant" => ChatRole.Assistant,
                        "tool" => ChatRole.Tool, // refused by the client: tools are unsupported
                        var other => throw Refuse(param, $"Role '{other}' is unsupported."),
                    };
                    break;
                case "content":
                    Content(f, param, contents);
                    break;
                case "name":
                    name = StringValue(f, param);
                    break;
                default:
                    // tool_calls, tool_call_id, function_call, audio, refusal and anything else: not supported.
                    throw Refuse(param, $"Message field '{field.Name}' is unsupported.");
            }
        }

        return new ChatMessage(role ?? throw Refuse(at + ".role", "role is required."), contents) { AuthorName = name };
    }

    private static void Content(JsonElement v, string param, List<AIContent> contents)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            contents.Add(new TextContent(v.GetString()));
            return;
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse(param, "content must be a string or an array of content parts.");
        }

        var index = 0;
        foreach (var part in v.EnumerateArray())
        {
            string at = Inv($"{param}[{index++}]");
            if (part.ValueKind != JsonValueKind.Object)
            {
                throw Refuse(at, "A content part must be an object.");
            }

            string? type = null;
            string? text = null;
            foreach (var field in Fields(part, at))
            {
                switch (field.Name)
                {
                    case "type":
                        type = StringValue(field.Value, at + ".type");
                        break;
                    case "text":
                        text = StringValue(field.Value, at + ".text");
                        break;
                    default:
                        throw Refuse(at + "." + field.Name, $"Content part field '{field.Name}' is unsupported; only text parts are accepted.");
                }
            }

            if (type != "text")
            {
                throw Refuse(at + ".type", $"Content part type '{type}' is unsupported; only text parts are accepted.");
            }

            contents.Add(new TextContent(text ?? throw Refuse(at + ".text", "A text part needs text.")));
        }
    }

    private static string Prompt(JsonElement v)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            return v.GetString()!;
        }

        if (v.ValueKind == JsonValueKind.Array && v.GetArrayLength() == 1 && v[0].ValueKind == JsonValueKind.String)
        {
            return v[0].GetString()!;
        }

        throw Refuse("prompt", "prompt must be a string (or an array of one string); token-id prompts and batches of prompts are unsupported.");
    }

    private static List<string> Stop(JsonElement v)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            return [v.GetString()!];
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse("stop", "stop must be a string or an array of strings.");
        }

        var stops = new List<string>();
        foreach (var item in v.EnumerateArray())
        {
            stops.Add(item.ValueKind == JsonValueKind.String ? item.GetString()! : throw Refuse("stop", "stop must be a string or an array of strings."));
        }

        return stops;
    }

    private static void ResponseFormat(JsonElement v)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse("response_format", "response_format must be an object.");
        }

        foreach (var field in Fields(v, "response_format"))
        {
            if (field.Name != "type" || field.Value.ValueKind != JsonValueKind.String || field.Value.GetString() != "text")
            {
                throw Refuse("response_format", "JSON and JSON-schema response formats are unsupported: structured output is not built yet (R-051). Only {\"type\":\"text\"} is accepted.");
            }
        }
    }

    private static bool StreamOptions(JsonElement? v, bool stream)
    {
        if (v is not { } options)
        {
            return false;
        }

        if (!stream)
        {
            throw Refuse("stream_options", "stream_options is only allowed when stream is true.");
        }

        if (options.ValueKind != JsonValueKind.Object)
        {
            throw Refuse("stream_options", "stream_options must be an object.");
        }

        var includeUsage = false;
        foreach (var field in Fields(options, "stream_options"))
        {
            includeUsage = field.Name == "include_usage"
                ? Bool(field.Value, "stream_options.include_usage")
                : throw Refuse("stream_options." + field.Name, $"Stream option '{field.Name}' is unsupported.");
        }

        return includeUsage;
    }
}
