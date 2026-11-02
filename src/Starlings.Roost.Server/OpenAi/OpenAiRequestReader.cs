using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Requests;
using Starlings.Roost.Server.Routing;
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
                    options.Tools = Tools(v);
                    break;
                case "tool_choice":
                    options.ToolMode = ToolChoice(v);
                    break;
                case "parallel_tool_calls":
                    options.AllowMultipleToolCalls = Bool(v, "parallel_tool_calls");
                    break;
                case "chat_template_kwargs": // vLLM's extension
                    ChatTemplateKwargs(v, options);
                    break;
                case "response_format":
                    options.ResponseFormat = ResponseFormats.Read(v, "response_format", flat: false);
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
        string? toolCallId = null;
        List<AIContent>? calls = null;
        string? reasoning = null;
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
                        "tool" => ChatRole.Tool,
                        var other => throw Refuse(param, $"Role '{other}' is unsupported."),
                    };
                    break;
                case "content":
                    Content(f, param, contents);
                    break;
                case "name":
                    name = StringValue(f, param);
                    break;
                case "tool_calls":
                    calls = ToolCalls(f, param);
                    break;
                case "tool_call_id":
                    toolCallId = StringValue(f, param);
                    break;
                case "reasoning_content": // vLLM's field, as its responses carry it
                    reasoning = StringValue(f, param);
                    break;
                default:
                    // function_call, audio, refusal and anything else: not supported.
                    throw Refuse(param, $"Message field '{field.Name}' is unsupported.");
            }
        }

        var r = role ?? throw Refuse(at + ".role", "role is required.");
        if (calls is not null && r != ChatRole.Assistant)
        {
            throw Refuse(at + ".tool_calls", "tool_calls is only allowed on assistant messages.");
        }

        if (reasoning is not null && r != ChatRole.Assistant)
        {
            throw Refuse(at + ".reasoning_content", "reasoning_content is only allowed on assistant messages.");
        }

        if (r == ChatRole.Tool)
        {
            // One tool result: the message's text, for the call it names.
            var callId = toolCallId ?? throw Refuse(at + ".tool_call_id", "A tool message needs tool_call_id.");
            var result = string.Concat(contents.OfType<TextContent>().Select(t => t.Text));
            return new ChatMessage(r, [new FunctionResultContent(callId, result)]) { AuthorName = name };
        }

        if (toolCallId is not null)
        {
            throw Refuse(at + ".tool_call_id", "tool_call_id is only allowed on tool messages.");
        }

        if (reasoning is not null)
        {
            contents.Insert(0, new TextReasoningContent(reasoning));
        }

        if (calls is not null)
        {
            contents.AddRange(calls);
        }

        return new ChatMessage(r, contents) { AuthorName = name };
    }

    /// <summary>
    /// An assistant message's <c>tool_calls</c>: <c>{"id","type":"function","function":{"name","arguments"}}</c>, with the arguments
    /// a JSON object in a string, as OpenAI's API sends them.
    /// </summary>
    private static List<AIContent> ToolCalls(JsonElement v, string param)
    {
        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse(param, "tool_calls must be an array.");
        }

        var calls = new List<AIContent>();
        foreach (var item in v.EnumerateArray())
        {
            string at = Inv($"{param}[{calls.Count}]");
            if (item.ValueKind != JsonValueKind.Object)
            {
                throw Refuse(at, "A tool call must be an object.");
            }

            string? id = null;
            string? type = null;
            JsonElement? function = null;
            foreach (var field in Fields(item, at))
            {
                switch (field.Name)
                {
                    case "id":
                        id = StringValue(field.Value, at + ".id");
                        break;
                    case "type":
                        type = StringValue(field.Value, at + ".type");
                        break;
                    case "function":
                        function = field.Value;
                        break;
                    default:
                        throw Refuse(at + "." + field.Name, $"Tool call field '{field.Name}' is unsupported.");
                }
            }

            if (type is not null and not "function")
            {
                throw Refuse(at + ".type", $"Tool call type '{type}' is unsupported; only \"function\" is accepted.");
            }

            var (name, arguments) = FunctionCall(function ?? throw Refuse(at + ".function", "A tool call needs function."), at + ".function");
            calls.Add(new FunctionCallContent(id ?? throw Refuse(at + ".id", "A tool call needs id."), name, arguments));
        }

        return calls;
    }

    private static (string Name, Dictionary<string, object?> Arguments) FunctionCall(JsonElement v, string at)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse(at, "function must be an object.");
        }

        string? name = null;
        Dictionary<string, object?>? arguments = null;
        foreach (var field in Fields(v, at))
        {
            switch (field.Name)
            {
                case "name":
                    name = StringValue(field.Value, at + ".name");
                    break;
                case "arguments":
                    arguments = ArgumentsObject(StringValue(field.Value, at + ".arguments"), at + ".arguments");
                    break;
                default:
                    throw Refuse(at + "." + field.Name, $"Function call field '{field.Name}' is unsupported.");
            }
        }

        return (name ?? throw Refuse(at + ".name", "A function call needs name."), arguments ?? []);
    }

    private static Dictionary<string, object?> ArgumentsObject(string json, string param)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw Refuse(param, "arguments must be a JSON object, as a string.");
            }

            var arguments = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var member in document.RootElement.EnumerateObject())
            {
                arguments[member.Name] = member.Value.Clone();
            }

            return arguments;
        }
        catch (JsonException)
        {
            throw Refuse(param, "arguments is not valid JSON.");
        }
    }

    /// <summary>
    /// <c>tools</c>: function definitions, <c>{"type":"function","function":{"name","description","parameters","strict"}}</c>.
    /// <c>strict</c> true is refused: nothing constrains decoding to the schema (structured output, R-051).
    /// </summary>
    private static List<AITool> Tools(JsonElement v)
    {
        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse("tools", "tools must be an array.");
        }

        var tools = new List<AITool>();
        foreach (var item in v.EnumerateArray())
        {
            string at = Inv($"tools[{tools.Count}]");
            if (item.ValueKind != JsonValueKind.Object)
            {
                throw Refuse(at, "A tool must be an object.");
            }

            string? type = null;
            JsonElement? function = null;
            foreach (var field in Fields(item, at))
            {
                switch (field.Name)
                {
                    case "type":
                        type = StringValue(field.Value, at + ".type");
                        break;
                    case "function":
                        function = field.Value;
                        break;
                    default:
                        throw Refuse(at + "." + field.Name, $"Tool field '{field.Name}' is unsupported.");
                }
            }

            if (type != "function")
            {
                throw Refuse(at + ".type", $"Tool type '{type}' is unsupported; only \"function\" is accepted.");
            }

            tools.Add(Function(function ?? throw Refuse(at + ".function", "A function tool needs function."), at + ".function"));
        }

        return tools;
    }

    private static DeclaredFunction Function(JsonElement v, string at)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse(at, "function must be an object.");
        }

        string? name = null;
        string? description = null;
        JsonElement parameters = default;
        foreach (var field in Fields(v, at))
        {
            switch (field.Name)
            {
                case "name":
                    name = StringValue(field.Value, at + ".name");
                    break;
                case "description":
                    description = StringValue(field.Value, at + ".description");
                    break;
                case "parameters":
                    parameters = field.Value.ValueKind == JsonValueKind.Object
                        ? field.Value.Clone()
                        : throw Refuse(at + ".parameters", "parameters must be a JSON schema object.");
                    break;
                case "strict":
                    if (Bool(field.Value, at + ".strict"))
                    {
                        throw Refuse(at + ".strict", "strict function calling is unsupported: nothing constrains decoding to the schema (structured output, R-051).");
                    }

                    break;
                default:
                    throw Refuse(at + "." + field.Name, $"Function field '{field.Name}' is unsupported.");
            }
        }

        return new DeclaredFunction(name is { Length: > 0 } ? name : throw Refuse(at + ".name", "A function needs a non-empty name."), description, parameters);
    }

    /// <summary><c>tool_choice</c>: <c>none</c>, <c>auto</c>, <c>required</c> or a named function. The client refuses a required call.</summary>
    private static ChatToolMode ToolChoice(JsonElement v)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            return v.GetString() switch
            {
                "none" => ChatToolMode.None,
                "auto" => ChatToolMode.Auto,
                "required" => ChatToolMode.RequireAny,
                var other => throw Refuse("tool_choice", $"tool_choice '{other}' is unsupported."),
            };
        }

        if (v.ValueKind == JsonValueKind.Object
            && v.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String && type.GetString() == "function"
            && v.TryGetProperty("function", out var function) && function.ValueKind == JsonValueKind.Object
            && function.TryGetProperty("name", out var name) && name.ValueKind == JsonValueKind.String)
        {
            return ChatToolMode.RequireSpecific(name.GetString()!);
        }

        throw Refuse("tool_choice", "tool_choice must be \"none\", \"auto\", \"required\" or {\"type\":\"function\",\"function\":{\"name\":...}}.");
    }

    /// <summary>vLLM's <c>chat_template_kwargs</c>: only <c>enable_thinking</c>, a Boolean, which enables the template's thinking mode.</summary>
    private static void ChatTemplateKwargs(JsonElement v, ChatOptions options)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse("chat_template_kwargs", "chat_template_kwargs must be an object.");
        }

        foreach (var field in Fields(v, "chat_template_kwargs"))
        {
            if (field.Name != RoostChatClient.EnableThinkingProperty)
            {
                throw Refuse("chat_template_kwargs." + field.Name, $"Chat template argument '{field.Name}' is unsupported; only enable_thinking is accepted.");
            }

            (options.AdditionalProperties ??= [])[RoostChatClient.EnableThinkingProperty] = Bool(field.Value, "chat_template_kwargs.enable_thinking");
        }
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
