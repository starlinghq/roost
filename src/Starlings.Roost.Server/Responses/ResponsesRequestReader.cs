using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Server.Routing;
using static Starlings.Roost.Server.Routing.RequestFields;

namespace Starlings.Roost.Server.Responses;

/// <summary>A checked <c>POST /v1/responses</c> body.</summary>
/// <param name="Messages">The conversation from <c>input</c>.</param>
/// <param name="Options">The options for <see cref="RoostChatClient"/>; <c>instructions</c> is <see cref="ChatOptions.Instructions"/>.</param>
/// <param name="Stream">Whether to answer with the Responses API's server-sent events.</param>
/// <param name="Store">Whether to keep the response for <c>GET /v1/responses/{id}</c> (OpenAI's default: true).</param>
/// <param name="Background">Whether to answer at once and generate in the background (needs <paramref name="Store"/>).</param>
/// <param name="Metadata">The caller's key-value pairs, echoed back.</param>
/// <param name="User">The caller's end-user identifier, echoed back.</param>
internal sealed record ResponsesRequest(
    IReadOnlyList<ChatMessage> Messages,
    ChatOptions Options,
    bool Stream,
    bool Store,
    bool Background,
    IReadOnlyList<KeyValuePair<string, string>> Metadata,
    string? User);

/// <summary>
/// Reads OpenAI Responses API bodies field by field, as <c>OpenAiRequestReader</c> reads the chat route's: every field is
/// mapped, accepted at a value that changes nothing, or refused with a 400 naming it; an unknown field is refused. <c>input</c>
/// is a string (one user message) or a list of message items with text parts (<c>input_text</c>, and <c>output_text</c> on
/// assistant items); other item and part types are refused by path, for example <c>input[1].type</c>. An absent
/// <c>temperature</c> means greedy decoding, as on the chat route.
/// </summary>
internal static class ResponsesRequestReader
{
    public static ResponsesRequest Read(JsonElement body)
    {
        var options = new ChatOptions();
        List<ChatMessage>? messages = null;
        string? model = null;
        bool stream = false;
        bool store = true;
        bool background = false;
        string? user = null;
        var metadata = new List<KeyValuePair<string, string>>();
        foreach (var field in Fields(body))
        {
            var v = field.Value;
            string name = field.Name;
            switch (name)
            {
                case "model":
                    model = StringValue(v, name);
                    break;
                case "input":
                    messages = Input(v);
                    break;
                case "instructions":
                    options.Instructions = StringValue(v, name);
                    break;
                case "max_output_tokens":
                    options.MaxOutputTokens = PositiveInt(v, name);
                    break;
                case "temperature":
                    options.Temperature = Number(v, name);
                    break;
                case "top_p":
                    options.TopP = Number(v, name);
                    break;
                case "stream":
                    stream = Bool(v, name);
                    break;
                case "store":
                    store = Bool(v, name);
                    break;
                case "background":
                    background = Bool(v, name);
                    break;
                case "metadata":
                    Metadata(v, metadata);
                    break;
                case "user":
                    user = StringValue(v, name);
                    break;
                case "safety_identifier":
                case "prompt_cache_key":
                    // Identifiers for the caller's records and a caching hint; they change nothing about the output. Checked, not used.
                    StringValue(v, name);
                    break;
                case "service_tier":
                    if (StringValue(v, name) is not ("auto" or "default"))
                    {
                        throw Refuse(name, "service_tier is unsupported except \"auto\" and \"default\": the server has one tier.");
                    }

                    break;
                case "parallel_tool_calls":
                    // Without tools there are no tool calls to run in parallel; either value changes nothing.
                    Bool(v, name);
                    break;
                case "tools":
                    if (v.ValueKind != JsonValueKind.Array || v.GetArrayLength() != 0)
                    {
                        throw Refuse(name, "Tools are unsupported: tool calling is not built yet (R-053).");
                    }

                    break;
                case "tool_choice":
                    if (v.ValueKind != JsonValueKind.String || v.GetString() != "none")
                    {
                        throw Refuse(name, "tool_choice is unsupported except \"none\": tools are unsupported.");
                    }

                    break;
                case "text":
                    TextConfig(v, options);
                    break;
                case "truncation":
                    if (StringValue(v, name) != "disabled")
                    {
                        throw Refuse(name, "truncation is unsupported except \"disabled\": a prompt longer than the context is refused.");
                    }

                    break;
                case "include":
                    if (v.ValueKind != JsonValueKind.Array || v.GetArrayLength() != 0)
                    {
                        throw Refuse(name, "include is unsupported: there is no extra output to include.");
                    }

                    break;
                case "reasoning":
                    throw Refuse(name, "Reasoning options are unsupported: reasoning output is not built yet (R-053).");
                case "previous_response_id":
                case "conversation":
                    throw Refuse(name, $"'{name}' is unsupported: Roost does not carry conversation state between responses (conversation_state); send the whole conversation in input.");
                case "prompt":
                    throw Refuse(name, "Prompt templates are unsupported.");
                case "top_logprobs":
                    throw Refuse(name, "Log probabilities are unsupported.");
                case "max_tool_calls":
                    throw Refuse(name, "max_tool_calls is unsupported: tools are unsupported.");
                case "stream_options":
                    throw Refuse(name, "stream_options is unsupported.");
                default:
                    throw Unknown(name);
            }
        }

        options.ModelId = model ?? throw Refuse("model", "model is required.");
        if (messages is null)
        {
            throw Refuse("input", "input is required.");
        }

        if (background && !store)
        {
            throw Refuse("background", "background needs store to be true: a background response is read back with GET /v1/responses/{id}.");
        }

        if (background && stream)
        {
            throw Refuse("background", "background with stream is unsupported; poll GET /v1/responses/{id} instead.");
        }

        return new ResponsesRequest(messages, options, stream, store, background, metadata, user);
    }

    private static List<ChatMessage> Input(JsonElement v)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            return [new ChatMessage(ChatRole.User, v.GetString())];
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse("input", "input must be a string or an array of input items.");
        }

        var list = new List<ChatMessage>();
        foreach (var item in v.EnumerateArray())
        {
            list.Add(Item(item, Inv($"input[{list.Count}]")));
        }

        return list;
    }

    private static ChatMessage Item(JsonElement v, string at)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse(at, "An input item must be an object.");
        }

        ChatRole? role = null;
        JsonElement? content = null;
        foreach (var field in Fields(v, at))
        {
            string param = at + "." + field.Name;
            switch (field.Name)
            {
                case "type":
                    if (StringValue(field.Value, param) != "message")
                    {
                        throw Refuse(param, $"Input item type '{field.Value.GetString()}' is unsupported; only message items are accepted.");
                    }

                    break;
                case "role":
                    role = StringValue(field.Value, param) switch
                    {
                        "system" or "developer" => ChatRole.System,
                        "user" => ChatRole.User,
                        "assistant" => ChatRole.Assistant,
                        var other => throw Refuse(param, $"Role '{other}' is unsupported."),
                    };
                    break;
                case "content":
                    content = field.Value;
                    break;
                case "id":
                case "status":
                    // Present on output items sent back as input; they change nothing.
                    StringValue(field.Value, param);
                    break;
                default:
                    throw Refuse(param, $"Input item field '{field.Name}' is unsupported.");
            }
        }

        var r = role ?? throw Refuse(at + ".role", "role is required.");
        var c = content ?? throw Refuse(at + ".content", "content is required.");
        return new ChatMessage(r, Content(c, at + ".content", r == ChatRole.Assistant));
    }

    private static string Content(JsonElement v, string param, bool assistant)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            return v.GetString()!;
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse(param, "content must be a string or an array of content parts.");
        }

        string textType = assistant ? "output_text" : "input_text";
        var text = new StringBuilder();
        var index = 0;
        foreach (var part in v.EnumerateArray())
        {
            string at = Inv($"{param}[{index++}]");
            if (part.ValueKind != JsonValueKind.Object)
            {
                throw Refuse(at, "A content part must be an object.");
            }

            string? type = null;
            string? partText = null;
            foreach (var field in Fields(part, at))
            {
                switch (field.Name)
                {
                    case "type":
                        type = StringValue(field.Value, at + ".type");
                        break;
                    case "text":
                        partText = StringValue(field.Value, at + ".text");
                        break;
                    case "annotations" when assistant:
                    case "logprobs" when assistant:
                        if (field.Value.ValueKind != JsonValueKind.Array || field.Value.GetArrayLength() != 0)
                        {
                            throw Refuse(at + "." + field.Name, $"'{field.Name}' on an input part is unsupported unless empty.");
                        }

                        break;
                    default:
                        throw Refuse(at + "." + field.Name, $"Content part field '{field.Name}' is unsupported; only text parts are accepted.");
                }
            }

            if (type != textType)
            {
                throw Refuse(at + ".type", $"Content part type '{type}' is unsupported here; only '{textType}' parts are accepted.");
            }

            text.Append(partText ?? throw Refuse(at + ".text", "A text part needs text."));
        }

        return text.ToString();
    }

    private static void Metadata(JsonElement v, List<KeyValuePair<string, string>> metadata)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse("metadata", "metadata must be an object of strings.");
        }

        foreach (var field in Fields(v, "metadata"))
        {
            metadata.Add(new(field.Name, StringValue(field.Value, "metadata." + field.Name)));
        }
    }

    private static void TextConfig(JsonElement v, ChatOptions options)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse("text", "text must be an object.");
        }

        foreach (var field in Fields(v, "text"))
        {
            if (field.Name != "format")
            {
                throw Refuse("text." + field.Name, $"Text option '{field.Name}' is unsupported.");
            }

            options.ResponseFormat = ResponseFormats.Read(field.Value, "text.format", flat: true);
        }
    }
}
