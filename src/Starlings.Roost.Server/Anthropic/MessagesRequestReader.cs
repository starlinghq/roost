using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using static Starlings.Roost.Server.Routing.RequestFields;

namespace Starlings.Roost.Server.Anthropic;

/// <summary>A checked <c>POST /v1/messages</c> (or <c>count_tokens</c>) body.</summary>
/// <param name="Messages">The conversation, user and assistant turns.</param>
/// <param name="Options">The options for <see cref="RoostChatClient"/>; <c>system</c> is <see cref="ChatOptions.Instructions"/>.</param>
/// <param name="Stream">Whether to answer with Anthropic's server-sent events.</param>
internal sealed record MessagesRequest(IReadOnlyList<ChatMessage> Messages, ChatOptions Options, bool Stream);

/// <summary>
/// Reads Anthropic Messages API bodies field by field, as <c>OpenAiRequestReader</c> reads OpenAI's: every field is mapped,
/// accepted at a value that changes nothing, or refused naming it; an unknown field is refused. <c>system</c> becomes the chat
/// template's system message, <c>stop_sequences</c>, <c>temperature</c>, <c>top_p</c> and <c>top_k</c> pass to
/// <see cref="RoostChatClient"/> (which refuses sampling in M0, naming the field), and only text content blocks are accepted. An
/// absent <c>temperature</c> means greedy decoding, not Anthropic's default of 1.0, as on the OpenAI routes.
/// </summary>
internal static class MessagesRequestReader
{
    /// <summary>Reads a <c>POST /v1/messages</c> body; <paramref name="countTokens"/> reads <c>count_tokens</c>' body (no <c>max_tokens</c>, <c>stream</c> or sampling).</summary>
    public static MessagesRequest Read(JsonElement body, bool countTokens = false)
    {
        var options = new ChatOptions();
        List<ChatMessage>? messages = null;
        string? model = null;
        int? maxTokens = null;
        bool stream = false;
        foreach (var field in Fields(body))
        {
            var v = field.Value;
            string name = field.Name;
            switch (name)
            {
                case "model":
                    model = StringValue(v, name);
                    break;
                case "messages":
                    messages = Messages(v);
                    break;
                case "system":
                    options.Instructions = Text(v, "system");
                    break;
                case "tools":
                    if (v.ValueKind != JsonValueKind.Array || v.GetArrayLength() != 0)
                    {
                        throw Refuse(name, "Tools are unsupported: tool calling is not built yet (R-053).");
                    }

                    break;
                case "tool_choice":
                    if (v.ValueKind != JsonValueKind.Object || !v.TryGetProperty("type", out var choice) || choice.ValueKind != JsonValueKind.String
                        || choice.GetString() != "none" || v.EnumerateObject().Count() != 1)
                    {
                        throw Refuse(name, "tool_choice is unsupported except {\"type\":\"none\"}: tools are unsupported.");
                    }

                    break;
                case "thinking":
                    if (v.ValueKind != JsonValueKind.Object || !v.TryGetProperty("type", out var thinking) || thinking.ValueKind != JsonValueKind.String
                        || thinking.GetString() != "disabled" || v.EnumerateObject().Count() != 1)
                    {
                        throw Refuse(name, "Extended thinking is unsupported: reasoning output is not built yet (R-053). Only {\"type\":\"disabled\"} is accepted.");
                    }

                    break;
                case "max_tokens" when !countTokens:
                    maxTokens = PositiveInt(v, name);
                    break;
                case "stream" when !countTokens:
                    stream = Bool(v, name);
                    break;
                case "stop_sequences" when !countTokens:
                    options.StopSequences = Strings(v, name);
                    break;
                case "temperature" when !countTokens:
                    options.Temperature = Number(v, name);
                    break;
                case "top_p" when !countTokens:
                    options.TopP = Number(v, name);
                    break;
                case "top_k" when !countTokens:
                    options.TopK = Int(v, name);
                    break;
                case "metadata" when !countTokens:
                    Metadata(v);
                    break;
                case "service_tier" when !countTokens:
                    if (StringValue(v, name) is not ("auto" or "standard_only"))
                    {
                        throw Refuse(name, "service_tier is unsupported except \"auto\" and \"standard_only\": the server has one tier.");
                    }

                    break;
                default:
                    // container, mcp_servers, context_management and anything else.
                    throw Unknown(name);
            }
        }

        options.ModelId = model ?? throw Refuse("model", "model is required.");
        if (messages is null)
        {
            throw Refuse("messages", "messages is required.");
        }

        if (!countTokens)
        {
            options.MaxOutputTokens = maxTokens ?? throw Refuse("max_tokens", "max_tokens is required.");
        }

        return new MessagesRequest(messages, options, stream);
    }

    private static List<ChatMessage> Messages(JsonElement v)
    {
        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse("messages", "messages must be an array.");
        }

        var list = new List<ChatMessage>();
        foreach (var item in v.EnumerateArray())
        {
            string at = Inv($"messages[{list.Count}]");
            if (item.ValueKind != JsonValueKind.Object)
            {
                throw Refuse(at, "A message must be an object.");
            }

            ChatRole? role = null;
            string? text = null;
            foreach (var field in Fields(item, at))
            {
                string param = at + "." + field.Name;
                switch (field.Name)
                {
                    case "role":
                        role = StringValue(field.Value, param) switch
                        {
                            "user" => ChatRole.User,
                            "assistant" => ChatRole.Assistant,
                            var other => throw Refuse(param, $"Role '{other}' is unsupported; Anthropic messages are 'user' or 'assistant' (the system prompt is 'system')."),
                        };
                        break;
                    case "content":
                        text = Text(field.Value, param);
                        break;
                    default:
                        throw Refuse(param, $"Message field '{field.Name}' is unsupported.");
                }
            }

            list.Add(new ChatMessage(role ?? throw Refuse(at + ".role", "role is required."), text ?? throw Refuse(at + ".content", "content is required.")));
        }

        if (list.Count > 0 && list[^1].Role == ChatRole.Assistant)
        {
            throw Refuse(Inv($"messages[{list.Count - 1}].role"), "A final assistant message (a prefilled reply) is unsupported: the chat template always opens a new assistant turn.");
        }

        return list;
    }

    /// <summary>A string, or an array of text blocks joined with newlines; any other block type is refused naming it.</summary>
    private static string Text(JsonElement v, string param)
    {
        if (v.ValueKind == JsonValueKind.String)
        {
            return v.GetString()!;
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse(param, $"{param} must be a string or an array of content blocks.");
        }

        var text = new StringBuilder();
        var index = 0;
        foreach (var block in v.EnumerateArray())
        {
            string at = Inv($"{param}[{index}]");
            if (block.ValueKind != JsonValueKind.Object)
            {
                throw Refuse(at, "A content block must be an object.");
            }

            string? type = null;
            string? blockText = null;
            foreach (var field in Fields(block, at))
            {
                switch (field.Name)
                {
                    case "type":
                        type = StringValue(field.Value, at + ".type");
                        break;
                    case "text":
                        blockText = StringValue(field.Value, at + ".text");
                        break;
                    case "cache_control":
                        // A prompt-caching hint; it changes no output. Only Anthropic's one type is accepted.
                        if (field.Value.ValueKind != JsonValueKind.Object || !field.Value.TryGetProperty("type", out var cache)
                            || cache.ValueKind != JsonValueKind.String || cache.GetString() != "ephemeral")
                        {
                            throw Refuse(at + ".cache_control", "cache_control must be {\"type\":\"ephemeral\"}.");
                        }

                        break;
                    case "citations":
                        if (field.Value.ValueKind != JsonValueKind.Array || field.Value.GetArrayLength() != 0)
                        {
                            throw Refuse(at + ".citations", "Citations are unsupported.");
                        }

                        break;
                    default:
                        throw Refuse(at + "." + field.Name, $"Content block field '{field.Name}' is unsupported; only text blocks are accepted.");
                }
            }

            if (type != "text")
            {
                throw Refuse(at + ".type", $"Content block type '{type}' is unsupported; only text blocks are accepted.");
            }

            if (index > 0)
            {
                text.Append('\n');
            }

            text.Append(blockText ?? throw Refuse(at + ".text", "A text block needs text."));
            index++;
        }

        return text.ToString();
    }

    private static List<string> Strings(JsonElement v, string param)
    {
        if (v.ValueKind != JsonValueKind.Array)
        {
            throw Refuse(param, $"{param} must be an array of strings.");
        }

        return [.. v.EnumerateArray().Select(item => item.ValueKind == JsonValueKind.String ? item.GetString()! : throw Refuse(param, $"{param} must be an array of strings."))];
    }

    private static void Metadata(JsonElement v)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Refuse("metadata", "metadata must be an object.");
        }

        foreach (var field in Fields(v, "metadata"))
        {
            // An end-user identifier for the caller's own records; it changes nothing about the output. Checked, not used.
            _ = field.Name == "user_id"
                ? StringValue(field.Value, "metadata.user_id")
                : throw Refuse("metadata." + field.Name, $"Metadata field '{field.Name}' is unsupported.");
        }
    }
}
