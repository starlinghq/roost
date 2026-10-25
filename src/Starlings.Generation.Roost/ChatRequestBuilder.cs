using System.Globalization;
using Microsoft.Extensions.AI;
using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost;

/// <summary>A validated chat request: the conversation for the template, the stop strings and the output limit.</summary>
/// <param name="Messages">The conversation, with <c>Instructions</c> as a leading system message.</param>
/// <param name="StopSequences">The request's stop strings.</param>
/// <param name="StopMatcher">The matcher for them, made by the client's factory.</param>
/// <param name="MaxOutputTokens">Requested output tokens before the context cap.</param>
internal sealed record PreparedChat(IReadOnlyList<RoostChatMessage> Messages, IReadOnlyList<string> StopSequences, IStopMatcher StopMatcher, int MaxOutputTokens);

/// <summary>
/// Checks a conversation and its <see cref="ChatOptions"/> against what M0 runs, before any work starts. Every option that is
/// set and not supported is refused with <see cref="Starlings.Roost.Requests.RoostErrorCode.InvalidOption"/> naming it (the
/// OpenAI parameter name, as the server's 400 body will, R-037); nothing is ignored.
/// </summary>
internal static class ChatRequestBuilder
{
    private const string GreedyOnly = "M0 decodes greedily only (decision 15, R-025); sampling is unsupported until the fused sampler (R-042).";

    public static PreparedChat Build(IEnumerable<ChatMessage> messages, ChatOptions? options, string modelId, IStopMatcherFactory stopFactory, int defaultMaxOutputTokens)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var maxOutput = defaultMaxOutputTokens;
        IReadOnlyList<string> stops = [];
        string? instructions = null;
        if (options is not null)
        {
            CheckOptions(options, modelId);
            if (options.MaxOutputTokens is { } max)
            {
                maxOutput = max > 0 ? max : throw RoostChatException.InvalidOption("max_tokens", string.Create(CultureInfo.InvariantCulture, $"MaxOutputTokens must be positive; got {max}."));
            }

            if (options.StopSequences is { Count: > 0 } s)
            {
                if (s.Any(string.IsNullOrEmpty))
                {
                    throw RoostChatException.InvalidOption("stop", "Stop sequences must be non-empty strings.");
                }

                stops = [.. s];
            }

            instructions = options.Instructions;
        }

        var conversation = new List<RoostChatMessage>();
        if (!string.IsNullOrEmpty(instructions))
        {
            conversation.Add(RoostChatMessage.System(instructions));
        }

        var index = 0;
        foreach (var message in messages)
        {
            ArgumentNullException.ThrowIfNull(message);
            var mapped = Map(message, index);
            if (mapped.Role == RoostChatRole.System && instructions is { Length: > 0 })
            {
                throw RoostChatException.InvalidOption("instructions", "Instructions are rendered as the system message; the conversation already has one.");
            }

            conversation.Add(mapped);
            index++;
        }

        if (index == 0)
        {
            throw RoostChatException.InvalidOption("messages", "The conversation is empty.");
        }

        IStopMatcher matcher;
        try
        {
            matcher = stopFactory.Create(stops);
        }
        catch (NotSupportedException e)
        {
            throw new RoostChatException(new Starlings.Roost.Requests.RoostError(Starlings.Roost.Requests.RoostErrorCode.InvalidOption, "Stop sequences are unsupported: " + e.Message, "stop"), e);
        }

        return new PreparedChat(conversation, stops, matcher, maxOutput);
    }

    private static void CheckOptions(ChatOptions o, string modelId)
    {
        if (o.Temperature is { } t && t != 0f)
        {
            throw RefuseInvariant("temperature", $"Temperature {t} is unsupported: {GreedyOnly} Leave it unset or pass 0.");
        }

        if (o.TopP is { } p && p != 1f)
        {
            throw RefuseInvariant("top_p", $"TopP {p} is unsupported: {GreedyOnly}");
        }

        if (o.TopK is { } k && k != 1)
        {
            throw RefuseInvariant("top_k", $"TopK {k} is unsupported: {GreedyOnly}");
        }

        if (o.Seed is not null)
        {
            throw RefuseInvariant("seed", $"Seed is unsupported: {GreedyOnly}");
        }

        if (o.FrequencyPenalty is { } f && f != 0f)
        {
            throw Refuse("frequency_penalty", "FrequencyPenalty is unsupported in M0 (the eager worker applies no penalties).");
        }

        if (o.PresencePenalty is { } pp && pp != 0f)
        {
            throw Refuse("presence_penalty", "PresencePenalty is unsupported in M0 (the eager worker applies no penalties).");
        }

        if (o.Tools is { Count: > 0 })
        {
            throw Refuse("tools", "Tools are unsupported in M0; tool calling arrives with the server in M1 (R-037).");
        }

        if (o.ToolMode is not null and not NoneChatToolMode)
        {
            throw Refuse("tool_choice", "ToolMode is unsupported in M0: tools are unsupported.");
        }

        if (o.AllowMultipleToolCalls is not null)
        {
            throw Refuse("parallel_tool_calls", "AllowMultipleToolCalls is unsupported in M0: tools are unsupported.");
        }

        if (o.ResponseFormat is not null and not ChatResponseFormatText)
        {
            throw Refuse("response_format", "JSON and JSON-schema response formats are unsupported in M0; structured output arrives in M1 (R-051).");
        }

        if (o.ConversationId is not null)
        {
            throw Refuse("conversation_id", "ConversationId is unsupported: Roost keeps no conversation state; send the whole conversation.");
        }

        if (o.ModelId is { } m && !string.Equals(m, modelId, StringComparison.Ordinal))
        {
            throw RefuseInvariant("model", $"Model '{m}' is not served by this client; it serves '{modelId}'.");
        }

        if (o.RawRepresentationFactory is not null)
        {
            throw Refuse("raw_representation_factory", "RawRepresentationFactory is unsupported: Roost has no provider-specific options object.");
        }

        if (o.AdditionalProperties is { Count: > 0 } extra)
        {
            var key = extra.Keys.First();
            throw RefuseInvariant(key, $"Additional property '{key}' is unsupported; Roost accepts no additional properties in M0.");
        }
    }

    private static RoostChatMessage Map(ChatMessage message, int index)
    {
        string param = string.Create(CultureInfo.InvariantCulture, $"messages[{index}]");
        RoostChatRole role;
        if (message.Role == ChatRole.System)
        {
            role = RoostChatRole.System;
        }
        else if (message.Role == ChatRole.User)
        {
            role = RoostChatRole.User;
        }
        else if (message.Role == ChatRole.Assistant)
        {
            role = RoostChatRole.Assistant;
        }
        else if (message.Role == ChatRole.Tool)
        {
            throw Refuse(param + ".role", "Tool messages are unsupported in M0: tools are unsupported.");
        }
        else
        {
            throw RefuseInvariant(param + ".role", $"Role '{message.Role}' is unsupported.");
        }

        var text = new System.Text.StringBuilder();
        string? reasoning = null;
        foreach (var content in message.Contents)
        {
            switch (content)
            {
                case TextContent t:
                    text.Append(t.Text);
                    break;
                case TextReasoningContent r when role == RoostChatRole.Assistant:
                    reasoning = (reasoning ?? string.Empty) + r.Text;
                    break;
                default:
                    throw RefuseInvariant(param + ".content", $"Content of type {content.GetType().Name} is unsupported in M0; only text is accepted (and reasoning text on assistant messages).");
            }
        }

        return new RoostChatMessage(role, text.ToString(), reasoning);
    }

    private static RoostChatException RefuseInvariant(string param, FormattableString message) =>
        RoostChatException.InvalidOption(param, FormattableString.Invariant(message));

    private static RoostChatException Refuse(string param, string message) => RoostChatException.InvalidOption(param, message);
}
