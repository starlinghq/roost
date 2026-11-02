using System.Collections.Immutable;
using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Roost.Generation;
using Starlings.Roost.Native;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost;

/// <summary>A validated chat request: the conversation and tools for the template, the stop strings and the output limit.</summary>
/// <param name="Messages">The conversation, with <c>Instructions</c> as a leading system message.</param>
/// <param name="Tools">The tools to render and parse calls for; empty for none (also when <c>ToolMode</c> is none).</param>
/// <param name="EnableThinking">Whether the template opens the think block and the completion is parsed for reasoning.</param>
/// <param name="StopSequences">The request's stop strings.</param>
/// <param name="StopMatcher">The matcher for them, made by the client's factory.</param>
/// <param name="MaxOutputTokens">Requested output tokens before the context cap.</param>
internal sealed record PreparedChat(
    IReadOnlyList<RoostChatMessage> Messages,
    ImmutableArray<RoostToolDefinition> Tools,
    bool EnableThinking,
    IReadOnlyList<string> StopSequences,
    IStopMatcher StopMatcher,
    int MaxOutputTokens);

/// <summary>A validated raw-text completion request: the prompt as given, the stop strings and the output limit.</summary>
/// <param name="Prompt">The prompt text, tokenised without a chat template.</param>
/// <param name="StopSequences">The request's stop strings.</param>
/// <param name="StopMatcher">The matcher for them, made by the client's factory.</param>
/// <param name="MaxOutputTokens">Requested output tokens before the context cap.</param>
internal sealed record PreparedPrompt(string Prompt, IReadOnlyList<string> StopSequences, IStopMatcher StopMatcher, int MaxOutputTokens);

/// <summary>
/// Checks a conversation and its <see cref="ChatOptions"/> against what Roost runs, before any work starts. Every option that is
/// set and not supported is refused with <see cref="RoostErrorCode.InvalidOption"/> naming it (the OpenAI parameter name, as the
/// server's 400 body does, R-037); nothing is ignored.
/// </summary>
internal static class ChatRequestBuilder
{
    private const string GreedyOnly = "M0 decodes greedily only (decision 15, R-025); sampling is unsupported until the fused sampler (R-042).";
    private const string NoTemplate = "unsupported for a raw completion: the prompt is not rendered by a chat template.";

    public static PreparedChat Build(IEnumerable<ChatMessage> messages, ChatOptions? options, RoostModelProfile model, IStopMatcherFactory stopFactory, int defaultMaxOutputTokens, GrammarTokenizer? grammar = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var (stops, maxOutput) = CheckCommon(options, model.ModelId, defaultMaxOutputTokens, grammar);
        var tools = Tools(options, model.EffectiveToolCallFormat);
        var enableThinking = Thinking(options, model.EffectiveReasoningFormat);

        // An empty Instructions string means no instructions, as null does: an empty system message would change the
        // rendered prompt without saying anything.
        string? instructions = options?.Instructions is { Length: > 0 } given ? given : null;

        var conversation = new List<RoostChatMessage>();
        if (instructions is not null)
        {
            conversation.Add(RoostChatMessage.System(instructions));
        }

        var index = 0;
        foreach (var message in messages)
        {
            ArgumentNullException.ThrowIfNull(message);
            var mapped = Map(message, index);
            if (mapped[0].Role == RoostChatRole.System && instructions is not null)
            {
                throw RoostChatException.InvalidOption("instructions", "Instructions are rendered as the system message; the conversation already has one.");
            }

            conversation.AddRange(mapped);
            index++;
        }

        if (index == 0)
        {
            throw RoostChatException.InvalidOption("messages", "The conversation is empty.");
        }

        return new PreparedChat(conversation, tools, enableThinking, stops, CreateMatcher(stopFactory, stops), maxOutput);
    }

    /// <summary>
    /// Checks a raw-text completion: the same options as a chat, except <c>Instructions</c>, tools and thinking, which a prompt
    /// without a chat template has nowhere to put.
    /// </summary>
    public static PreparedPrompt BuildRaw(string prompt, ChatOptions? options, string modelId, IStopMatcherFactory stopFactory, int defaultMaxOutputTokens, GrammarTokenizer? grammar = null)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var (stops, maxOutput) = CheckCommon(options, modelId, defaultMaxOutputTokens, grammar);
        if (options is not null)
        {
            if (options.Instructions is { Length: > 0 })
            {
                throw Refuse("instructions", "Instructions are " + NoTemplate);
            }

            if (options.Tools is { Count: > 0 })
            {
                throw Refuse("tools", "Tools are " + NoTemplate);
            }

            if (options.ToolMode is not null and not NoneChatToolMode)
            {
                throw Refuse("tool_choice", "ToolMode is " + NoTemplate);
            }

            if (options.AllowMultipleToolCalls is not null)
            {
                throw Refuse("parallel_tool_calls", "AllowMultipleToolCalls is " + NoTemplate);
            }

            if (options.AdditionalProperties is { Count: > 0 } extra)
            {
                var key = extra.Keys.First();
                throw RefuseInvariant(key, $"Additional property '{key}' is {NoTemplate}");
            }
        }

        if (prompt.Length == 0)
        {
            throw Refuse("prompt", "The prompt is empty.");
        }

        return new PreparedPrompt(prompt, stops, CreateMatcher(stopFactory, stops), maxOutput);
    }

    private static (IReadOnlyList<string> Stops, int MaxOutput) CheckCommon(ChatOptions? options, string modelId, int defaultMaxOutputTokens, GrammarTokenizer? grammar)
    {
        var maxOutput = defaultMaxOutputTokens;
        IReadOnlyList<string> stops = [];
        if (options is null)
        {
            return (stops, maxOutput);
        }

        CheckOptions(options, modelId, grammar);
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

        return (stops, maxOutput);
    }

    private static IStopMatcher CreateMatcher(IStopMatcherFactory stopFactory, IReadOnlyList<string> stops)
    {
        try
        {
            return stopFactory.Create(stops);
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The stop matcher refused the stop sequences: " + e.Message, "stop"), e);
        }
    }

    private static void CheckOptions(ChatOptions o, string modelId, GrammarTokenizer? grammar)
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

        // Seed is accepted: greedy decoding is deterministic, so the seed's contract (the same output for the same input) holds
        // without it. It changes nothing in M0; the sampler (R-042) honours it.
        if (o.FrequencyPenalty is { } f && f != 0f)
        {
            throw Refuse("frequency_penalty", "FrequencyPenalty is unsupported in M0 (the eager worker applies no penalties).");
        }

        if (o.PresencePenalty is { } pp && pp != 0f)
        {
            throw Refuse("presence_penalty", "PresencePenalty is unsupported in M0 (the eager worker applies no penalties).");
        }

        if (o.ResponseFormat is not null and not ChatResponseFormatText)
        {
            CheckResponseFormat(o.ResponseFormat, grammar);
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
            foreach (var key in extra.Keys)
            {
                if (!string.Equals(key, RoostChatClient.EnableThinkingProperty, StringComparison.Ordinal))
                {
                    throw RefuseInvariant(key, $"Additional property '{key}' is unsupported; Roost accepts only '{RoostChatClient.EnableThinkingProperty}'.");
                }
            }
        }
    }

    /// <summary>The grammar a JSON response format compiles to: its schema, or any JSON object when it has none.</summary>
    internal static GrammarSpec GrammarOf(ChatResponseFormatJson format) =>
        format.Schema is { } schema ? GrammarSpec.JsonSchema(schema.GetRawText()) : GrammarSpec.JsonObject;

    /// <summary>
    /// R-051: a JSON or JSON-schema response format is compiled with the grammar engine, so a schema it cannot enforce is the
    /// caller's 400 (S10 finding 3), and then refused, because the worker's sampler does not take token masks yet (contract C2
    /// has the grammar slot but no mask table). Any other format type is refused by name.
    /// </summary>
    private static void CheckResponseFormat(ChatResponseFormat format, GrammarTokenizer? grammar)
    {
        if (format is not ChatResponseFormatJson json)
        {
            throw Refuse("response_format", $"Response format {format.GetType().Name} is unsupported; Roost accepts text, JSON and JSON-schema formats.");
        }

        if (json.Schema is { ValueKind: not JsonValueKind.Object } schema)
        {
            throw Refuse("response_format", $"The JSON schema must be an object; got {schema.ValueKind}.");
        }

        if (grammar is null)
        {
            throw Refuse("response_format", "JSON and JSON-schema response formats need the grammar engine: this client has no RoostChatClientOptions.Grammar (the roost-grammar library over the model's tokenizer).");
        }

        try
        {
            grammar.Validate(GrammarOf(json));
        }
        catch (GrammarException e) when (e.IsInvalidGrammar)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The response format's schema cannot be enforced: " + e.Message, "response_format"), e);
        }
        catch (GrammarException e)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.Internal, "The grammar engine failed: " + e.Message), e);
        }

        throw Refuse("response_format", "The response format compiles but cannot be applied yet: the worker's sampler takes no token masks until the step plan (contract C2) carries them, so the output would not be constrained (R-051).");
    }

    /// <summary>
    /// The tools to render and parse: <c>AIFunction</c> declarations (name, description, JSON schema), refused when the model has
    /// no tool-call format. <c>ToolMode</c> none accepts the tools and renders none of them, so no call is parsed; requiring a
    /// call is refused because nothing constrains decoding to one (structured output, R-051).
    /// </summary>
    private static ImmutableArray<RoostToolDefinition> Tools(ChatOptions? o, RoostToolCallFormat format)
    {
        if (o is null)
        {
            return [];
        }

        if (o.ToolMode is RequiredChatToolMode)
        {
            throw Refuse("tool_choice", "Requiring a tool call (ToolMode RequireAny or RequireSpecific) is unsupported: nothing constrains decoding to a call (structured output, R-051). Use auto or none.");
        }

        if (o.AllowMultipleToolCalls == false)
        {
            throw Refuse("parallel_tool_calls", "AllowMultipleToolCalls false is unsupported: nothing constrains decoding to one call. Leave it unset or true.");
        }

        if (o.Tools is not { Count: > 0 } given)
        {
            return [];
        }

        if (format == RoostToolCallFormat.None)
        {
            throw Refuse("tools", "Tools are unsupported for this model profile: it has no tool-call format (RoostModelProfile.ToolCallFormat).");
        }

        var tools = ImmutableArray.CreateBuilder<RoostToolDefinition>(given.Count);
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var tool in given)
        {
            if (tool is not AIFunction function)
            {
                throw RefuseInvariant("tools", $"Tool '{tool?.Name}' of type {tool?.GetType().Name} is unsupported; only function declarations (AIFunction) are accepted.");
            }

            if (string.IsNullOrEmpty(function.Name) || !names.Add(function.Name))
            {
                throw RefuseInvariant("tools", $"Tool names must be non-empty and distinct; got '{function.Name}'.");
            }

            string? parameters = function.JsonSchema.ValueKind switch
            {
                JsonValueKind.Object => function.JsonSchema.GetRawText(),
                JsonValueKind.Undefined or JsonValueKind.Null => null,
                _ => throw RefuseInvariant("tools", $"The parameters schema of tool '{function.Name}' must be a JSON object."),
            };
            tools.Add(new RoostToolDefinition(function.Name, function.Description ?? string.Empty, parameters));
        }

        return o.ToolMode is NoneChatToolMode ? [] : tools.MoveToImmutable();
    }

    /// <summary>Reads <see cref="RoostChatClient.EnableThinkingProperty"/>: a Boolean, refused when the model has no reasoning format.</summary>
    private static bool Thinking(ChatOptions? o, RoostReasoningFormat format)
    {
        if (o?.AdditionalProperties is null || !o.AdditionalProperties.TryGetValue(RoostChatClient.EnableThinkingProperty, out var value))
        {
            return false;
        }

        bool enabled = value switch
        {
            bool b => b,
            JsonElement { ValueKind: JsonValueKind.True } => true,
            JsonElement { ValueKind: JsonValueKind.False } => false,
            _ => throw Refuse(RoostChatClient.EnableThinkingProperty, "enable_thinking must be a Boolean."),
        };
        if (enabled && format == RoostReasoningFormat.None)
        {
            throw Refuse(RoostChatClient.EnableThinkingProperty, "Thinking is unsupported for this model profile: it has no reasoning format (RoostModelProfile.ReasoningFormat).");
        }

        return enabled;
    }

    /// <summary>
    /// Maps one message. A tool message becomes one template message per result (each <c>FunctionResultContent</c>, or its text
    /// as one result), as the Qwen templates take one tool message per call.
    /// </summary>
    private static List<RoostChatMessage> Map(ChatMessage message, int index)
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
            role = RoostChatRole.Tool;
        }
        else
        {
            throw RefuseInvariant(param + ".role", $"Role '{message.Role}' is unsupported.");
        }

        if (!string.IsNullOrEmpty(message.AuthorName))
        {
            throw Refuse(param + ".name", "Message author names are unsupported: the chat template has no field for them.");
        }

        if (message.AdditionalProperties is { Count: > 0 } messageExtra)
        {
            throw RefuseInvariant(param + ".additional_properties", $"Message additional property '{messageExtra.Keys.First()}' is unsupported.");
        }

        var text = new System.Text.StringBuilder();
        string? reasoning = null;
        var calls = ImmutableArray.CreateBuilder<RoostToolCall>();
        var results = new List<RoostChatMessage>();
        foreach (var content in message.Contents)
        {
            if (content.Annotations is { Count: > 0 })
            {
                throw Refuse(param + ".annotations", "Content annotations (citations and the like) are unsupported: the chat template has no field for them.");
            }

            if (content.AdditionalProperties is { Count: > 0 } contentExtra)
            {
                throw RefuseInvariant(param + ".content", $"Content additional property '{contentExtra.Keys.First()}' is unsupported.");
            }

            switch (content)
            {
                case TextContent t:
                    text.Append(t.Text);
                    break;
                case TextReasoningContent r when role == RoostChatRole.Assistant:
                    reasoning = (reasoning ?? string.Empty) + r.Text;
                    break;
                case FunctionCallContent call when role == RoostChatRole.Assistant:
                    calls.Add(new RoostToolCall(call.Name, ArgumentsJson(call, param)));
                    break;
                case FunctionResultContent result when role == RoostChatRole.Tool:
                    results.Add(RoostChatMessage.Tool(ResultText(result.Result, param)));
                    break;
                default:
                    throw RefuseInvariant(param + ".content", $"Content of type {content.GetType().Name} is unsupported on a {message.Role} message; accepted: text, reasoning and function calls on assistant messages, function results on tool messages.");
            }
        }

        if (role != RoostChatRole.Tool)
        {
            return [new RoostChatMessage(role, text.ToString(), reasoning, calls.Count == 0 ? default : calls.ToImmutable())];
        }

        if (results.Count > 0 && text.Length > 0)
        {
            throw Refuse(param + ".content", "A tool message holds function results or text, not both.");
        }

        return results.Count > 0 ? results : [RoostChatMessage.Tool(text.ToString())];
    }

    /// <summary>A call's arguments as a JSON object, serialized with Microsoft.Extensions.AI's default options.</summary>
    private static string ArgumentsJson(FunctionCallContent call, string param)
    {
        if (call.Arguments is null)
        {
            return "{}";
        }

        try
        {
            return JsonSerializer.Serialize(call.Arguments, AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(IDictionary<string, object?>)));
        }
        catch (Exception e) when (e is NotSupportedException or JsonException or InvalidOperationException)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The arguments of function call '" + call.Name + "' cannot be serialized to JSON: " + e.Message, param + ".content"), e);
        }
    }

    /// <summary>A function result as the tool message's text: a string as is, anything else as JSON.</summary>
    private static string ResultText(object? result, string param)
    {
        switch (result)
        {
            case null:
                return string.Empty;
            case string s:
                return s;
            case JsonElement { ValueKind: JsonValueKind.String } e:
                return e.GetString()!;
            case JsonElement e:
                return e.GetRawText();
        }

        try
        {
            return JsonSerializer.Serialize(result, AIJsonUtilities.DefaultOptions.GetTypeInfo(result.GetType()));
        }
        catch (Exception e) when (e is NotSupportedException or JsonException or InvalidOperationException)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "A function result cannot be serialized to JSON: " + e.Message, param + ".content"), e);
        }
    }

    private static RoostChatException RefuseInvariant(string param, FormattableString message) =>
        RoostChatException.InvalidOption(param, FormattableString.Invariant(message));

    private static RoostChatException Refuse(string param, string message) => RoostChatException.InvalidOption(param, message);
}
