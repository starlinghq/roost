using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Roost.Generation;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost;

/// <summary>A validated chat request: the conversation for the template, the stop strings and the output limit.</summary>
/// <param name="Messages">The conversation, with <c>Instructions</c> as a leading system message.</param>
/// <param name="StopSequences">The request's stop strings.</param>
/// <param name="StopMatcher">The matcher for them, made by the client's factory.</param>
/// <param name="MaxOutputTokens">Requested output tokens before the context cap.</param>
/// <param name="Sampling">The sampling parameters (greedy unless the worker samples on the CPU); <c>MaxTokens</c> is set by the client.</param>
internal sealed record PreparedChat(IReadOnlyList<RoostChatMessage> Messages, IReadOnlyList<string> StopSequences, IStopMatcher StopMatcher, int MaxOutputTokens, SamplingParameters Sampling);

/// <summary>A validated raw-text completion request: the prompt as given, the stop strings and the output limit.</summary>
/// <param name="Prompt">The prompt text, tokenised without a chat template.</param>
/// <param name="StopSequences">The request's stop strings.</param>
/// <param name="StopMatcher">The matcher for them, made by the client's factory.</param>
/// <param name="MaxOutputTokens">Requested output tokens before the context cap.</param>
/// <param name="Sampling">The sampling parameters (greedy unless the worker samples on the CPU); <c>MaxTokens</c> is set by the client.</param>
internal sealed record PreparedPrompt(string Prompt, IReadOnlyList<string> StopSequences, IStopMatcher StopMatcher, int MaxOutputTokens, SamplingParameters Sampling);

/// <summary>
/// Checks a conversation and its <see cref="ChatOptions"/> against what the worker runs, before any work starts. Every option
/// that is set and not supported is refused with <see cref="Starlings.Roost.Requests.RoostErrorCode.InvalidOption"/> naming it
/// (the OpenAI parameter name, as the server's 400 body will, R-037); nothing is ignored.
/// </summary>
/// <remarks>
/// <para><b>Sampling (R-042).</b> With <see cref="RoostSampler.Cpu"/> the worker runs the CPU sampler, so <c>Temperature</c>,
/// <c>TopP</c>, <c>TopK</c>, <c>Seed</c>, <c>FrequencyPenalty</c>, <c>PresencePenalty</c> and the additional properties
/// <see cref="RoostChatClient.MinPKey"/>, <see cref="RoostChatClient.RepetitionPenaltyKey"/>,
/// <see cref="RoostChatClient.LogprobsKey"/> and <see cref="RoostChatClient.TopLogprobsKey"/> are accepted and range-checked.
/// An unset temperature is greedy (0), as on the GPU worker; under greedy decoding top-p, top-k and min-p cannot change the
/// chosen token (the most likely token is always kept), so they are accepted, while penalties still apply. With
/// <see cref="RoostSampler.GreedyOnly"/> (the CUDA worker until the GPU sampler exists) only values equivalent to greedy
/// decoding without penalties or logprobs are accepted.</para>
/// </remarks>
internal static class ChatRequestBuilder
{
    private const string MinPKey = RoostChatClient.MinPKey;
    private const string RepetitionPenaltyKey = RoostChatClient.RepetitionPenaltyKey;
    private const string LogprobsKey = RoostChatClient.LogprobsKey;
    private const string TopLogprobsKey = RoostChatClient.TopLogprobsKey;
    private const int MaxTopLogprobs = 20;

    private const string GreedyOnly = "The CUDA worker decodes greedily only (decision 15, R-025); sampling needs the GPU sampler (R-042) or a worker with the CPU sampler (RoostWorkerOptions.Sampler).";

    private static SamplingParameters Greedy { get; } = new() { Temperature = 0f };

    public static PreparedChat Build(IEnumerable<ChatMessage> messages, ChatOptions? options, string modelId, IStopMatcherFactory stopFactory, int defaultMaxOutputTokens, RoostSampler sampler = RoostSampler.GreedyOnly)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var (stops, maxOutput, sampling) = CheckCommon(options, modelId, defaultMaxOutputTokens, sampler);

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
            if (mapped.Role == RoostChatRole.System && instructions is not null)
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

        return new PreparedChat(conversation, stops, CreateMatcher(stopFactory, stops), maxOutput, sampling);
    }

    /// <summary>
    /// Checks a raw-text completion: the same options as a chat, except <c>Instructions</c>, which a prompt without a chat
    /// template has nowhere to put.
    /// </summary>
    public static PreparedPrompt BuildRaw(string prompt, ChatOptions? options, string modelId, IStopMatcherFactory stopFactory, int defaultMaxOutputTokens, RoostSampler sampler = RoostSampler.GreedyOnly)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var (stops, maxOutput, sampling) = CheckCommon(options, modelId, defaultMaxOutputTokens, sampler);
        if (options?.Instructions is { Length: > 0 })
        {
            throw Refuse("instructions", "Instructions are unsupported for a raw completion: the prompt is not rendered by a chat template.");
        }

        if (prompt.Length == 0)
        {
            throw Refuse("prompt", "The prompt is empty.");
        }

        return new PreparedPrompt(prompt, stops, CreateMatcher(stopFactory, stops), maxOutput, sampling);
    }

    private static (IReadOnlyList<string> Stops, int MaxOutput, SamplingParameters Sampling) CheckCommon(ChatOptions? options, string modelId, int defaultMaxOutputTokens, RoostSampler sampler)
    {
        var maxOutput = defaultMaxOutputTokens;
        IReadOnlyList<string> stops = [];
        if (options is null)
        {
            return (stops, maxOutput, Greedy);
        }

        CheckOptions(options, modelId, sampler);
        var sampling = sampler == RoostSampler.Cpu ? Sampling(options) : Greedy;
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

        return (stops, maxOutput, sampling);
    }

    // The CPU sampler's parameters, each checked for the range CpuOps.Sample defines (and StepPlanReader enforces).
    private static SamplingParameters Sampling(ChatOptions o)
    {
        float temperature = o.Temperature ?? 0f;
        if (!float.IsFinite(temperature) || temperature < 0f)
        {
            throw RefuseInvariant("temperature", $"Temperature must be a finite number at least 0; got {temperature}.");
        }

        float topP = o.TopP ?? 1f;
        if (!float.IsFinite(topP) || topP <= 0f || topP > 1f)
        {
            throw RefuseInvariant("top_p", $"TopP must be in (0, 1]; got {topP}.");
        }

        // -1 is vLLM's spelling of "disabled", 0 Roost's.
        int topK = o.TopK ?? 0;
        if (topK < -1)
        {
            throw RefuseInvariant("top_k", $"TopK must be at least 0 (or -1 for disabled); got {topK}.");
        }

        float frequency = Penalty(o.FrequencyPenalty, "frequency_penalty");
        float presence = Penalty(o.PresencePenalty, "presence_penalty");
        var extra = o.AdditionalProperties;
        float minP = Float(extra, MinPKey) ?? 0f;
        if (!float.IsFinite(minP) || minP < 0f || minP > 1f)
        {
            throw RefuseInvariant(MinPKey, $"min_p must be in [0, 1]; got {minP}.");
        }

        float repetition = Float(extra, RepetitionPenaltyKey) ?? 1f;
        if (!float.IsFinite(repetition) || repetition <= 0f)
        {
            throw RefuseInvariant(RepetitionPenaltyKey, $"repetition_penalty must be a positive number; got {repetition}.");
        }

        bool logprobs = Bool(extra, LogprobsKey) ?? false;
        int? top = Int(extra, TopLogprobsKey);
        if (top is { } n && (n < 0 || n > MaxTopLogprobs))
        {
            throw RefuseInvariant(TopLogprobsKey, $"top_logprobs must be in [0, {MaxTopLogprobs}]; got {n}.");
        }

        if (top is > 0 && !logprobs)
        {
            throw Refuse(TopLogprobsKey, "top_logprobs needs logprobs set to true.");
        }

        return new SamplingParameters
        {
            Temperature = temperature,
            TopP = topP,
            TopK = Math.Max(0, topK),
            MinP = minP,
            RepetitionPenalty = repetition,
            PresencePenalty = presence,
            FrequencyPenalty = frequency,
            Seed = o.Seed is { } seed ? unchecked((ulong)seed) : null,
            Logprobs = logprobs,
            TopLogprobs = top ?? 0,
        };
    }

    // OpenAI's range for both penalties.
    private static float Penalty(float? value, string param)
    {
        float v = value ?? 0f;
        return float.IsFinite(v) && v >= -2f && v <= 2f ? v : throw RefuseInvariant(param, $"{param} must be in [-2, 2]; got {v}.");
    }

    private static float? Float(AdditionalPropertiesDictionary? extra, string key)
    {
        if (extra is null || !extra.TryGetValue(key, out var value) || value is null)
        {
            return null;
        }

        return value switch
        {
            float f => f,
            double d => (float)d,
            decimal m => (float)m,
            int i => i,
            long l => l,
            JsonElement { ValueKind: JsonValueKind.Number } e => (float)e.GetDouble(),
            _ => throw Refuse(key, key + " must be a number."),
        };
    }

    private static int? Int(AdditionalPropertiesDictionary? extra, string key)
    {
        if (extra is null || !extra.TryGetValue(key, out var value) || value is null)
        {
            return null;
        }

        return value switch
        {
            int i => i,
            long l when l is >= int.MinValue and <= int.MaxValue => (int)l,
            JsonElement { ValueKind: JsonValueKind.Number } e when e.TryGetInt32(out var i) => i,
            _ => throw Refuse(key, key + " must be an integer."),
        };
    }

    private static bool? Bool(AdditionalPropertiesDictionary? extra, string key)
    {
        if (extra is null || !extra.TryGetValue(key, out var value) || value is null)
        {
            return null;
        }

        return value switch
        {
            bool b => b,
            JsonElement { ValueKind: JsonValueKind.True } => true,
            JsonElement { ValueKind: JsonValueKind.False } => false,
            _ => throw Refuse(key, key + " must be a boolean."),
        };
    }

    private static IStopMatcher CreateMatcher(IStopMatcherFactory stopFactory, IReadOnlyList<string> stops)
    {
        try
        {
            return stopFactory.Create(stops);
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException)
        {
            throw new RoostChatException(new Starlings.Roost.Requests.RoostError(Starlings.Roost.Requests.RoostErrorCode.InvalidOption, "The stop matcher refused the stop sequences: " + e.Message, "stop"), e);
        }
    }

    private static void CheckOptions(ChatOptions o, string modelId, RoostSampler sampler)
    {
        if (sampler != RoostSampler.Cpu)
        {
            CheckGreedyOnly(o);
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
            foreach (var key in extra.Keys)
            {
                bool known = key is MinPKey or RepetitionPenaltyKey or LogprobsKey or TopLogprobsKey;
                if (!known)
                {
                    throw RefuseInvariant(key, $"Additional property '{key}' is unsupported; Roost accepts only {MinPKey}, {RepetitionPenaltyKey}, {LogprobsKey} and {TopLogprobsKey}, with the CPU sampler.");
                }

                if (sampler != RoostSampler.Cpu)
                {
                    throw RefuseInvariant(key, $"{key} is unsupported: {GreedyOnly}");
                }
            }
        }
    }

    // The CUDA worker's rules: values equivalent to greedy decoding without penalties or logprobs.
    private static void CheckGreedyOnly(ChatOptions o)
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
        // without it.
        if (o.FrequencyPenalty is { } f && f != 0f)
        {
            throw Refuse("frequency_penalty", "FrequencyPenalty is unsupported: the CUDA worker applies no penalties until the GPU sampler (R-042).");
        }

        if (o.PresencePenalty is { } pp && pp != 0f)
        {
            throw Refuse("presence_penalty", "PresencePenalty is unsupported: the CUDA worker applies no penalties until the GPU sampler (R-042).");
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
