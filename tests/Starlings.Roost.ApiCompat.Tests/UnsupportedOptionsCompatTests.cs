using OpenAI.Chat;
using OpenAI.Responses;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.ApiCompat.Tests.Support;
using static Starlings.Roost.ApiCompat.Tests.Support.Refusals;

namespace Starlings.Roost.ApiCompat.Tests;

/// <summary>
/// Every option the official client can set that Roost does not support is a 400 <c>invalid_option</c> whose <c>param</c> names
/// it, as the client sends it, before any work reaches the worker. Options set to a value that changes nothing (for example
/// <c>seed</c> under greedy decoding, <c>user</c>, <c>temperature</c> 0) are accepted and covered by the success tests.
/// </summary>
public sealed class UnsupportedOptionsCompatTests : IAsyncLifetime
{
    private static readonly Dictionary<string, Action<ChatCompletionOptions>> s_chat = new()
    {
        ["temperature"] = o => o.Temperature = 0.7f,
        ["top_p"] = o => o.TopP = 0.9f,
        ["frequency_penalty"] = o => o.FrequencyPenalty = 0.5f,
        ["presence_penalty"] = o => o.PresencePenalty = 0.5f,
        ["logprobs"] = o => o.IncludeLogProbabilities = true,
        ["top_logprobs"] = o => o.TopLogProbabilityCount = 2,
        ["logit_bias"] = o => o.LogitBiases[50] = 1,
        ["store"] = o => o.StoredOutputEnabled = true,
        ["reasoning_effort"] = o => o.ReasoningEffortLevel = ChatReasoningEffortLevel.Low,
        ["response_format"] = o => o.ResponseFormat = ChatResponseFormat.CreateJsonObjectFormat(),
        ["response_format/json_schema"] = o => o.ResponseFormat = ChatResponseFormat.CreateJsonSchemaFormat("answer", BinaryData.FromString("""{"type":"object"}""")),
        ["modalities"] = o => o.ResponseModalities = ChatResponseModalities.Text | ChatResponseModalities.Audio,
        ["metadata"] = o => o.Metadata["k"] = "v",
        ["service_tier"] = o => o.ServiceTier = ChatServiceTier.Flex,
        ["safety_identifier"] = o => o.SafetyIdentifier = "caller-1",
        ["web_search_options"] = o => o.WebSearchOptions = new ChatWebSearchOptions(),
        ["prediction"] = o => o.OutputPrediction = ChatOutputPrediction.CreateStaticContentPrediction("Hello"),
        ["n"] = o => o.Patch.Set("$.n"u8, 2),
        ["top_k"] = o => o.Patch.Set("$.top_k"u8, 40),
        ["chat_template_kwargs.add_generation_prompt"] = o => o.Patch.Set("$.chat_template_kwargs.add_generation_prompt"u8, false),
    };

    private static readonly Dictionary<string, Action<CreateResponseOptions>> s_responses = new()
    {
        ["temperature"] = o => o.Temperature = 0.7f,
        ["top_p"] = o => o.TopP = 0.9f,
        ["tools"] = o => o.Tools.Add(ResponseTool.CreateFunctionTool("f", BinaryData.FromString("""{"type":"object"}"""), false)),
        ["tool_choice"] = o => o.ToolChoice = ResponseToolChoice.CreateAutoChoice(),
        ["reasoning"] = o => o.ReasoningOptions = new ResponseReasoningOptions { ReasoningEffortLevel = ResponseReasoningEffortLevel.Low },
        ["previous_response_id"] = o => o.PreviousResponseId = "resp_1",
        ["text.format"] = o => o.TextOptions = new ResponseTextOptions { TextFormat = ResponseTextFormat.CreateJsonObjectFormat() },
        ["truncation"] = o => o.TruncationMode = ResponseTruncationMode.Auto,
        ["include"] = o => o.IncludedProperties.Add(IncludedResponseProperty.ReasoningEncryptedContent),
        ["top_logprobs"] = o => o.TopLogProbabilityCount = 2,
        ["max_tool_calls"] = o => o.MaxToolCallCount = 2,
        ["service_tier"] = o => o.ServiceTier = ResponseServiceTier.Flex,
        ["background"] = o =>
        {
            o.BackgroundModeEnabled = true;
            o.StoredOutputEnabled = false;
        },
    };

    private ApiCompatRig? _rig;

    private ApiCompatRig Rig => _rig!;

    public static TheoryData<string> ChatOptions => [.. s_chat.Keys];

    public static TheoryData<string> ResponsesOptions => [.. s_responses.Keys];

    public async ValueTask InitializeAsync() => _rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("ok") });

    public async ValueTask DisposeAsync()
    {
        if (_rig is not null)
        {
            await _rig.DisposeAsync();
        }
    }

    [Theory]
    [MemberData(nameof(ChatOptions))]
    public async Task AChatOptionIsRefusedByNameWholeAndStreamed(string key)
    {
        var param = key.Split('/')[0];
        var options = new ChatCompletionOptions();
        s_chat[key](options);
        ChatMessage[] ask = [new UserChatMessage("Hi")];

        await AssertRefusedAsync(() => Rig.Chat.CompleteChatAsync(ask, options, Ct), 400, "invalid_option", param);
        await AssertRefusedAsync(
            async () =>
            {
                await foreach (var _ in Rig.Chat.CompleteChatStreamingAsync(ask, options, Ct))
                {
                }
            },
            400,
            "invalid_option",
            param);
        Assert.Empty(Rig.Executor.Steps);
    }

    [Theory]
    [MemberData(nameof(ResponsesOptions))]
    public async Task AResponsesOptionIsRefusedByName(string param)
    {
        var options = new CreateResponseOptions(ApiCompatRig.ModelId, [ResponseItem.CreateUserMessageItem("Hi")]);
        s_responses[param](options);

        await AssertRefusedAsync(() => Rig.Responses.CreateResponseAsync(options, Ct), 400, "invalid_option", param);
        Assert.Empty(Rig.Executor.Steps);
    }

    [Fact]
    public async Task UnsupportedContentPartsAreRefusedByPath()
    {
        ChatMessage[] image = [new UserChatMessage(ChatMessageContentPart.CreateTextPart("What is this?"), ChatMessageContentPart.CreateImagePart(new Uri("https://example.com/a.png")))];
        await AssertRefusedAsync(() => Rig.Chat.CompleteChatAsync(image, cancellationToken: Ct), 400, "invalid_option", "messages[0].content[1].image_url");

        ChatMessage[] audio = [new UserChatMessage(ChatMessageContentPart.CreateInputAudioPart(BinaryData.FromBytes([1, 2, 3]), ChatInputAudioFormat.Wav))];
        await AssertRefusedAsync(() => Rig.Chat.CompleteChatAsync(audio, cancellationToken: Ct), 400, "invalid_option", "messages[0].content[0].input_audio");

        ChatMessage[] named = [new UserChatMessage("Hi") { ParticipantName = "ann" }];
        await AssertRefusedAsync(() => Rig.Chat.CompleteChatAsync(named, cancellationToken: Ct), 400, "invalid_option", "messages[0].name");

        var input = ResponseItem.CreateUserMessageItem([ResponseContentPart.CreateInputTextPart("What is this?"), ResponseContentPart.CreateInputImagePart(new Uri("https://example.com/a.png"))]);
        await AssertRefusedAsync(() => Rig.Responses.CreateResponseAsync(new CreateResponseOptions(ApiCompatRig.ModelId, [input]), Ct), 400, "invalid_option", "input[0].content[1].image_url");

        Assert.Empty(Rig.Executor.Steps);
    }

    [Fact]
    public async Task AnotherModelIsRefusedByName()
    {
        var other = Rig.OpenAi.GetChatClient("another-model");

        await AssertRefusedAsync(() => other.CompleteChatAsync([new UserChatMessage("Hi")], cancellationToken: Ct), 400, "invalid_option", "model");
        await AssertRefusedAsync(() => Rig.Responses.CreateResponseAsync("another-model", "Hi", cancellationToken: Ct), 400, "invalid_option", "model");
    }
}
