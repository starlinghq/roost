using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Tests;

/// <summary>Every unsupported option is refused by name before any work starts; nothing is silently ignored.</summary>
public sealed class OptionValidationTests : IAsyncLifetime
{
    private static readonly ChatMessage[] Hi = [new(ChatRole.User, "Hi")];
    private readonly List<ScriptedExecutor> _executors = [];
    private RoostChatClient? _client;

    public static TheoryData<string, string> RefusedOptions() => new()
    {
        { "temperature", "temperature" },
        { "top_p", "top_p" },
        { "top_k", "top_k" },
        { "frequency_penalty", "frequency_penalty" },
        { "presence_penalty", "presence_penalty" },
        { "tools", "tools" },
        { "tool_mode", "tool_choice" },
        { "allow_multiple_tool_calls", "parallel_tool_calls" },
        { "json", "response_format" },
        { "json_schema", "response_format" },
        { "conversation_id", "conversation_id" },
        { "model", "model" },
        { "raw_representation_factory", "raw_representation_factory" },
        { "additional_properties", "logit_bias" },
        { "max_tokens_zero", "max_tokens" },
        { "empty_stop", "stop" },
    };

    public async ValueTask InitializeAsync() =>
        _client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") }, _executors);

    public async ValueTask DisposeAsync()
    {
        if (_client is not null)
        {
            await _client.DisposeAsync();
        }
    }

    [Theory]
    [MemberData(nameof(RefusedOptions))]
    public void AnUnsupportedOptionIsRefusedAtTheCallNamingIt(string option, string param)
    {
        var options = Make(option);

        var e = Assert.Throws<RoostChatException>(() => _client!.GetStreamingResponseAsync(Hi, options, TestContext.Current.CancellationToken));

        Assert.Equal(RoostErrorCode.InvalidOption, e.Error.Code);
        Assert.Equal(param, e.Error.Param);
        Assert.Equal(400, e.Error.HttpStatus);
        Assert.Empty(_executors.Single().Steps);
    }

    [Fact]
    public async Task GreedyEquivalentValuesAreAccepted()
    {
        var options = new ChatOptions
        {
            Temperature = 0f,
            Seed = 42, // accepted: greedy decoding is deterministic, so a seed changes nothing
            TopP = 1f,
            TopK = 1,
            FrequencyPenalty = 0f,
            PresencePenalty = 0f,
            ToolMode = ChatToolMode.None,
            ResponseFormat = ChatResponseFormat.Text,
            ModelId = TestRig.ModelId,
            StopSequences = [],
            Tools = [],
        };

        var response = await _client!.GetResponseAsync(Hi, options, TestContext.Current.CancellationToken);

        Assert.Equal("ok", response.Text);
    }

    [Theory]
    [InlineData("tool_message")]
    [InlineData("image")]
    [InlineData("function_call")]
    [InlineData("empty")]
    [InlineData("author_name")]
    [InlineData("message_properties")]
    [InlineData("content_properties")]
    [InlineData("annotations")]
    public void UnsupportedMessagesAreRefused(string kind)
    {
        ChatMessage[] messages = kind switch
        {
            "tool_message" => [new ChatMessage(ChatRole.User, "Hi"), new ChatMessage(ChatRole.Tool, "42")],
            "image" => [new ChatMessage(ChatRole.User, [new TextContent("what is this"), new DataContent(new byte[] { 1, 2, 3 }, "image/png")])],
            "function_call" => [new ChatMessage(ChatRole.User, "Hi"), new ChatMessage(ChatRole.Assistant, [new FunctionCallContent("c1", "f")])],
            "author_name" => [new ChatMessage(ChatRole.User, "Hi") { AuthorName = "alice" }],
            "message_properties" => [new ChatMessage(ChatRole.User, "Hi") { AdditionalProperties = new AdditionalPropertiesDictionary { ["x"] = 1 } }],
            "content_properties" => [new ChatMessage(ChatRole.User, [new TextContent("Hi") { AdditionalProperties = new AdditionalPropertiesDictionary { ["x"] = 1 } }])],
            "annotations" => [new ChatMessage(ChatRole.User, [new TextContent("Hi") { Annotations = [new CitationAnnotation { Title = "t" }] }])],
            _ => [],
        };

        var e = Assert.Throws<RoostChatException>(() => _client!.GetStreamingResponseAsync(messages, cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal(RoostErrorCode.InvalidOption, e.Error.Code);
        Assert.StartsWith("messages", e.Error.Param, StringComparison.Ordinal);
    }

    private static ChatOptions Make(string option) => option switch
    {
        "temperature" => new ChatOptions { Temperature = 0.7f },
        "top_p" => new ChatOptions { TopP = 0.9f },
        "top_k" => new ChatOptions { TopK = 40 },
        "frequency_penalty" => new ChatOptions { FrequencyPenalty = 0.5f },
        "presence_penalty" => new ChatOptions { PresencePenalty = 0.5f },
        "tools" => new ChatOptions { Tools = [AIFunctionFactory.Create(() => 42, "answer")] },
        "tool_mode" => new ChatOptions { ToolMode = ChatToolMode.RequireAny },
        "allow_multiple_tool_calls" => new ChatOptions { AllowMultipleToolCalls = true },
        "json" => new ChatOptions { ResponseFormat = ChatResponseFormat.Json },
        "json_schema" => new ChatOptions { ResponseFormat = ChatResponseFormat.ForJsonSchema(System.Text.Json.JsonDocument.Parse("{\"type\":\"object\"}").RootElement) },
        "conversation_id" => new ChatOptions { ConversationId = "c1" },
        "model" => new ChatOptions { ModelId = "another-model" },
        "raw_representation_factory" => new ChatOptions { RawRepresentationFactory = _ => new object() },
        "additional_properties" => new ChatOptions { AdditionalProperties = new AdditionalPropertiesDictionary { ["logit_bias"] = 1 } },
        "max_tokens_zero" => new ChatOptions { MaxOutputTokens = 0 },
        "empty_stop" => new ChatOptions { StopSequences = [""] },
        _ => throw new ArgumentOutOfRangeException(nameof(option)),
    };
}
