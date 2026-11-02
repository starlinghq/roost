using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Tests;

/// <summary>
/// Tools and thinking through <see cref="RoostChatClient"/> end to end (R-053): the request's tools and thinking switch reach
/// the chat template, and the scripted completion comes back as <see cref="FunctionCallContent"/> and
/// <see cref="TextReasoningContent"/>. The completions are written in the Qwen3.5 template's format (see Parsing/TemplateCorpus).
/// </summary>
public sealed class ToolCallingTests
{
    // A Qwen3.5 tools prompt is about 1,500 byte tokens with the tiny tokenizer: 128 blocks of 16 give a 2,048-token context.
    private const int Blocks = 128;

    private const string WeatherCall = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

    private static readonly ChatMessage[] Ask = [new(ChatRole.User, "Weather in Paris?")];

    private static readonly AIFunction Weather = AIFunctionFactory.Create((string city, int days) => "sunny", "get_weather", "Get the weather forecast.");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static RoostToolDefinition WeatherDefinition => new(Weather.Name, Weather.Description, Weather.JsonSchema.GetRawText());

    private static Task<RoostChatClient> StartAsync(string completion, List<ScriptedExecutor>? executors = null, Func<RoostModelProfile, RoostModelProfile>? profile = null) =>
        TestRig.InProcessAsync(
            new Script { Tokens = TestRig.Encode(completion) },
            executors,
            configure: o => o with { Model = profile?.Invoke(TestRig.Profile(Blocks)) ?? TestRig.Profile(Blocks) },
            limits: TestRig.Limits(Blocks));

    private static int PromptTokens(string prompt) => TestRig.Encode(prompt).Length;

    [Fact]
    public async Task AToolCallComesBackAsFunctionCallContentWithTypedArgumentsAndFinishReasonToolCalls()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await StartAsync("I will check.\n\n" + WeatherCall, executors);

        var response = await client.GetResponseAsync(Ask, new ChatOptions { Tools = [Weather] }, Ct);

        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
        Assert.Equal("I will check.", response.Text.TrimEnd());
        var call = Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        Assert.Equal("get_weather", call.Name);
        Assert.StartsWith("call_roost-", call.CallId, StringComparison.Ordinal);
        Assert.Equal("Paris", ((JsonElement)call.Arguments!["city"]!).GetString());
        Assert.Equal(3, ((JsonElement)call.Arguments["days"]!).GetInt32()); // cast with the tool's schema (integer)
        Assert.Equal("""{"city":"Paris","days":3}""", Assert.IsType<RoostToolCall>(call.RawRepresentation).ArgumentsJson);

        // The template rendered the tools block: the prompt is the Qwen3.5 rendering with the tool definition.
        var expected = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("Weather in Paris?")], [WeatherDefinition]);
        Assert.Equal(PromptTokens(expected), response.Usage!.InputTokenCount);
        Assert.Empty(executors.Single().Rejections);
    }

    [Fact]
    public async Task StreamingReturnsEachCallOnceAndNeverItsMarkup()
    {
        await using var client = await StartAsync(WeatherCall + "\n" + WeatherCall.Replace("Paris", "Oslo", StringComparison.Ordinal));

        var updates = new List<ChatResponseUpdate>();
        await foreach (var update in client.GetStreamingResponseAsync(Ask, new ChatOptions { Tools = [Weather] }, Ct))
        {
            updates.Add(update);
        }

        var calls = updates.SelectMany(u => u.Contents).OfType<FunctionCallContent>().ToList();
        Assert.Equal(["Paris", "Oslo"], calls.Select(c => ((JsonElement)c.Arguments!["city"]!).GetString()));
        Assert.Equal(2, calls.Select(c => c.CallId).Distinct().Count());
        Assert.DoesNotContain(updates, u => u.Text.Contains("tool_call", StringComparison.Ordinal));
        Assert.Equal(ChatFinishReason.ToolCalls, updates[^1].FinishReason);
    }

    [Fact]
    public async Task ThinkingSplitsReasoningFromTheAnswer()
    {
        await using var client = await StartAsync("The user greets me.\n</think>\n\nHello!");
        var options = new ChatOptions { AdditionalProperties = new() { [RoostChatClient.EnableThinkingProperty] = true } };

        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], options, Ct);

        Assert.Equal("Hello!", response.Text);
        Assert.Equal("The user greets me.", string.Concat(response.Messages.SelectMany(m => m.Contents).OfType<TextReasoningContent>().Select(r => r.Text)));
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        var expected = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("Hi")], default, enableThinking: true);
        Assert.Equal(PromptTokens(expected), response.Usage!.InputTokenCount);
    }

    [Fact]
    public async Task ThinkingWithToolsGivesReasoningThenCalls()
    {
        await using var client = await StartAsync("Need the forecast.\n</think>\n\n" + WeatherCall);
        var options = new ChatOptions { Tools = [Weather], AdditionalProperties = new() { [RoostChatClient.EnableThinkingProperty] = true } };

        var contents = new List<AIContent>();
        await foreach (var update in client.GetStreamingResponseAsync(Ask, options, Ct))
        {
            contents.AddRange(update.Contents.Where(c => c is not UsageContent));
        }

        Assert.All(contents.TakeWhile(c => c is TextReasoningContent), c => Assert.IsType<TextReasoningContent>(c));
        Assert.Equal("Need the forecast.", string.Concat(contents.OfType<TextReasoningContent>().Select(r => r.Text)));
        Assert.IsType<FunctionCallContent>(Assert.Single(contents, c => c is not TextReasoningContent));
    }

    [Fact]
    public async Task ACallAndItsResultInTheConversationAreRenderedByTheTemplate()
    {
        await using var client = await StartAsync("It is sunny in Paris.");
        ChatMessage[] conversation =
        [
            new(ChatRole.User, "Weather in Paris?"),
            new(ChatRole.Assistant, [new FunctionCallContent("call_1", "get_weather", new Dictionary<string, object?> { ["city"] = "Paris", ["days"] = 3 })]),
            new(ChatRole.Tool, [new FunctionResultContent("call_1", "sunny")]),
        ];

        var response = await client.GetResponseAsync(conversation, new ChatOptions { Tools = [Weather] }, Ct);

        Assert.Equal("It is sunny in Paris.", response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        var expected = Qwen35RoostChatTemplate.Instance.Render(
            [
                RoostChatMessage.User("Weather in Paris?"),
                RoostChatMessage.Assistant(string.Empty, null, [new RoostToolCall("get_weather", """{"city":"Paris","days":3}""")]),
                RoostChatMessage.Tool("sunny"),
            ],
            [WeatherDefinition]);
        Assert.Equal(PromptTokens(expected), response.Usage!.InputTokenCount);
    }

    [Fact]
    public async Task ToolModeNoneRendersNoToolsAndParsesNoCalls()
    {
        await using var client = await StartAsync(WeatherCall);

        var response = await client.GetResponseAsync(Ask, new ChatOptions { Tools = [Weather], ToolMode = ChatToolMode.None }, Ct);

        Assert.Equal(WeatherCall, response.Text);
        Assert.Empty(response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal(PromptTokens(Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("Weather in Paris?")])), response.Usage!.InputTokenCount);
    }

    [Fact]
    public async Task WithoutToolsToolMarkupIsPlainText()
    {
        await using var client = await StartAsync(WeatherCall);

        var response = await client.GetResponseAsync(Ask, cancellationToken: Ct);

        Assert.Equal(WeatherCall, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }

    [Fact]
    public async Task TheQwen3TemplateKeepsRefusingToolsAndThinking()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await StartAsync("ok", executors, p => p with { ChatTemplate = Qwen3RoostChatTemplate.Instance });

        var tools = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Ask, new ChatOptions { Tools = [Weather] }, Ct));
        var thinking = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(
            Ask, new ChatOptions { AdditionalProperties = new() { [RoostChatClient.EnableThinkingProperty] = true } }, Ct));

        Assert.Equal((RoostErrorCode.InvalidOption, "tools"), (tools.Error.Code, tools.Error.Param));
        Assert.Equal((RoostErrorCode.InvalidOption, RoostChatClient.EnableThinkingProperty), (thinking.Error.Code, thinking.Error.Param));
        Assert.Equal(RoostToolCallFormat.Hermes, client.Model.EffectiveToolCallFormat);
        Assert.Empty(executors.Single().Steps);
    }

    [Fact]
    public async Task AProfileWithoutFormatsRefusesToolsAndThinkingAtTheCall()
    {
        await using var client = await StartAsync("ok", profile: p => p with { ToolCallFormat = RoostToolCallFormat.None, ReasoningFormat = RoostReasoningFormat.None });

        var tools = Assert.Throws<RoostChatException>(() => client.GetStreamingResponseAsync(Ask, new ChatOptions { Tools = [Weather] }, Ct));
        var thinking = Assert.Throws<RoostChatException>(() => client.GetStreamingResponseAsync(
            Ask, new ChatOptions { AdditionalProperties = new() { [RoostChatClient.EnableThinkingProperty] = true } }, Ct));

        Assert.Equal("tools", tools.Error.Param);
        Assert.Equal(RoostChatClient.EnableThinkingProperty, thinking.Error.Param);
    }

    [Fact]
    public void ARawCompletionRefusesToolsAndThinking()
    {
        var builder = () => ChatRequestBuilder.BuildRaw("x", new ChatOptions { Tools = [Weather] }, TestRig.ModelId, new SimpleStopMatcherFactory(), 8);
        var thinking = () => ChatRequestBuilder.BuildRaw("x", new ChatOptions { AdditionalProperties = new() { [RoostChatClient.EnableThinkingProperty] = true } }, TestRig.ModelId, new SimpleStopMatcherFactory(), 8);

        Assert.Equal("tools", Assert.Throws<RoostChatException>(builder).Error.Param);
        Assert.Equal(RoostChatClient.EnableThinkingProperty, Assert.Throws<RoostChatException>(thinking).Error.Param);
    }
}
