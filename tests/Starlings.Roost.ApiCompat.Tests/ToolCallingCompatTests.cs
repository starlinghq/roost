using System.Text.Json;
using OpenAI.Chat;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.ApiCompat.Tests.Support;
using Starlings.Roost.Generation;
using static Starlings.Roost.ApiCompat.Tests.Support.Refusals;

namespace Starlings.Roost.ApiCompat.Tests;

/// <summary>
/// Tool calls and reasoning on <c>POST /v1/chat/completions</c> through the official client: declared tools, the parsed
/// <c>tool_calls</c> (whole and streamed), and a call and its result sent back as conversation history. The scripted
/// completions use the Qwen3.5 template's format, which the server parses (R-053).
/// </summary>
public sealed class ToolCallingCompatTests
{
    private const int Blocks = 128; // a 2,048-token context: the tools prompt is about 1,500 byte tokens

    private const string WeatherCall = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    private const string Schema = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""";

    private static ChatTool Weather => ChatTool.CreateFunctionTool("get_weather", "Get the weather.", BinaryData.FromString(Schema));

    private static RoostToolDefinition RoostWeather => new("get_weather", "Get the weather.", Schema);

    private static ChatCompletionOptions WithWeather(ChatToolChoice? choice = null)
    {
        var options = new ChatCompletionOptions { ToolChoice = choice };
        options.Tools.Add(Weather);
        return options;
    }

    private static Task<ApiCompatRig> StartAsync(string completion) =>
        ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode(completion) }, attentionBlocks: Blocks);

    [Fact]
    public async Task AWholeCompletionCarriesAToolCall()
    {
        await using var rig = await StartAsync(WeatherCall);

        var options = WithWeather(ChatToolChoice.CreateAutoChoice());
        options.AllowParallelToolCalls = true;
        ChatCompletion completion = await rig.Chat.CompleteChatAsync([new UserChatMessage("Weather in Paris?")], options, Ct);

        Assert.Equal(ChatFinishReason.ToolCalls, completion.FinishReason);
        Assert.Empty(completion.Content);
        var call = Assert.Single(completion.ToolCalls);
        Assert.StartsWith("call_", call.Id, StringComparison.Ordinal);
        Assert.Equal(ChatToolCallKind.Function, call.Kind);
        Assert.Equal("get_weather", call.FunctionName);
        using var arguments = JsonDocument.Parse(call.FunctionArguments);
        Assert.Equal("Paris", arguments.RootElement.GetProperty("city").GetString());

        var prompt = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("Weather in Paris?")], [RoostWeather]);
        Assert.Equal(ApiCompatRig.Encode(prompt).Length, completion.Usage.InputTokenCount);
    }

    [Fact]
    public async Task AStreamedToolCallAssemblesFromItsUpdates()
    {
        await using var rig = await StartAsync("Look it up.\n</think>\n\n" + WeatherCall);

        var options = WithWeather();
        options.Patch.Set("$.chat_template_kwargs.enable_thinking"u8, true);
        var updates = new List<StreamingChatToolCallUpdate>();
        ChatFinishReason? finish = null;
        var content = string.Empty;
        await foreach (var update in rig.Chat.CompleteChatStreamingAsync([new UserChatMessage("Weather in Paris?")], options, Ct))
        {
            updates.AddRange(update.ToolCallUpdates);
            content += string.Concat(update.ContentUpdate.Select(p => p.Text));
            finish ??= update.FinishReason;
        }

        Assert.Equal(ChatFinishReason.ToolCalls, finish);
        Assert.Equal(string.Empty, content); // the reasoning is not content
        Assert.All(updates, u => Assert.Equal(0, u.Index));
        var first = updates[0];
        Assert.StartsWith("call_", first.ToolCallId, StringComparison.Ordinal);
        Assert.Equal("get_weather", first.FunctionName);
        var arguments = string.Concat(updates.Select(u => u.FunctionArgumentsUpdate?.ToString()));
        using var doc = JsonDocument.Parse(arguments);
        Assert.Equal("Paris", doc.RootElement.GetProperty("city").GetString());
    }

    [Fact]
    public async Task AToolCallAndItsResultSentBackAreRendered()
    {
        await using var rig = await StartAsync("Sunny.");

        ChatCompletion completion = await rig.Chat.CompleteChatAsync(
            [
                new UserChatMessage("Weather in Paris?"),
                new AssistantChatMessage([ChatToolCall.CreateFunctionToolCall("call_1", "get_weather", BinaryData.FromString("""{"city": "Paris"}"""))]),
                new ToolChatMessage("call_1", "18 C"),
            ],
            WithWeather(),
            Ct);

        Assert.Equal("Sunny.", completion.Content[0].Text);
        var prompt = Qwen35RoostChatTemplate.Instance.Render(
            [
                RoostChatMessage.User("Weather in Paris?"),
                RoostChatMessage.Assistant(string.Empty, toolCalls: [new RoostToolCall("get_weather", """{"city":"Paris"}""")]),
                RoostChatMessage.Tool("18 C"),
            ],
            [RoostWeather]);
        Assert.Equal(ApiCompatRig.Encode(prompt).Length, completion.Usage.InputTokenCount);
    }

    [Fact]
    public async Task ToolChoiceNoneReturnsTheMarkupAsContent()
    {
        await using var rig = await StartAsync(WeatherCall);

        ChatCompletion completion = await rig.Chat.CompleteChatAsync(
            [new UserChatMessage("Weather in Paris?")], WithWeather(ChatToolChoice.CreateNoneChoice()), Ct);

        Assert.Equal(ChatFinishReason.Stop, completion.FinishReason);
        Assert.Empty(completion.ToolCalls);
        Assert.Equal(WeatherCall, completion.Content[0].Text);
    }

    [Fact]
    public async Task ToolOptionsThatConstrainDecodingAreRefusedByName()
    {
        await using var rig = await StartAsync(WeatherCall);
        ChatMessage[] ask = [new UserChatMessage("Weather in Paris?")];

        await AssertRefusedAsync(() => rig.Chat.CompleteChatAsync(ask, WithWeather(ChatToolChoice.CreateRequiredChoice()), Ct), 400, "invalid_option", "tool_choice");
        await AssertRefusedAsync(() => rig.Chat.CompleteChatAsync(ask, WithWeather(ChatToolChoice.CreateFunctionChoice("get_weather")), Ct), 400, "invalid_option", "tool_choice");

        var serial = WithWeather();
        serial.AllowParallelToolCalls = false;
        await AssertRefusedAsync(() => rig.Chat.CompleteChatAsync(ask, serial, Ct), 400, "invalid_option", "parallel_tool_calls");

        var strict = new ChatCompletionOptions();
        strict.Tools.Add(ChatTool.CreateFunctionTool("get_weather", "Get the weather.", BinaryData.FromString(Schema), functionSchemaIsStrict: true));
        await AssertRefusedAsync(() => rig.Chat.CompleteChatAsync(ask, strict, Ct), 400, "invalid_option", "tools[0].function.strict");

        Assert.Empty(rig.Executor.Steps); // refused before any work
    }
}
