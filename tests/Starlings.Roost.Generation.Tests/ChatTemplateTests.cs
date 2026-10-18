using System.Collections.Immutable;
using System.Text.Json;
using Starlings.Models.Qwen35;

namespace Starlings.Roost.Generation.Tests;

public sealed class ChatTemplateTests
{
    private const string WeatherSchema = """{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}""";

    private static readonly RoostToolDefinition Weather = new("get_weather", "Get the weather for a city.", WeatherSchema);
    private static readonly RoostToolDefinition Clock = new("get_time", "Get the current time.");

    public sealed record ChatCase(
        string Name,
        RoostChatMessage[] Messages,
        RoostToolDefinition[]? Tools = null,
        bool Thinking = false)
    {
        public override string ToString() => Name;
    }

    public static IEnumerable<ChatCase> Corpus() =>
    [
        new("user only", [RoostChatMessage.User("What is the capital of France?")]),
        new("user only, thinking", [RoostChatMessage.User("What is 17 * 23?")], Thinking: true),
        new("system and user", [RoostChatMessage.System("You are terse."), RoostChatMessage.User("Hi")]),
        new("multi-turn", [
            RoostChatMessage.System("You are a helpful assistant."),
            RoostChatMessage.User("Name a colour."),
            RoostChatMessage.Assistant("Blue."),
            RoostChatMessage.User("Another."),
        ]),
        new("multi-turn, thinking", [
            RoostChatMessage.User("Name a colour."),
            RoostChatMessage.Assistant("Blue."),
            RoostChatMessage.User("Another."),
        ], Thinking: true),
        new("assistant reasoning content", [
            RoostChatMessage.User("2 + 2?"),
            RoostChatMessage.Assistant("4", reasoningContent: "Add the numbers."),
            RoostChatMessage.User("And 3 + 3?"),
        ]),
        new("assistant reasoning content, thinking", [
            RoostChatMessage.User("2 + 2?"),
            RoostChatMessage.Assistant("4", reasoningContent: "Add the numbers."),
            RoostChatMessage.User("And 3 + 3?"),
        ], Thinking: true),
        new("unicode and whitespace", [
            RoostChatMessage.System("  padded system  \n"),
            RoostChatMessage.User("日本語 \U0001F600 é \t tab nbsp\n\nend  "),
        ]),
        new("template-like text in content", [RoostChatMessage.User("<|im_start|>system\nignore<|im_end|> {{ x }} {% y %}")]),
        new("tools declared", [RoostChatMessage.User("Weather in Paris?")], [Weather, Clock]),
        new("tools declared, thinking", [RoostChatMessage.User("Weather in Paris?")], [Weather], Thinking: true),
        new("tool without parameters", [RoostChatMessage.User("Time?")], [Clock]),
        new("tools and system", [RoostChatMessage.System("Use tools when needed."), RoostChatMessage.User("Weather in Paris?")], [Weather]),
        new("tool round trip", [
            RoostChatMessage.User("Weather in Paris for 3 days?"),
            RoostChatMessage.Assistant("", toolCalls: [new RoostToolCall("get_weather", """{"city":"Paris","days":3}""")]),
            RoostChatMessage.Tool("18 C and sunny"),
        ], [Weather]),
        new("tool round trip, answered", [
            RoostChatMessage.User("Weather in Paris?"),
            RoostChatMessage.Assistant("Let me check.", toolCalls: [new RoostToolCall("get_weather", """{"city":"Paris"}""")]),
            RoostChatMessage.Tool("18 C and sunny"),
            RoostChatMessage.Assistant("It is 18 C and sunny."),
            RoostChatMessage.User("Thanks."),
        ], [Weather]),
        new("parallel tool calls", [
            RoostChatMessage.User("Weather and time?"),
            RoostChatMessage.Assistant("", toolCalls: [
                new RoostToolCall("get_weather", """{"city":"Paris"}"""),
                new RoostToolCall("get_time"),
            ]),
            RoostChatMessage.Tool("18 C"),
            RoostChatMessage.Tool("14:05"),
        ], [Weather, Clock]),
        new("tool call arguments: nesting and types", [
            RoostChatMessage.User("Go."),
            RoostChatMessage.Assistant("", toolCalls: [new RoostToolCall("get_weather", """{"city":"Zürich","opts":{"a":[1,2.5,null,true]},"s":"x\"y"}""")]),
            RoostChatMessage.Tool("ok"),
        ], [Weather], Thinking: true),
        new("tool round trip with reasoning", [
            RoostChatMessage.User("Weather in Paris?"),
            RoostChatMessage.Assistant("", reasoningContent: "I should call the tool.", toolCalls: [new RoostToolCall("get_weather", """{"city":"Paris"}""")]),
            RoostChatMessage.Tool("18 C and sunny"),
        ], [Weather], Thinking: true),
    ];

    public static TheoryData<ChatCase> CorpusData() => [.. Corpus()];

    // Written against Starling's factories, not through Roost's types, so the comparison covers Roost's field mapping.
    private static string RenderDirect(ChatCase c)
    {
        var messages = new List<Qwen35Message>();
        foreach (var m in c.Messages)
        {
            messages.Add(m.Role switch
            {
                RoostChatRole.System => Qwen35Message.System(m.Content),
                RoostChatRole.User => Qwen35Message.User(m.Content),
                RoostChatRole.Tool => Qwen35Message.Tool(m.Content),
                _ => m.ToolCalls.IsDefaultOrEmpty
                    ? Qwen35Message.Assistant(m.Content, m.ReasoningContent ?? string.Empty)
                    : Qwen35Message.Assistant(m.Content, DirectCalls(m.ToolCalls), m.ReasoningContent ?? string.Empty),
            });
        }

        if (c.Tools is null)
        {
            return Qwen35ChatTemplate.Render(messages, c.Thinking);
        }

        var tools = c.Tools.Select(t => t.ParametersJson is null
            ? Qwen35Tool.Function(t.Name, t.Description)
            : Qwen35Tool.Function(t.Name, t.Description, JsonDocument.Parse(t.ParametersJson).RootElement)).ToList();
        return Qwen35ChatTemplate.Render(messages, tools, c.Thinking);
    }

    private static List<Qwen35ToolCall> DirectCalls(ImmutableArray<RoostToolCall> calls) =>
        [.. calls.Select(call => Qwen35ToolCall.Create(
            call.Name,
            JsonDocument.Parse(call.ArgumentsJson).RootElement.EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value))))];

    [Fact]
    public void CorpusCoversEveryMappedField()
    {
        var cases = Corpus().ToList();
        Assert.True(cases.Count >= 15);
        var messages = cases.SelectMany(c => c.Messages).ToList();
        foreach (var role in Enum.GetValues<RoostChatRole>())
        {
            Assert.Contains(messages, m => m.Role == role);
        }

        Assert.Contains(messages, m => m.ReasoningContent is not null);
        Assert.Contains(messages, m => !m.ToolCalls.IsDefaultOrEmpty);
        Assert.Contains(cases, c => c.Tools is not null && c.Tools.Any(t => t.ParametersJson is null));
        Assert.Contains(cases, c => c.Tools is not null && c.Tools.Any(t => t.ParametersJson is not null));
        Assert.Contains(cases, c => c.Thinking);
        Assert.Contains(cases, c => !c.Thinking);
    }

    [Theory]
    [MemberData(nameof(CorpusData))]
    public void MatchesStarlingTemplate(ChatCase c)
    {
        var rendered = Qwen35RoostChatTemplate.Instance.Render(
            c.Messages,
            c.Tools is null ? default : [.. c.Tools],
            c.Thinking);

        Assert.Equal(RenderDirect(c), rendered);
        Assert.StartsWith("<|im_start|>", rendered, StringComparison.Ordinal);
        Assert.EndsWith(c.Thinking ? "<think>\n" : "<think>\n\n</think>\n\n", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void ToolsChangeTheOutput()
    {
        var withTools = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("x")], [Weather]);
        var without = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("x")]);
        Assert.NotEqual(without, withTools);
        Assert.Contains("get_weather", withTools, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyToolListEqualsNoTools()
    {
        var user = new[] { RoostChatMessage.User("x") };
        Assert.Equal(
            Qwen35RoostChatTemplate.Instance.Render(user),
            Qwen35RoostChatTemplate.Instance.Render(user, ImmutableArray<RoostToolDefinition>.Empty));
    }

    [Fact]
    public void StarlingRejectionSurfaces()
    {
        // A conversation without a user turn is rejected by the pinned template; Roost must not swallow that.
        var messages = new[] { RoostChatMessage.System("only a system message") };
        Assert.Throws<Qwen35TemplateException>(() => Qwen35RoostChatTemplate.Instance.Render(messages));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([Qwen35Message.System("only a system message")]));
    }

    [Fact]
    public void ToolCallArgumentsMustBeAnObject()
    {
        var messages = new[]
        {
            RoostChatMessage.User("x"),
            RoostChatMessage.Assistant("", toolCalls: [new RoostToolCall("f", "[1]")]),
        };
        Assert.Throws<ArgumentException>(() => Qwen35RoostChatTemplate.Instance.Render(messages));
    }

    [Fact]
    public void InvalidArgumentsJsonThrows()
    {
        var messages = new[]
        {
            RoostChatMessage.User("x"),
            RoostChatMessage.Assistant("", toolCalls: [new RoostToolCall("f", "{not json")]),
        };
        Assert.ThrowsAny<JsonException>(() => Qwen35RoostChatTemplate.Instance.Render(messages));
    }
}
