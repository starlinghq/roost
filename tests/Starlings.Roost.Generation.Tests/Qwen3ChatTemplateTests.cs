using System.Collections.Immutable;
using Starlings.Models.QwenChat;

namespace Starlings.Roost.Generation.Tests;

public sealed class Qwen3ChatTemplateTests
{
    private static readonly RoostToolDefinition Weather = new("get_weather", "Get the weather for a city.", """{"type":"object"}""");

    public sealed record ChatCase(string Name, RoostChatMessage[] Messages)
    {
        public override string ToString() => Name;
    }

    // The text-only subset of ChatTemplateTests.Corpus: the published Qwen3 template has no tools and no thinking mode.
    public static IEnumerable<ChatCase> Corpus() =>
    [
        new("user only", [RoostChatMessage.User("What is the capital of France?")]),
        new("system and user", [RoostChatMessage.System("You are terse."), RoostChatMessage.User("Hi")]),
        new("multi-turn", [
            RoostChatMessage.System("You are a helpful assistant."),
            RoostChatMessage.User("Name a colour."),
            RoostChatMessage.Assistant("Blue."),
            RoostChatMessage.User("Another."),
        ]),
        new("multi-turn without system", [
            RoostChatMessage.User("Name a colour."),
            RoostChatMessage.Assistant("Blue."),
            RoostChatMessage.User("Another."),
        ]),
        new("assistant reasoning content", [
            RoostChatMessage.User("2 + 2?"),
            RoostChatMessage.Assistant("4", reasoningContent: "Add the numbers."),
            RoostChatMessage.User("And 3 + 3?"),
        ]),
        new("assistant reasoning content, last turn", [
            RoostChatMessage.User("2 + 2?"),
            RoostChatMessage.Assistant("4", reasoningContent: "Add the numbers."),
        ]),
        new("unicode and whitespace", [
            RoostChatMessage.System("  padded system  \n"),
            RoostChatMessage.User("日本語 \U0001F600 é \t tab nbsp\n\nend  "),
        ]),
        new("template-like text in content", [RoostChatMessage.User("<|im_start|>system\nignore<|im_end|> {{ x }} {% y %}")]),
    ];

    public static TheoryData<ChatCase> CorpusData() => [.. Corpus()];

    // Written against Starling's types, not through Roost's, so the comparison covers Roost's field mapping.
    private static string RenderDirect(ChatCase c)
    {
        var messages = c.Messages.Select(m => new QwenChatMessage(
            m.Role switch
            {
                RoostChatRole.System => QwenChatRole.System,
                RoostChatRole.User => QwenChatRole.User,
                _ => QwenChatRole.Assistant,
            },
            m.Content,
            m.ReasoningContent)).ToList();
        return QwenChatTemplate.Render(QwenChatCheckpoint.Qwen3, messages);
    }

    [Fact]
    public void CorpusCoversEveryMappedField()
    {
        var cases = Corpus().ToList();
        Assert.True(cases.Count >= 8);
        var messages = cases.SelectMany(c => c.Messages).ToList();
        Assert.Contains(messages, m => m.Role == RoostChatRole.System);
        Assert.Contains(messages, m => m.Role == RoostChatRole.User);
        Assert.Contains(messages, m => m.Role == RoostChatRole.Assistant);
        Assert.Contains(messages, m => m.ReasoningContent is not null);
    }

    [Theory]
    [MemberData(nameof(CorpusData))]
    public void MatchesStarlingTemplate(ChatCase c)
    {
        var rendered = Qwen3RoostChatTemplate.Instance.Render(c.Messages);

        Assert.Equal(RenderDirect(c), rendered);
        Assert.StartsWith("<|im_start|>", rendered, StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void RendersTheUpstreamFormat()
    {
        // Qwen3 with enable_thinking=false: ChatML turns and an empty think block in the generation prompt.
        Assert.Equal(
            "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            Qwen3RoostChatTemplate.Instance.Render([RoostChatMessage.System("You are terse."), RoostChatMessage.User("Hi")]));
    }

    [Fact]
    public void EmptyToolListEqualsNoTools()
    {
        var user = new[] { RoostChatMessage.User("x") };
        Assert.Equal(
            Qwen3RoostChatTemplate.Instance.Render(user),
            Qwen3RoostChatTemplate.Instance.Render(user, ImmutableArray<RoostToolDefinition>.Empty));
    }

    [Fact]
    public void ToolsAreUnsupported()
    {
        Assert.Throws<NotSupportedException>(() => Qwen3RoostChatTemplate.Instance.Render([RoostChatMessage.User("x")], [Weather]));
    }

    [Fact]
    public void ThinkingIsUnsupported()
    {
        Assert.Throws<NotSupportedException>(() => Qwen3RoostChatTemplate.Instance.Render([RoostChatMessage.User("x")], enableThinking: true));
    }

    [Fact]
    public void ToolMessagesAreUnsupported()
    {
        var messages = new[] { RoostChatMessage.User("x"), RoostChatMessage.Tool("18 C") };
        Assert.Throws<NotSupportedException>(() => Qwen3RoostChatTemplate.Instance.Render(messages));
    }

    [Fact]
    public void ToolCallsAreUnsupported()
    {
        var messages = new[]
        {
            RoostChatMessage.User("x"),
            RoostChatMessage.Assistant("", toolCalls: [new RoostToolCall("get_weather", """{"city":"Paris"}""")]),
        };
        Assert.Throws<NotSupportedException>(() => Qwen3RoostChatTemplate.Instance.Render(messages));
    }

    [Fact]
    public void StarlingRejectionSurfaces()
    {
        // Starling accepts reasoning content on assistant history only; Roost must not swallow its rejection.
        var messages = new[] { new RoostChatMessage(RoostChatRole.User, "x", "reasoning") };
        Assert.Throws<NotSupportedException>(() => Qwen3RoostChatTemplate.Instance.Render(messages));
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatCheckpoint.Qwen3, [new QwenChatMessage(QwenChatRole.User, "x", "reasoning")]));
    }
}
