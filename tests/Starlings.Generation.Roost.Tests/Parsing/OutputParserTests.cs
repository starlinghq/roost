using System.Collections.Immutable;
using System.Text.Json;
using Starlings.Generation.Roost.Parsing;
using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost.Tests.Parsing;

/// <summary>
/// The R-053 parsers on the local corpus (<see cref="TemplateCorpus"/>): whole completions, and every split into deltas, which
/// must give the same parts as the whole.
/// </summary>
public sealed class OutputParserTests
{
    public static TheoryData<string> Qwen35Cases() => [.. TemplateCorpus.Qwen35Cases().Select(c => c.Name)];

    public static TheoryData<string> HermesCases() => [.. TemplateCorpus.HermesCases().Select(c => c.Name)];

    public static TheoryData<string> ThinkCases() => [.. TemplateCorpus.ThinkCases().Select(c => c.Name)];

    [Theory]
    [MemberData(nameof(Qwen35Cases))]
    public void ARenderedQwen35TurnParsesBackIntoItsReasoningContentAndCalls(string name)
    {
        var c = TemplateCorpus.Qwen35Cases().Single(x => x.Name == name);
        var completion = TemplateCorpus.RenderQwen35(c);

        var parts = Merge(ParseWhole(Qwen35Parser(c.Tools), completion));

        var reasoning = string.Concat(parts.Where(p => p.Kind == OutputPartKind.Reasoning).Select(p => p.Text));
        var text = string.Concat(parts.Where(p => p.Kind == OutputPartKind.Text).Select(p => p.Text));
        var calls = parts.Where(p => p.Kind == OutputPartKind.ToolCall).Select(p => p.ToolCall!).ToList();
        Assert.Equal(c.Turn.ReasoningContent ?? string.Empty, reasoning);
        Assert.Equal(c.Turn.Content, text);
        var expected = c.Turn.ToolCalls.IsDefault ? [] : c.Turn.ToolCalls;
        Assert.Equal(expected.Length, calls.Count);
        for (var i = 0; i < calls.Count; i++)
        {
            Assert.Equal(expected[i].Name, calls[i].Name);
            AssertJsonEqual(expected[i].ArgumentsJson, calls[i].ArgumentsJson);
        }

        // Reasoning comes first, then the answer and the calls: no reasoning part after a non-reasoning part.
        var firstOther = parts.FindIndex(p => p.Kind != OutputPartKind.Reasoning);
        Assert.True(firstOther < 0 || parts.Skip(firstOther).All(p => p.Kind != OutputPartKind.Reasoning));
    }

    [Theory]
    [MemberData(nameof(Qwen35Cases))]
    public void EverySplitOfARenderedQwen35TurnGivesTheWholeParts(string name)
    {
        var c = TemplateCorpus.Qwen35Cases().Single(x => x.Name == name);
        var completion = TemplateCorpus.RenderQwen35(c);

        AssertSplitsMatch(() => Qwen35Parser(c.Tools), completion);
    }

    [Theory]
    [MemberData(nameof(HermesCases))]
    public void HermesCompletionsParseAsExpected(string name)
    {
        var c = TemplateCorpus.HermesCases().Single(x => x.Name == name);

        var parts = Merge(ParseWhole(HermesParser(), c.Completion));

        Assert.Equal(c.Expected, parts.Select(Describe));
    }

    [Theory]
    [MemberData(nameof(HermesCases))]
    public void EverySplitOfAHermesCompletionGivesTheWholeParts(string name)
    {
        var c = TemplateCorpus.HermesCases().Single(x => x.Name == name);

        AssertSplitsMatch(HermesParser, c.Completion);
    }

    [Theory]
    [MemberData(nameof(ThinkCases))]
    public void ThinkBlocksParseAsExpected(string name)
    {
        var c = TemplateCorpus.ThinkCases().Single(x => x.Name == name);

        var parts = Merge(ParseWhole(ThinkParser(), c.Completion));

        Assert.Equal(c.Expected, parts.Select(Describe));
    }

    [Theory]
    [MemberData(nameof(ThinkCases))]
    public void EverySplitOfAThinkBlockGivesTheWholeParts(string name)
    {
        var c = TemplateCorpus.ThinkCases().Single(x => x.Name == name);

        AssertSplitsMatch(ThinkParser, c.Completion);
    }

    [Fact]
    public void ThinkingAndHermesCallsChain()
    {
        var completion = "Need the time.\n</think>\n\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
        Func<RoostOutputParser> make = () => RoostOutputParser.Create(true, RoostReasoningFormat.ThinkTags, [TemplateCorpus.Clock], RoostToolCallFormat.Hermes)!;

        var parts = Merge(ParseWhole(make(), completion));

        Assert.Equal(["R:Need the time.", "C:get_time {}"], parts.Select(Describe));
        AssertSplitsMatch(make, completion);
    }

    [Fact]
    public void ToolCallMarkupInsideReasoningIsReasoning()
    {
        // Calls are parsed in the answer only: a call the model drafts while thinking is not a call.
        var completion = "Maybe <tool_call>{\"name\": \"get_time\"}</tool_call>?\n</think>\n\nNo call.";

        var parts = Merge(ParseWhole(RoostOutputParser.Create(true, RoostReasoningFormat.ThinkTags, [TemplateCorpus.Clock], RoostToolCallFormat.Hermes)!, completion));

        Assert.Equal(["R:Maybe <tool_call>{\"name\": \"get_time\"}</tool_call>?", "T:No call."], parts.Select(Describe));
    }

    [Fact]
    public void WithoutThinkingAQwen35CompletionIsAnswerAndCalls()
    {
        var completion = "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

        var parts = Merge(ParseWhole(RoostOutputParser.Create(false, RoostReasoningFormat.ThinkTags, [TemplateCorpus.Weather], RoostToolCallFormat.Qwen35)!, completion));

        Assert.Equal(OutputPartKind.Text, parts[0].Kind);
        Assert.Equal("Checking.", parts[0].Text.TrimEnd());
        Assert.Equal("get_weather", Assert.Single(parts, p => p.Kind == OutputPartKind.ToolCall).ToolCall!.Name);
    }

    [Fact]
    public void AMalformedQwen35BlockIsTextVerbatim()
    {
        var completion = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</function>\n</tool_call>";

        var parts = Merge(ParseWhole(RoostOutputParser.Create(false, RoostReasoningFormat.ThinkTags, [TemplateCorpus.Weather], RoostToolCallFormat.Qwen35)!, completion));

        Assert.Equal(["T:" + completion], parts.Select(Describe));
    }

    [Fact]
    public void NoThinkingAndNoToolsMeansNoParser()
    {
        Assert.Null(RoostOutputParser.Create(false, RoostReasoningFormat.ThinkTags, [], RoostToolCallFormat.Qwen35));
        Assert.Null(RoostOutputParser.Create(true, RoostReasoningFormat.None, [TemplateCorpus.Clock], RoostToolCallFormat.None));
    }

    private static RoostOutputParser Qwen35Parser(ImmutableArray<RoostToolDefinition> tools) =>
        RoostOutputParser.Create(true, RoostReasoningFormat.ThinkTags, tools, RoostToolCallFormat.Qwen35)!;

    private static RoostOutputParser HermesParser() =>
        RoostOutputParser.Create(false, RoostReasoningFormat.None, [TemplateCorpus.Clock], RoostToolCallFormat.Hermes)!;

    private static RoostOutputParser ThinkParser() =>
        RoostOutputParser.Create(true, RoostReasoningFormat.ThinkTags, [], RoostToolCallFormat.None)!;

    private static List<OutputPart> ParseWhole(RoostOutputParser parser, string completion) => [.. parser.Append(completion), .. parser.Complete()];

    /// <summary>Single-character deltas, and every split into two deltas, give the whole completion's parts.</summary>
    private static void AssertSplitsMatch(Func<RoostOutputParser> make, string completion)
    {
        var whole = Merge(ParseWhole(make(), completion)).Select(Describe).ToList();

        var single = make();
        var parts = new List<OutputPart>();
        foreach (var ch in completion)
        {
            parts.AddRange(single.Append(ch.ToString()));
        }

        parts.AddRange(single.Complete());
        Assert.Equal(whole, Merge(parts).Select(Describe));

        for (var at = 1; at < completion.Length; at++)
        {
            var parser = make();
            var split = new List<OutputPart>();
            split.AddRange(parser.Append(completion[..at]));
            split.AddRange(parser.Append(completion[at..]));
            split.AddRange(parser.Complete());
            Assert.Equal(whole, Merge(split).Select(Describe));
        }
    }

    private static List<OutputPart> Merge(IEnumerable<OutputPart> parts)
    {
        var merged = new List<OutputPart>();
        foreach (var part in parts)
        {
            Assert.True(part.Kind == OutputPartKind.ToolCall || part.Text.Length > 0, "no empty text parts");
            if (part.Kind != OutputPartKind.ToolCall && merged.Count > 0 && merged[^1].Kind == part.Kind)
            {
                merged[^1] = merged[^1] with { Text = merged[^1].Text + part.Text };
            }
            else
            {
                merged.Add(part);
            }
        }

        return merged;
    }

    private static string Describe(OutputPart part) => part.Kind switch
    {
        OutputPartKind.Text => "T:" + part.Text,
        OutputPartKind.Reasoning => "R:" + part.Text,
        _ => "C:" + part.ToolCall!.Name + " " + part.ToolCall.ArgumentsJson,
    };

    private static void AssertJsonEqual(string expected, string actual)
    {
        using var e = JsonDocument.Parse(expected);
        using var a = JsonDocument.Parse(actual);
        Assert.True(JsonElement.DeepEquals(e.RootElement, a.RootElement), $"expected {expected}, got {actual}");
    }
}
