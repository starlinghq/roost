using System.Collections.Immutable;
using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost.Tests.Parsing;

/// <summary>One assistant turn of the local corpus: what the model should answer, and the tools of the request.</summary>
/// <param name="Name">The case name.</param>
/// <param name="Turn">The assistant turn (reasoning, content and tool calls).</param>
/// <param name="Tools">The request's tools.</param>
internal sealed record CorpusCase(string Name, RoostChatMessage Turn, ImmutableArray<RoostToolDefinition> Tools)
{
    public override string ToString() => Name;
}

/// <summary>A hand-written completion and the parts it must parse into.</summary>
/// <param name="Name">The case name.</param>
/// <param name="Completion">The completion text.</param>
/// <param name="Expected">The expected parts, adjacent text merged: <c>T:</c> text, <c>R:</c> reasoning, <c>C:</c> call as <c>name args-json</c>.</param>
internal sealed record RawCase(string Name, string Completion, string[] Expected)
{
    public override string ToString() => Name;
}

/// <summary>
/// The local tool-call and reasoning corpus of R-053. The build plan's template corpus (P-029) is not in this repository, so
/// these cases stand in for it, and they are not fixtures: no status may flip to experimental or supported on them.
/// </summary>
/// <remarks>
/// <para><b>Qwen3.5.</b> Each case is an assistant turn rendered by Starling's published template
/// (<see cref="Qwen35RoostChatTemplate"/> over <c>Qwen35ChatTemplate</c>, <c>Starlings.Models.Qwen35</c> 0.3.0) as the last turn of
/// a conversation: the template writes the think block and the <c>&lt;tool_call&gt;</c> blocks exactly as it instructs the model to
/// answer, so the rendered turn is the completion a well-behaved model produces with thinking enabled. Parsing it must give back
/// the turn's reasoning, content and calls.</para>
/// <para><b>Hermes.</b> Starling's published Qwen3 template is text only (no tools), so these completions are written by hand in
/// the format the upstream Qwen2.5 and Qwen3 chat templates instruct: one <c>&lt;tool_call&gt;</c> block per call holding
/// <c>{"name": ..., "arguments": ...}</c> on its own line.</para>
/// </remarks>
internal static class TemplateCorpus
{
    private const string AssistantStart = "<|im_start|>assistant\n";
    private const string TurnEnd = "<|im_end|>";

    public static readonly RoostToolDefinition Weather = new(
        "get_weather",
        "Get the current weather for a city.",
        """{"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}""");

    public static readonly RoostToolDefinition Search = new(
        "search",
        "Search the catalogue.",
        """{"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}, "filter": {"type": "object"}, "score": {"type": "number"}}}""");

    public static readonly RoostToolDefinition Clock = new("get_time", "The current time.");

    private static readonly ImmutableArray<RoostToolDefinition> AllTools = [Weather, Search, Clock];

    public static IEnumerable<CorpusCase> Qwen35Cases()
    {
        static CorpusCase Case(string name, string content, string? reasoning, params RoostToolCall[] calls) =>
            new(name, RoostChatMessage.Assistant(content, reasoning, calls.Length == 0 ? default : [.. calls]), AllTools);

        yield return Case("answer-with-reasoning", "Paris is the capital of France.", "The user asks for a capital. France: Paris.");
        yield return Case("answer-multiline-reasoning", "Four.", "Two plus two.\n\nThat is four.");
        yield return Case("one-call", string.Empty, "Need the weather.", new RoostToolCall("get_weather", """{"city": "Paris"}"""));
        yield return Case("content-then-call", "Let me check.", "Weather lookup.", new RoostToolCall("get_weather", """{"city": "Paris", "unit": "c"}"""));
        yield return Case(
            "two-calls-typed-arguments",
            string.Empty,
            "Search twice.",
            new RoostToolCall("search", """{"query": "red shoes", "limit": 5, "exact": true, "tags": ["sale", "new"], "filter": {"size": 42}, "score": 0.5}"""),
            new RoostToolCall("get_weather", """{"city": "Oslo"}"""));
        yield return Case("call-without-arguments", string.Empty, "Clock.", new RoostToolCall("get_time"));
        yield return Case("multiline-string-value", string.Empty, "Search a phrase.", new RoostToolCall("search", "{\"query\": \"line one\\nline two\"}"));
        yield return Case("unicode", "Voilà : 中文 \U0001F600", "Réponse en français.");
        yield return Case("markup-like-text", "Use a <b>bold</b> tag, or a < b.", "HTML question.");
    }

    /// <summary>The completion a Qwen3.5 model writes for <paramref name="c"/> with thinking enabled: the rendered turn.</summary>
    public static string RenderQwen35(CorpusCase c)
    {
        var prompt = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("Question?"), c.Turn], c.Tools);
        var start = prompt.IndexOf(AssistantStart, StringComparison.Ordinal) + AssistantStart.Length;
        var end = prompt.IndexOf(TurnEnd, start, StringComparison.Ordinal);
        return prompt[start..end];
    }

    public static IEnumerable<RawCase> HermesCases()
    {
        yield return new("text-only", "The capital of France is Paris.", ["T:The capital of France is Paris."]);
        yield return new("one-call", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", ["C:get_weather {\"city\": \"Paris\"}"]);
        yield return new("content-then-call", "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", ["T:Let me check.\n", "C:get_weather {\"city\": \"Paris\"}"]);
        yield return new(
            "two-calls",
            "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"shoes\", \"limit\": 5, \"tags\": [\"a\"], \"filter\": {\"x\": null}}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>",
            ["C:search {\"query\": \"shoes\", \"limit\": 5, \"tags\": [\"a\"], \"filter\": {\"x\": null}}", "C:get_time {}"]);
        yield return new("no-arguments-member", "<tool_call>\n{\"name\": \"get_time\"}\n</tool_call>", ["C:get_time {}"]);
        yield return new("extra-members-ignored", "<tool_call>{\"name\": \"get_time\", \"arguments\": {}, \"id\": 3}</tool_call>", ["C:get_time {}"]);
        yield return new("whitespace-around-calls-dropped", "\n\n<tool_call>{\"name\": \"get_time\"}</tool_call>\n\n", ["C:get_time {}"]);
        yield return new("text-after-call-kept", "<tool_call>{\"name\": \"get_time\"}</tool_call>\nDone.", ["C:get_time {}", "T:\nDone."]);
        yield return new("whitespace-only-without-call-kept", "\n \n", ["T:\n \n"]);
        yield return new("invalid-json-is-text", "<tool_call>{\"name\": get_time}</tool_call>", ["T:<tool_call>{\"name\": get_time}</tool_call>"]);
        yield return new("missing-name-is-text", "<tool_call>{\"arguments\": {}}</tool_call>", ["T:<tool_call>{\"arguments\": {}}</tool_call>"]);
        yield return new("array-arguments-is-text", "<tool_call>{\"name\": \"f\", \"arguments\": [1]}</tool_call>", ["T:<tool_call>{\"name\": \"f\", \"arguments\": [1]}</tool_call>"]);
        yield return new("unfinished-block-is-text", "Sure.\n<tool_call>\n{\"name\": \"get_time\"", ["T:Sure.\n<tool_call>\n{\"name\": \"get_time\""]);
        yield return new("tag-prefix-in-text", "a <tool b <tool_c </tool_call>", ["T:a <tool b <tool_c </tool_call>"]);
        yield return new("ends-in-tag-prefix", "See <tool_ca", ["T:See <tool_ca"]);
        yield return new("malformed-then-call", "<tool_call>oops</tool_call><tool_call>{\"name\": \"get_time\"}</tool_call>", ["T:<tool_call>oops</tool_call>", "C:get_time {}"]);
    }

    public static IEnumerable<RawCase> ThinkCases()
    {
        yield return new("reasoning-then-answer", "Think.\n</think>\n\nAnswer.", ["R:Think.", "T:Answer."]);
        yield return new("repeated-open-tag", "<think>\nThink.\n</think>\n\nAnswer.", ["R:Think.", "T:Answer."]);
        yield return new("leading-line-feeds", "\n\nThink.\n\n</think>Answer.", ["R:Think.", "T:Answer."]);
        yield return new("empty-reasoning", "\n</think>\n\nAnswer.", ["T:Answer."]);
        yield return new("ends-inside-reasoning", "Still thinking about", ["R:Still thinking about"]);
        yield return new("ends-in-close-prefix", "Thinking </thi", ["R:Thinking </thi"]);
        yield return new("inner-line-feeds-kept", "a\n\nb\n</think>\nc\n\nd\n", ["R:a\n\nb", "T:c\n\nd\n"]);
        yield return new("second-close-is-answer", "a</think>b</think>c", ["R:a", "T:b</think>c"]);
        yield return new("open-tag-prefix-is-reasoning", "<thin", ["R:<thin"]);
        yield return new("no-answer", "Thought.\n</think>\n\n", ["R:Thought."]);
    }
}
