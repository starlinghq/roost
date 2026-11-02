using Starlings.Roost.Native.Tests.Support;

namespace Starlings.Roost.Native.Tests;

/// <summary>Matchers through the managed bindings: masks, consuming, the stop after a complete value, and grammar errors.</summary>
public sealed class MatcherTests
{
    private const string Flat = """{"type":"object","properties":{"a":{"type":"integer"},"b":{"enum":["x","y"]}},"required":["a","b"],"additionalProperties":false}""";

    [Fact]
    public void AValidInstanceIsAllowedTokenByTokenAndThenOnlyTheStopId()
    {
        var tokenizer = Grammar.Require();
        using var matcher = tokenizer.CreateMatcher(GrammarSpec.JsonSchema(Flat));
        var mask = new uint[matcher.MaskWords];
        Assert.False(matcher.IsAccepting);
        foreach (var token in tokenizer.Tokenize("""{"a":-7,"b":"x"}"""))
        {
            matcher.ComputeMask(mask);
            Assert.True(Grammar.Allowed(mask, token));
            Assert.True(matcher.TryConsume(token));
        }

        Assert.True(matcher.IsAccepting);
        Assert.True(matcher.IsStopped);
        matcher.ComputeMask(mask);
        Assert.Equal([Grammar.StopId], Grammar.AllowedTokens(mask, matcher.VocabSize));
        Assert.True(matcher.TryConsume(Grammar.StopId)); // accepted without being consumed (S10 finding 2)
        Assert.False(matcher.TryConsume('x'));
    }

    [Fact]
    public void ARejectedTokenLeavesTheMatcherAsItWas()
    {
        var tokenizer = Grammar.Require();
        using var matcher = tokenizer.CreateMatcher(GrammarSpec.JsonObject);
        var before = new uint[matcher.MaskWords];
        matcher.ComputeMask(before);
        Assert.False(Grammar.Allowed(before, 'a'));
        Assert.False(matcher.TryConsume('a'));
        var after = new uint[matcher.MaskWords];
        matcher.ComputeMask(after);
        Assert.Equal(before, after);
        var range = Assert.Throws<GrammarException>(() => matcher.TryConsume(5000));
        Assert.Equal(GrammarStatus.Argument, range.Status);
    }

    [Fact]
    public void JsonIsCompact()
    {
        var tokenizer = Grammar.Require();
        using var matcher = tokenizer.CreateMatcher(GrammarSpec.JsonObject);
        Assert.True(matcher.TryConsume('{'));
        var mask = new uint[matcher.MaskWords];
        matcher.ComputeMask(mask);
        Assert.False(Grammar.Allowed(mask, ' '));
        Assert.False(Grammar.Allowed(mask, '\n'));
        Assert.True(Grammar.Allowed(mask, '"'));
        Assert.True(Grammar.Allowed(mask, '}'));
    }

    [Theory]
    [InlineData(GrammarKind.Regex, "(yes|no)[0-9]{2}", "no42")]
    [InlineData(GrammarKind.Lark, "start: \"a\" \"b\"+", "abbb")]
    public void RegexAndLarkMatch(GrammarKind kind, string source, string text)
    {
        var tokenizer = Grammar.Require();
        using var matcher = tokenizer.CreateMatcher(new GrammarSpec(kind, source));
        foreach (var token in tokenizer.Tokenize(text))
        {
            Assert.True(matcher.TryConsume(token));
        }

        Assert.True(matcher.IsAccepting);
    }

    [Theory]
    [InlineData(GrammarKind.JsonSchema, """{"type":"array","uniqueItems":true}""", "uniqueItems")]
    [InlineData(GrammarKind.JsonSchema, "{not json", "not JSON")]
    [InlineData(GrammarKind.JsonSchema, "[1,2]", "must be a JSON object")]
    [InlineData(GrammarKind.Lark, "start: missing_rule", "")]
    public void AnInvalidGrammarIsTheCallersFault(GrammarKind kind, string source, string fragment)
    {
        var tokenizer = Grammar.Require();
        var e = Assert.Throws<GrammarException>(() => tokenizer.Validate(new GrammarSpec(kind, source)));
        Assert.Equal(GrammarStatus.Grammar, e.Status);
        Assert.True(e.IsInvalidGrammar);
        Assert.Contains(fragment, e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnknownKindIsAnArgumentError()
    {
        var tokenizer = Grammar.Require();
        var e = Assert.Throws<GrammarException>(() => tokenizer.Validate(new GrammarSpec((GrammarKind)99, "")));
        Assert.Equal(GrammarStatus.Argument, e.Status);
    }

    [Fact]
    public void AMaskOfTheWrongSizeIsRefused()
    {
        var tokenizer = Grammar.Require();
        using var matcher = tokenizer.CreateMatcher(GrammarSpec.JsonObject);
        _ = Assert.Throws<ArgumentException>(() => matcher.ComputeMask(new uint[3]));
    }

    [Fact]
    public void PaddedRowsAreNeverAllowed()
    {
        _ = Grammar.Require();
        using var padded = Grammar.Library!.CreateTokenizer(Grammar.TokenizerJson, 300, [Grammar.StopId]);
        using var matcher = padded.CreateMatcher(new GrammarSpec(GrammarKind.Regex, "[a-z]+"));
        var mask = new uint[matcher.MaskWords];
        matcher.ComputeMask(mask);
        Assert.All(Enumerable.Range(264, 36), v => Assert.False(Grammar.Allowed(mask, v)));
    }

    [Fact]
    public void ADisposedMatcherThrows()
    {
        var tokenizer = Grammar.Require();
        var matcher = tokenizer.CreateMatcher(GrammarSpec.JsonObject);
        matcher.Dispose();
        matcher.Dispose();
        _ = Assert.Throws<ObjectDisposedException>(() => matcher.TryConsume('{'));
    }

    [Fact]
    public void MatchersCanBeMadeFromSeveralThreads()
    {
        var tokenizer = Grammar.Require();
        var made = 0;
        Parallel.For(0, 16, _ =>
        {
            using var matcher = tokenizer.CreateMatcher(GrammarSpec.JsonSchema(Flat));
            var mask = new uint[matcher.MaskWords];
            matcher.ComputeMask(mask);
            Assert.Equal(['{'], Grammar.AllowedTokens(mask, matcher.VocabSize));
            Interlocked.Increment(ref made);
        });
        Assert.Equal(16, made);
    }
}
