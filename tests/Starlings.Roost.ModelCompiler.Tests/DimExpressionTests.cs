namespace Starlings.Roost.ModelCompiler.Tests;

public class DimExpressionTests
{
    private static readonly Dictionary<string, long> Symbols = new() { ["a"] = 6, ["b"] = 4 };

    [Theory]
    [InlineData("42", 42)]
    [InlineData("a*b", 24)]
    [InlineData("a*b+1", 25)]
    [InlineData("a+b*2", 14)]
    [InlineData("(a+b)*2", 20)]
    [InlineData("a/4", 1)]
    [InlineData("a%4", 2)]
    [InlineData("a - b - 1", 1)]
    [InlineData("-a+10", 4)]
    public void Evaluates(string text, long expected) => Assert.Equal(expected, DimExpression.Evaluate(text, Symbols));

    [Theory]
    [InlineData("")]
    [InlineData("a +")]
    [InlineData("(a")]
    [InlineData("a b")]
    [InlineData("c")]
    [InlineData("a/0")]
    public void Rejects(string text)
    {
        Assert.False(DimExpression.TryEvaluate(text, Symbols, out _, out string? error));
        Assert.False(string.IsNullOrEmpty(error));
    }

    [Fact]
    public void ListsNamesInOrderOfFirstUse() =>
        Assert.Equal(["tokens", "hidden"], DimExpression.Names("tokens*hidden+tokens*2"));
}
