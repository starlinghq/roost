namespace Starlings.Roost.ModelCompiler.Tests;

// Layers is internal to the generator; the tests reach it through the generated output of small definitions.
public class LayersTests
{
    [Theory]
    [InlineData("0:4", "0,1,2,3")]
    [InlineData("3", "3")]
    [InlineData("0:8%4=3", "3,7")]
    [InlineData("0:8%4!=3", "0,1,2,4,5,6")]
    [InlineData("0:2,5:7", "0,1,5,6")]
    [InlineData("0:2,1:3", "0,1,2")]
    public void LayerSetsExpandAsWritten(string set, string expected)
    {
        string source = GeneratorHarness.Model("[Weight(\"l{i}\", \"d\", Layers = \"" + set + "\")]");
        var r = GeneratorHarness.Run(source);
        string text = Assert.Single(r.Sources, s => s.Contains("WeightSlots", StringComparison.Ordinal) && s.Contains("Forward", StringComparison.Ordinal));
        var slots = System.Text.RegularExpressions.Regex.Matches(text, "new WeightSlot\\(\"l(\\d+)\"")
            .Select(m => m.Groups[1].Value).ToArray();
        Assert.Equal(expected, string.Join(",", slots));
    }

    [Fact]
    public void PlaceholdersAcceptOffsets()
    {
        string source = GeneratorHarness.Model("[Weight(\"l{i+1}.x{i-1}\", \"d\", Layers = \"1:3\")]");
        var r = GeneratorHarness.Run(source);
        string text = Assert.Single(r.Sources, s => s.Contains("Forward<TOps>", StringComparison.Ordinal));
        Assert.Contains("\"l2.x0\"", text, StringComparison.Ordinal);
        Assert.Contains("\"l3.x1\"", text, StringComparison.Ordinal);
    }
}
