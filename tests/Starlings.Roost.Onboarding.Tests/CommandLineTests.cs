using Starlings.Roost.Onboarding;

namespace Starlings.Roost.Onboarding.Tests;

public sealed class CommandLineTests
{
    private static (int Code, string Out, string Err) Run(params string[] args)
    {
        using var stdout = new StringWriter();
        using var stderr = new StringWriter();
        int code = Program.Run(args, stdout, stderr);
        return (code, stdout.ToString(), stderr.ToString());
    }

    [Fact]
    public void InspectWritesTheReportAndExitsZeroWhenTheModelComposes()
    {
        var (code, output, err) = Run("inspect", ConfigInspectorTests.DataPath("qwen3-0.6b.config.json"));

        Assert.Equal(0, code);
        var report = ConfigInspector.FromJson(output);
        Assert.Equal(CompositionVerdict.Composes, report.Verdict);
        Assert.EndsWith("composes" + Environment.NewLine, err, StringComparison.Ordinal);
    }

    [Fact]
    public void InspectExitsOneAndNamesTheMissingOps()
    {
        var (code, output, err) = Run("inspect", ConfigInspectorTests.DataPath("deepseek-v3.config.json"), "--block-size", "64");

        Assert.Equal(1, code);
        Assert.Equal(CompositionVerdict.MissingOps, ConfigInspector.FromJson(output).Verdict);
        Assert.Equal(64, ConfigInspector.FromJson(output).StateGroups[0].BlockSize);
        Assert.Contains("missing_ops: ", err, StringComparison.Ordinal);
        Assert.Contains("MoeRouteTopK", err, StringComparison.Ordinal);
    }

    [Fact]
    public void InspectWritesToOutWhenAsked()
    {
        string path = Path.Combine(Path.GetTempPath(), "roost-onboard-" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            var (code, output, _) = Run("inspect", ConfigInspectorTests.DataPath("gemma-3-1b.config.json"), "--out", path, "--kv-dtype", "F16");
            Assert.Equal(0, code);
            Assert.Empty(output);
            Assert.Equal("F16", ConfigInspector.FromJson(File.ReadAllText(path)).StateGroups[0].DType);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData]
    [InlineData("inspect")]
    [InlineData("inspect", "a.json", "b.json")]
    [InlineData("inspect", "a.json", "--block-size", "0")]
    [InlineData("inspect", "a.json", "--kv-dtype", "I4")]
    [InlineData("inspect", "a.json", "--block-size")]
    [InlineData("validate-recipe")]
    [InlineData("frobnicate")]
    public void BadArgumentsExitTwo(params string[] args) => Assert.Equal(2, Run(args).Code);

    [Fact]
    public void AnUnreadableConfigExitsTwo()
    {
        var (code, _, err) = Run("inspect", Path.Combine(AppContext.BaseDirectory, "TestData", "no-such-config.json"));
        Assert.Equal(2, code);
        Assert.Contains("no-such-config.json", err, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateRecipeExitsZeroForTheCommittedRecipes()
    {
        string root = RecipeValidatorTests.RepositoryRoot();
        var (code, output, _) = Run("validate-recipe", "--root", root, Path.Combine(root, "evidence", "recipes"));
        Assert.Equal(0, code);
        Assert.Empty(output);
    }

    [Fact]
    public void ValidateRecipeExitsOneWithOneLinePerProblem()
    {
        string path = Path.Combine(Path.GetTempPath(), "bad-" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            File.WriteAllText(path, """{"plugin":"roost-v1"}""");
            var (code, output, _) = Run("validate-recipe", path);
            Assert.Equal(1, code);
            Assert.Contains(path + ": : 'schema_version' is required", output, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SchemaPrintsTheEmbeddedSchema()
    {
        var (code, output, _) = Run("schema");
        Assert.Equal(0, code);
        Assert.Equal(RecipeValidator.SchemaText(), output);
    }
}
