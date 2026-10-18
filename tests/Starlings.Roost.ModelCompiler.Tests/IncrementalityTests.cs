using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using static Starlings.Roost.ModelCompiler.Tests.GeneratorHarness;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>The generator's output step is cached when the definition has not changed (S4 incrementality table).</summary>
public class IncrementalityTests
{
    private static SyntaxTree Parse(string text, string path) =>
        CSharpSyntaxTree.ParseText(text, path: path, cancellationToken: TestContext.Current.CancellationToken);

    private static string[] Reasons(GeneratorDriver driver)
    {
        var r = driver.GetRunResult().Results[0];
        return r.TrackedOutputSteps.SelectMany(kv => kv.Value).SelectMany(s => s.Outputs).Select(o => o.Item2.ToString()).ToArray();
    }

    private static (GeneratorDriver Driver, Compilation Compilation, SyntaxTree Def, SyntaxTree Other) Start()
    {
        var def = Parse(Model(), path: "Model.cs");
        var other = Parse("namespace T { public class Other { public int X() => 1; } }", path: "Other.cs");
        var compilation = CSharpCompilation.Create("T", [def, other], References(), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            [new ModelGenerator().AsSourceGenerator()],
            driverOptions: new GeneratorDriverOptions(IncrementalGeneratorOutputKind.None, trackIncrementalGeneratorSteps: true));
        return (driver.RunGenerators(compilation, TestContext.Current.CancellationToken), compilation, def, other);
    }

    [Fact]
    public void FirstRunEmitsNew()
    {
        var (driver, _, _, _) = Start();
        Assert.Contains("New", Reasons(driver));
        Assert.Empty(driver.GetRunResult().Diagnostics);
    }

    [Fact]
    public void EditingAnUnrelatedFileIsCached()
    {
        var (driver, compilation, _, other) = Start();
        var edited = Parse("namespace T { public class Other { public int X() => 2; public int Y() => 3; } }", path: "Other.cs");
        driver = driver.RunGenerators(compilation.ReplaceSyntaxTree(other, edited), TestContext.Current.CancellationToken);
        Assert.All(Reasons(driver), r => Assert.Equal("Cached", r));
    }

    [Fact]
    public void EditingTheClassBodyBelowTheAttributesIsCached()
    {
        var (driver, compilation, def, _) = Start();
        var edited = Parse(Model().Replace("{ }", "{ public static int Helper() => 1; }", StringComparison.Ordinal), path: "Model.cs");
        driver = driver.RunGenerators(compilation.ReplaceSyntaxTree(def, edited), TestContext.Current.CancellationToken);
        Assert.All(Reasons(driver), r => Assert.Equal("Cached", r));
    }

    [Fact]
    public void ChangingANodeReEmits()
    {
        var (driver, compilation, def, _) = Start();
        var edited = Parse(Model().Replace("Vocab=d", "Vocab=d; ", StringComparison.Ordinal).Replace("Tokens=tokens", "Tokens=tokens*1", StringComparison.Ordinal), path: "Model.cs");
        driver = driver.RunGenerators(compilation.ReplaceSyntaxTree(def, edited), TestContext.Current.CancellationToken);
        Assert.DoesNotContain("Cached", Reasons(driver));
    }

    [Fact]
    public void OutputIsStableAcrossCompilationsOfTheSameDefinition()
    {
        var (driver, _, _, _) = Start();
        var (driver2, _, _, _) = Start();
        Assert.Equal(
            driver.GetRunResult().GeneratedTrees.Select(t => t.ToString()).ToArray(),
            driver2.GetRunResult().GeneratedTrees.Select(t => t.ToString()).ToArray());
    }
}
