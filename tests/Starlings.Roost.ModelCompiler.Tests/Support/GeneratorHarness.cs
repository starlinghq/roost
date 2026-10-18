using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Starlings.Roost.Ops;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>Runs the generator in-process over a source text, the way the compiler does.</summary>
public static class GeneratorHarness
{
    public sealed record Result(ImmutableArray<Diagnostic> Generator, ImmutableArray<Diagnostic> Compile, string[] Sources, Compilation Output);

    public static List<MetadataReference> References(bool withOps = true)
    {
        var tpa = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator);
        // The test host's own assemblies include the Roost ones; only the platform belongs in a compilation without them.
        var refs = tpa.Where(p => !Path.GetFileName(p).StartsWith("Starlings.", StringComparison.Ordinal))
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p)).ToList();
        if (withOps)
        {
            foreach (var t in new[] { typeof(IRoostOps), typeof(DeviceSpan), typeof(Starlings.Roost.Graph.OpGraph) })
            {
                refs.Add(MetadataReference.CreateFromFile(t.Assembly.Location));
            }
        }

        return refs;
    }

    public static CSharpCompilation Compile(string source, bool withOps = true, string path = "Model.cs") =>
        CSharpCompilation.Create(
            "T",
            [CSharpSyntaxTree.ParseText(source, path: path)],
            References(withOps),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: false, nullableContextOptions: NullableContextOptions.Enable));

    public static Result Run(string source, bool withOps = true)
    {
        var compilation = Compile(source, withOps);
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new ModelGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var output, out var diagnostics);
        var run = driver.GetRunResult();
        var sources = run.Results.SelectMany(r => r.GeneratedSources).Select(s => s.SourceText.ToString()).ToArray();
        return new Result(diagnostics, output.GetDiagnostics(), sources, output);
    }

    /// <summary>A definition with one embedding node; <paramref name="extra"/> is spliced in as further attributes.</summary>
    public static string Model(string extra = "", string output = "[Output(\"h\")]") =>
        "using Starlings.Roost.ModelCompiler;\n" +
        "namespace T;\n" +
        "[RoostModel(\"m\", Dtype = \"F32\")]\n" +
        "[Dim(\"d\", 8)]\n" +
        "[Weight(\"w\", \"d,d\")]\n" +
        "[Node(\"a\", \"Embedding\", \"Table=w; Output=>h[tokens,d]; Tokens=tokens; Dim=d; Vocab=d\")]\n" +
        extra + "\n" +
        output + "\n" +
        "public static partial class M { }\n";
}
