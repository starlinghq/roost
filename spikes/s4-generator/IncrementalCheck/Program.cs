using System;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Spike.Generator;

namespace Spike.IncrementalCheck;

// Drives the generator directly with step tracking and edits the compilation the way an IDE or an incremental build
// does, reporting whether the generator's output step was Cached/Unchanged (no re-emit) or re-ran.
internal static class Program
{
    private const string Def = """
        using Spike.Ops;
        namespace T;
        [RoostModel(Hidden = 16, Intermediate = 32, Vocab = 64)]
        [Node("h0", "Embedding", "tokens", Weight = "embed", Out = "H")]
        [Node("n0", "RmsNorm", "h0", Weight = "norm")]
        [Node("logits", "LmHead", "n0", Weight = "lm_head", Out = "V")]
        public static partial class M
        {
        //BODY
        }
        """;

    private static int Main()
    {
        var tpa = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator);
        var refs = tpa.Select(p => MetadataReference.CreateFromFile(p)).Append(MetadataReference.CreateFromFile(typeof(Spike.Ops.IRoostOps).Assembly.Location)).ToList();
        SyntaxTree Tree(string src, string path) => CSharpSyntaxTree.ParseText(src, path: path);
        var opts = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true);

        double ms = 0;
        var def0 = Tree(Def.Replace("//BODY", string.Empty), "Model.cs");
        var other0 = Tree("namespace T { public class Other { public int X() => 1; } }", "Other.cs");
        var comp = CSharpCompilation.Create("T", new[] { def0, other0 }, refs, opts);
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new[] { new ModelGenerator().AsSourceGenerator() }, driverOptions: new GeneratorDriverOptions(default, trackIncrementalGeneratorSteps: true));

        { var sw = System.Diagnostics.Stopwatch.StartNew(); driver = driver.RunGenerators(comp); ms = sw.Elapsed.TotalMilliseconds; }
        Console.WriteLine("baseline:              " + Describe(driver) + $" generator={ms:F1} ms");

        // 1. Edit an unrelated file.
        var other1 = Tree("namespace T { public class Other { public int X() => 2; public int Y() => 3; } }", "Other.cs");
        comp = comp.ReplaceSyntaxTree(other0, other1);
        { var sw = System.Diagnostics.Stopwatch.StartNew(); driver = driver.RunGenerators(comp); ms = sw.Elapsed.TotalMilliseconds; }
        Console.WriteLine("edit unrelated file:   " + Describe(driver) + $" generator={ms:F1} ms");

        // 2. Edit the definition file below the attributes (a method body in the model class).
        var def1 = Tree(Def.Replace("//BODY", "    public static int Helper() => 1;"), "Model.cs");
        comp = comp.ReplaceSyntaxTree(def0, def1);
        { var sw = System.Diagnostics.Stopwatch.StartNew(); driver = driver.RunGenerators(comp); ms = sw.Elapsed.TotalMilliseconds; }
        Console.WriteLine("edit def file (body):  " + Describe(driver) + $" generator={ms:F1} ms");

        // 3. Change a node in the definition.
        var def2 = Tree(Def.Replace("//BODY", "    public static int Helper() => 1;").Replace("Weight = \"norm\"", "Weight = \"norm2\""), "Model.cs");
        comp = comp.ReplaceSyntaxTree(def1, def2);
        { var sw = System.Diagnostics.Stopwatch.StartNew(); driver = driver.RunGenerators(comp); ms = sw.Elapsed.TotalMilliseconds; }
        Console.WriteLine("edit a node (weight):  " + Describe(driver) + $" generator={ms:F1} ms");

        // 4. Insert a comment line above the attributes: locations in the model shift, so the model changes.
        var def3 = Tree("// a comment line\n" + def2.ToString(), "Model.cs");
        comp = comp.ReplaceSyntaxTree(def2, def3);
        { var sw = System.Diagnostics.Stopwatch.StartNew(); driver = driver.RunGenerators(comp); ms = sw.Elapsed.TotalMilliseconds; }
        Console.WriteLine("comment above attrs:   " + Describe(driver) + $" generator={ms:F1} ms");
        return 0;
    }

    private static string Describe(GeneratorDriver d)
    {
        var r = d.GetRunResult().Results[0];
        var outs = r.TrackedOutputSteps.SelectMany(kv => kv.Value).SelectMany(s => s.Outputs).Select(o => o.Reason.ToString()).ToArray();
        var diag = r.Diagnostics.Length;
        return $"output step(s): [{string.Join(",", outs)}] sources={r.GeneratedSources.Length} diagnostics={diag}";
    }
}
