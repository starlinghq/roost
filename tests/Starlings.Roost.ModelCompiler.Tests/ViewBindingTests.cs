using Microsoft.CodeAnalysis;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>The view binding <c>FIELD=value[start:end]</c>: a column slice of a token-major node output, with checks on what it may name.</summary>
public class ViewBindingTests
{
    private static string Source(string q, string k = "x[8:16]", string v = "x[16:32]") =>
        "using Starlings.Roost.ModelCompiler;\n" +
        "namespace T;\n" +
        "[RoostModel(\"m\", Dtype = \"F32\")]\n" +
        "[Dim(\"d\", 8)]\n" +
        "[Dim(\"vocab\", 16)]\n" +
        "[Dim(\"key\", 8)]\n" +
        "[StateGroup(\"linear\", \"Recurrent\", Layers = \"0:1\", StateShape = \"4,2,4\", ConvShape = \"32,3\", SnapshotSlots = \"2\")]\n" +
        "[Weight(\"embed\", \"vocab,d\")]\n" +
        "[Weight(\"qkv\", \"32,d\")]\n" +
        "[Weight(\"wg\", \"4,d\")]\n" +
        "[Node(\"a\", \"Embedding\", \"Table=embed; Output=>h[tokens,d]; Tokens=tokens; Dim=d; Vocab=vocab\")]\n" +
        "[Node(\"p\", \"Gemm\", \"A=h; B=qkv; D=>x[tokens,32]; M=tokens; N=32; K=d; TransposeB=true; Alpha=1; WeightDType=F32\")]\n" +
        "[Node(\"g\", \"Gemm\", \"A=h; B=wg; D=>g[tokens,4]; M=tokens; N=4; K=d; TransposeB=true; Alpha=1; WeightDType=F32\")]\n" +
        "[Node(\"r\", \"GatedDeltaRule\", \"Q=" + q + "; K=" + k + "; V=" + v + "; QStride=32; KStride=32; VStride=32; G=g; Beta=g; Output=>o[tokens,16]; State=s:linear.state; Batch=seqs; " +
        "Tokens=tokens; KHeads=2; VHeads=4; KDim=4; VDim=4; Scale=0.5; UseInitialState=true; L2NormQk=true\", Layers = \"0:1\")]\n" +
        "[Output(\"o\")]\n" +
        "public static partial class M { }\n";

    private static string[] Ids(GeneratorHarness.Result r) => r.Generator.Select(d => d.Id).ToArray();

    [Fact]
    public void AViewBecomesAnAttributeAndAnOffsetSpanInTheForward()
    {
        var r = GeneratorHarness.Run(Source("x[0:key]"));
        Assert.Empty(r.Generator);
        Assert.Empty(r.Compile.Where(d => d.Severity >= DiagnosticSeverity.Warning));
        string code = Assert.Single(r.Sources, s => s.Contains("GatedDeltaRuleArgs", StringComparison.Ordinal));
        // F32 columns are 4 bytes: K starts 8 columns in (32 bytes), V 16 columns in (64 bytes); Q is the whole value from column 0.
        Assert.Contains("a.Q = View(", code, StringComparison.Ordinal);
        Assert.Contains(", 0L, 0L);", code, StringComparison.Ordinal);
        Assert.Contains(", 32L, 8L);", code, StringComparison.Ordinal);
        Assert.Contains(", 64L, 16L);", code, StringComparison.Ordinal);
        Assert.Contains("{ \"view.K\", \"8:16\" }", code, StringComparison.Ordinal);
        Assert.Contains("{ \"view.Q\", \"0:8\" }", code, StringComparison.Ordinal);
    }

    [Fact]
    public void ThreeViewsOfOneValueKeepTheValueAliveOnce()
    {
        var r = GeneratorHarness.Run(Source("x[0:8]"));
        string code = Assert.Single(r.Sources, s => s.Contains("OpNode(3, \"GatedDeltaRule\"", StringComparison.Ordinal));
        // The node reads the parent value (the same id) through Q, K and V.
        var m = System.Text.RegularExpressions.Regex.Match(code, @"OpNode\(3, ""GatedDeltaRule"", new int\[\] \{ (\d+), (\d+), (\d+),");
        Assert.True(m.Success);
        Assert.Equal(m.Groups[1].Value, m.Groups[2].Value);
        Assert.Equal(m.Groups[1].Value, m.Groups[3].Value);
    }

    [Theory]
    [InlineData("x[8]", "ROOST004")]
    [InlineData("x[0:key+99]", "ROOST004")]
    [InlineData("x[4:4]", "ROOST004")]
    [InlineData("x[-1:4]", "ROOST004")]
    [InlineData("x[0:nope]", "ROOST004")]
    [InlineData("qkv[0:8]", "ROOST004")]
    [InlineData("embed[0:8]", "ROOST004")]
    [InlineData("nothing[0:8]", "ROOST002")]
    public void BadViewsAreDiagnosedAtTheNode(string q, string id)
    {
        var r = GeneratorHarness.Run(Source(q));
        Assert.Contains(id, Ids(r));
        Assert.DoesNotContain(r.Sources, s => s.Contains("TOps.GatedDeltaRule", StringComparison.Ordinal));
    }

    [Fact]
    public void ANodeOutputWithAViewCannotBeWrittenInPlace()
    {
        var r = GeneratorHarness.Run(Source("x[0:8]=>y"));
        Assert.NotEmpty(r.Generator);
    }
}
