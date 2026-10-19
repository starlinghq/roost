using Microsoft.CodeAnalysis;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// GatedDeltaRuleArgs QStride, KStride and VStride (contract version 2) are int fields, so a node binds them like any
/// other scalar: a one-layer recurrent definition with strides on its GatedDeltaRule node generates and compiles clean,
/// and the generated forward assigns each stride from its dimension expression.
/// </summary>
public class GatedDeltaRuleStrideBindingTests
{
    private static string Source(string strides) =>
        "using Starlings.Roost.ModelCompiler;\n" +
        "namespace T;\n" +
        "[RoostModel(\"m\", Dtype = \"F32\")]\n" +
        "[Dim(\"d\", 8)]\n" +
        "[Dim(\"vocab\", 16)]\n" +
        "[Dim(\"k_heads\", 2)]\n" +
        "[Dim(\"v_heads\", 2)]\n" +
        "[Dim(\"k_dim\", 3)]\n" +
        "[Dim(\"v_dim\", 4)]\n" +
        "[StateGroup(\"linear\", \"Recurrent\", Layers = \"0:1\", StateShape = \"v_heads,k_dim,v_dim\", ConvShape = \"d,3\", SnapshotSlots = \"2\")]\n" +
        "[Weight(\"embed\", \"vocab,d\")]\n" +
        "[Weight(\"qkv\", \"2*k_heads*k_dim+v_heads*v_dim,d\")]\n" +
        "[Weight(\"wg\", \"v_heads,d\")]\n" +
        "[Node(\"a\", \"Embedding\", \"Table=embed; Output=>h[tokens,d]; Tokens=tokens; Dim=d; Vocab=vocab\")]\n" +
        "[Node(\"p\", \"Gemm\", \"A=h; B=qkv; D=>x[tokens,2*k_heads*k_dim+v_heads*v_dim]; M=tokens; N=2*k_heads*k_dim+v_heads*v_dim; K=d; TransposeB=true; Alpha=1\")]\n" +
        "[Node(\"g\", \"Gemm\", \"A=h; B=wg; D=>g[tokens,v_heads]; M=tokens; N=v_heads; K=d; TransposeB=true; Alpha=1\")]\n" +
        "[Node(\"r\", \"GatedDeltaRule\", \"Q=x; K=x; V=x; G=g; Beta=g; Output=>o[tokens,v_heads*v_dim]; State=s:linear.state; Batch=seqs; " +
        "Tokens=tokens; KHeads=k_heads; VHeads=v_heads; KDim=k_dim; VDim=v_dim; Scale=0.5; UseInitialState=true; L2NormQk=true" + strides + "\", Layers = \"0:1\")]\n" +
        "[Output(\"o\")]\n" +
        "public static partial class M { }\n";

    [Fact]
    public void StridesBindAsDimExpressionsInTheGeneratedForward()
    {
        var r = GeneratorHarness.Run(Source("; QStride=2*k_heads*k_dim+v_heads*v_dim; KStride=k_heads*k_dim+1; VStride=v_heads*v_dim"));

        Assert.Empty(r.Generator);
        Assert.Empty(r.Compile.Where(d => d.Severity >= DiagnosticSeverity.Warning));
        string forward = Assert.Single(r.Sources, s => s.Contains("GatedDeltaRuleArgs", StringComparison.Ordinal));
        Assert.Contains("a.QStride = (int)(", forward, StringComparison.Ordinal);
        Assert.Contains("a.KStride = (int)(", forward, StringComparison.Ordinal);
        Assert.Contains("a.VStride = (int)(", forward, StringComparison.Ordinal);

        // Without bindings the fields stay at their default of 0 (contiguous): the generator assigns nothing.
        var plain = GeneratorHarness.Run(Source(string.Empty));
        Assert.Empty(plain.Generator);
        Assert.DoesNotContain("Stride =", Assert.Single(plain.Sources, s => s.Contains("GatedDeltaRuleArgs", StringComparison.Ordinal)), StringComparison.Ordinal);
    }
}
