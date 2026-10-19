using Microsoft.CodeAnalysis;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// The Qwen3.5 gating ops of contract version 3 (GatedDeltaRuleGates, GatedRmsNorm, SigmoidMul) bind like any other op:
/// the generator reads their argument structs from the C3 vocabulary, so a definition that uses all three, with G as a
/// float32 value and a float32 norm weight as Qwen3.5 has them, generates and compiles clean.
/// </summary>
public class Qwen35GatingBindingTests
{
    private const string Source =
        "using Starlings.Roost.ModelCompiler;\n" +
        "namespace T;\n" +
        "[RoostModel(\"m\", Dtype = \"BF16\")]\n" +
        "[Dim(\"d\", 8)]\n" +
        "[Dim(\"vocab\", 16)]\n" +
        "[Dim(\"k_heads\", 2)]\n" +
        "[Dim(\"v_heads\", 2)]\n" +
        "[Dim(\"k_dim\", 4)]\n" +
        "[Dim(\"v_dim\", 4)]\n" +
        "[StateGroup(\"linear\", \"Recurrent\", Layers = \"0:1\", StateShape = \"v_heads,k_dim,v_dim\", ConvShape = \"d,3\", SnapshotSlots = \"2\")]\n" +
        "[Weight(\"embed\", \"vocab,d\")]\n" +
        "[Weight(\"wq\", \"k_heads*k_dim,d\")]\n" +
        "[Weight(\"wv\", \"v_heads*v_dim,d\")]\n" +
        "[Weight(\"wa\", \"v_heads,d\")]\n" +
        "[Weight(\"wb\", \"v_heads,d\")]\n" +
        "[Weight(\"wz\", \"v_heads*v_dim,d\")]\n" +
        "[Weight(\"a_log\", \"v_heads\", Dtype = \"F32\")]\n" +
        "[Weight(\"dt_bias\", \"v_heads\")]\n" +
        "[Weight(\"norm\", \"v_dim\", Dtype = \"F32\")]\n" +
        "[Node(\"e\", \"Embedding\", \"Table=embed; Output=>h[tokens,d]; Tokens=tokens; Dim=d; Vocab=vocab\")]\n" +
        "[Node(\"q\", \"Gemm\", \"A=h; B=wq; D=>q[tokens,k_heads*k_dim]; M=tokens; N=k_heads*k_dim; K=d; TransposeB=true; Alpha=1\")]\n" +
        "[Node(\"v\", \"Gemm\", \"A=h; B=wv; D=>v[tokens,v_heads*v_dim]; M=tokens; N=v_heads*v_dim; K=d; TransposeB=true; Alpha=1\")]\n" +
        "[Node(\"a\", \"Gemm\", \"A=h; B=wa; D=>a[tokens,v_heads]; M=tokens; N=v_heads; K=d; TransposeB=true; Alpha=1\")]\n" +
        "[Node(\"b\", \"Gemm\", \"A=h; B=wb; D=>b[tokens,v_heads]; M=tokens; N=v_heads; K=d; TransposeB=true; Alpha=1\")]\n" +
        "[Node(\"z\", \"Gemm\", \"A=h; B=wz; D=>z[tokens,v_heads*v_dim]; M=tokens; N=v_heads*v_dim; K=d; TransposeB=true; Alpha=1\")]\n" +
        "[Node(\"gates\", \"GatedDeltaRuleGates\", \"A=a; B=b; ALog=a_log; DtBias=dt_bias; G=>g[tokens,v_heads]:F32; Beta=>beta[tokens,v_heads]; " +
        "Tokens=tokens; VHeads=v_heads\")]\n" +
        "[Node(\"r\", \"GatedDeltaRule\", \"Q=q; K=q; V=v; G=g; Beta=beta; Output=>o[tokens,v_heads*v_dim]; State=s:linear.state; Batch=seqs; " +
        "Tokens=tokens; KHeads=k_heads; VHeads=v_heads; KDim=k_dim; VDim=v_dim; Scale=0; UseInitialState=true; L2NormQk=true\", Layers = \"0:1\")]\n" +
        "[Node(\"n\", \"GatedRmsNorm\", \"Input=o; Gate=z; Weight=norm; Output=>y[tokens,v_heads*v_dim]; Rows=tokens*v_heads; Dim=v_dim; Eps=1e-6\")]\n" +
        "[Node(\"og\", \"SigmoidMul\", \"Input=y; Gate=z; Output=>out[tokens,v_heads*v_dim]; Rows=tokens; Dim=v_heads*v_dim\")]\n" +
        "[Output(\"out\")]\n" +
        "public static partial class M { }\n";

    [Fact]
    public void TheGatingOpsGenerateAndCompile()
    {
        var r = GeneratorHarness.Run(Source);

        Assert.Empty(r.Generator);
        Assert.Empty(r.Compile.Where(d => d.Severity >= DiagnosticSeverity.Warning));
        string forward = Assert.Single(r.Sources, s => s.Contains("GatedDeltaRuleArgs", StringComparison.Ordinal));
        Assert.Contains("GatedDeltaRuleGates(", forward, StringComparison.Ordinal);
        Assert.Contains("GatedRmsNorm(", forward, StringComparison.Ordinal);
        Assert.Contains("SigmoidMul(", forward, StringComparison.Ordinal);
    }
}
