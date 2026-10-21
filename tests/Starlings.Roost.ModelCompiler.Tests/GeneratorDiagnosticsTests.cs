using Microsoft.CodeAnalysis;
using static Starlings.Roost.ModelCompiler.Tests.GeneratorHarness;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>Every diagnostic the generator reports, with what it names, and that a good definition compiles clean.</summary>
public static class DiagnosticExtensions
{
    public static string Msg(this Diagnostic d) => d.GetMessage(System.Globalization.CultureInfo.InvariantCulture);
}

public class GeneratorDiagnosticsTests
{
    private static Diagnostic Single(Result r, string id)
    {
        var d = r.Generator.Where(x => x.Id == id).ToList();
        Assert.Single(d);
        return d[0];
    }

    [Fact]
    public void AValidDefinitionGeneratesCodeThatCompilesWithoutDiagnostics()
    {
        var r = Run(Model());
        Assert.Empty(r.Generator);
        Assert.Empty(r.Compile.Where(d => d.Severity >= DiagnosticSeverity.Warning));
        Assert.Contains(r.Sources, s => s.Contains("public static void Forward<TOps>", StringComparison.Ordinal));
    }

    [Fact]
    public void MissingOpNamesTheOpTheNodeAndTheKnownOpsAndPointsAtTheAttribute()
    {
        var r = Run(Model("[Node(\"b\", \"FlashDecode\", \"Q=h=>o[tokens,d]\")]"));
        var d = Single(r, "ROOST001");
        string m = d.Msg();
        Assert.Equal(DiagnosticSeverity.Error, d.Severity);
        Assert.Contains("'FlashDecode'", m, StringComparison.Ordinal);
        Assert.Contains("node 'b'", m, StringComparison.Ordinal);
        Assert.Contains("known ops:", m, StringComparison.Ordinal);
        Assert.Contains("PagedAttention", m, StringComparison.Ordinal);
        Assert.Contains("GatedDeltaRule", m, StringComparison.Ordinal);
        Assert.Equal(7, d.Location.GetLineSpan().StartLinePosition.Line + 1);
        Assert.DoesNotContain(r.Sources, s => s.Contains("Forward<TOps>", StringComparison.Ordinal));
    }

    [Fact]
    public void NodesDownstreamOfAMissingOpAreNotReportedAgain()
    {
        var r = Run(Model(
            "[Node(\"b\", \"FlashDecode\", \"Q=h=>o[tokens,d]\")]\n[Node(\"c\", \"RmsNorm\", \"Input=o; Weight=w; Output=>n[tokens,d]; Rows=tokens; Dim=d\")]"));
        Assert.Equal(["ROOST001"], r.Generator.Select(d => d.Id).ToArray());
    }

    [Fact]
    public void UnknownValueNamesTheNodeTheFieldAndTheName()
    {
        var d = Single(Run(Model("[Node(\"b\", \"RmsNorm\", \"Input=nope; Weight=w; Output=>n[tokens,d]; Rows=tokens; Dim=d\")]")), "ROOST002");
        Assert.Contains("node", d.Msg(), StringComparison.Ordinal);
        Assert.Contains("'Input'", d.Msg(), StringComparison.Ordinal);
        Assert.Contains("'nope'", d.Msg(), StringComparison.Ordinal);
    }

    [Fact]
    public void UnknownFieldListsTheFieldsOfTheArgumentStruct()
    {
        var d = Single(Run(Model("[Node(\"b\", \"RmsNorm\", \"Input=h; Bogus=1; Output=>n[tokens,d]\")]")), "ROOST003");
        Assert.Contains("'Bogus'", d.Msg(), StringComparison.Ordinal);
        Assert.Contains("Eps", d.Msg(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("Input=h; Weight=w; Output=>n[tokens,d]; Rows=tokens; Dim=d; Eps=abc")]
    [InlineData("Input=h; Weight=w; Output=>n[tokens,d]; Rows=tokens; Dim=d; DType=Fp99")]
    [InlineData("Input=h; Weight=w; Output=>n[tokens,d]; Rows=tokens; Dim=d; Input=h")]
    [InlineData("Input=h; Weight=w; Output=>n[tokens,d]; Rows=h")]
    public void BindingThatDoesNotFitItsFieldIsReported(string args)
    {
        var r = Run(Model("[Node(\"b\", \"RmsNorm\", \"" + args + "\")]"));
        Assert.Contains(r.Generator, d => d.Id is "ROOST004" or "ROOST006");
    }

    [Fact]
    public void AnOutputWithoutAShapeIsReported()
    {
        var d = Single(Run(Model("[Node(\"b\", \"RmsNorm\", \"Input=h; Weight=w; Output=>n; Rows=tokens; Dim=d\")]")), "ROOST006");
        Assert.Contains("needs a shape", d.Msg(), StringComparison.Ordinal);
    }

    [Fact]
    public void AStepSymbolIsOnlyAllowedAsTheLeadingDimension()
    {
        var d = Single(Run(Model("[Node(\"b\", \"RmsNorm\", \"Input=h; Weight=w; Output=>n[d,tokens]; Rows=tokens; Dim=d\")]")), "ROOST006");
        Assert.Contains("first dimension", d.Msg(), StringComparison.Ordinal);
    }

    [Fact]
    public void ANameDeclaredTwiceIsReported()
    {
        var d = Single(Run(Model("[Node(\"b\", \"RmsNorm\", \"Input=h; Weight=w; Output=>h[tokens,d]; Rows=tokens; Dim=d\")]")), "ROOST007");
        Assert.Contains("'h'", d.Msg(), StringComparison.Ordinal);
    }

    [Fact]
    public void AModelWithoutOutputAndABadDtypeAreReported()
    {
        Assert.Contains("declares no [Output]", Single(Run(Model(output: string.Empty)), "ROOST008").Msg(), StringComparison.Ordinal);
        var bad = Run(Model().Replace("Dtype = \"F32\"", "Dtype = \"Q9\"", StringComparison.Ordinal));
        Assert.Contains("'Q9'", Single(bad, "ROOST008").Msg(), StringComparison.Ordinal);
    }

    [Fact]
    public void ReadingAValueAfterAnOpOverwroteItInPlaceIsReported()
    {
        // QkNorm normalises q and k in place; reading q again afterwards would read the new values under the old name.
        string extra =
            "[Node(\"g\", \"Embedding\", \"Table=w; Output=>g[tokens,d]; Tokens=tokens; Dim=d; Vocab=d\")]\n" +
            "[Node(\"qk\", \"QkNorm\", \"Q=h=>h2; K=g=>g2; QWeight=w; KWeight=w; Tokens=tokens; QHeads=1; KvHeads=1; HeadDim=d\")]\n" +
            "[Node(\"late\", \"Residual\", \"A=h; B=g2; Output=>r[tokens,d]; Elements=tokens*d\")]";
        var d = Single(Run(Model(extra)), "ROOST009");
        Assert.Contains("'h'", d.Msg(), StringComparison.Ordinal);
        Assert.Contains("'qk'", d.Msg(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("[Weight(\"l{i}.x\", \"d\", Layers = \"3:1\")]")]
    [InlineData("[Weight(\"l{x}.x\", \"d\", Layers = \"0:2\")]")]
    [InlineData("[Weight(\"l{i}.x\", \"d\")]")]
    [InlineData("[Weight(\"l{i}.x\", \"d\", Layers = \"0:4%0=1\")]")]
    public void ABadLayerSetOrPlaceholderIsReported(string weight)
    {
        Assert.Contains(Run(Model(weight)).Generator, d => d.Id == "ROOST010");
    }

    [Theory]
    [InlineData("[StateGroup(\"g\", \"Cross\", Layers = \"0:1\")]")]
    [InlineData("[StateGroup(\"g\", \"Full\", KvHeads = \"1\", HeadSize = \"8\", BlockSize = \"16\")]")]
    [InlineData("[StateGroup(\"g\", \"Full\", Layers = \"0:1\", KvHeads = \"1\", HeadSize = \"8\")]")]
    [InlineData("[StateGroup(\"g\", \"Recurrent\", Layers = \"0:1\")]")]
    [InlineData("[StateGroup(\"g\", \"Sliding\", Layers = \"0:1\", KvHeads = \"1\", HeadSize = \"8\", BlockSize = \"16\")]")]
    public void AnInvalidStateGroupIsReported(string group)
    {
        Assert.Contains(Run(Model(group)).Generator, d => d.Id == "ROOST011");
    }

    [Fact]
    public void SlidingAndFullGroupsDescribeThemselves()
    {
        var r = Run(Model(
            "[StateGroup(\"swa\", \"Sliding\", Layers = \"0:3\", KvHeads = \"2\", HeadSize = \"8\", BlockSize = \"16\", Window = \"128\", Dtype = \"F16\")]\n" +
            "[StateGroup(\"full\", \"Full\", Layers = \"3\", KvHeads = \"2\", HeadSize = \"8\", BlockSize = \"32\")]"));
        Assert.Empty(r.Generator);
        Assert.Empty(r.Compile.Where(d => d.Severity >= DiagnosticSeverity.Warning));
        string text = Assert.Single(r.Sources, s => s.Contains("CreateStateGroups", StringComparison.Ordinal));
        Assert.Contains("StateGroupKind.Sliding", text, StringComparison.Ordinal);
        Assert.Contains("Window = 128", text, StringComparison.Ordinal);
        Assert.Contains("DType = global::Starlings.Roost.Abi.DType.F16", text, StringComparison.Ordinal);
        // 3 layers x (K and V) x 16 tokens x 2 heads x 8 dims x 2 bytes.
        Assert.Contains("BytesPerBlock = " + (3 * 2 * 16 * 2 * 8 * 2) + "L", text, StringComparison.Ordinal);
        Assert.Contains("new global::Starlings.Roost.State.LayerKvShape(2, 2, 8)", text, StringComparison.Ordinal);
        Assert.Contains("\"swa.l0.k\"", text, StringComparison.Ordinal);
        Assert.Contains("\"full.l3.v\"", text, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("Output=>n[tokens,d]:Q9")]
    [InlineData("Output=>n[tokens,d]:4")]
    public void AnOutputDtypeThatIsNotAnElementTypeNameIsReported(string output)
    {
        var r = Run(Model("[Node(\"b\", \"RmsNorm\", \"Input=h; Weight=w; " + output + "; Rows=tokens; Dim=d\")]"));
        Assert.Contains(r.Generator, d => d.Id == "ROOST006");
    }

    [Fact]
    public void AFusedOpWithoutAKnownUnfusedFormIsReported()
    {
        var d = Single(Run(Model("[Node(\"b\", \"SiluMulQuant\", \"Input=h; Output=>q[tokens,d]; Rows=tokens; Dim=d\")]")), "ROOST012");
        Assert.Contains("'SiluMulQuant'", d.Msg(), StringComparison.Ordinal);
    }

    private static string Attention(string window) =>
        "[Node(\"attn\", \"PagedAttention\", \"Q=q; KCache=h; VCache=h; Output=>ao[tokens,d]; Batch=1; Tokens=tokens; QHeads=1; KvHeads=1; HeadDim=d; BlockSize=16; Causal=true" + window + "\")]";

    [Fact]
    public void AnUnboundPagedAttentionWindowLeftIsAnErrorNamingTheNodeAndSayingToBindMinusOne()
    {
        var r = Run(Model(Attention(string.Empty)));
        var d = Single(r, "ROOST015");
        Assert.Equal(DiagnosticSeverity.Error, d.Severity);
        Assert.Contains("'attn'", d.Msg(), StringComparison.Ordinal);
        Assert.Contains("WindowLeft=-1", d.Msg(), StringComparison.Ordinal);
        Assert.DoesNotContain(r.Sources, s => s.Contains("Forward<TOps>", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("; WindowLeft=-1")]
    [InlineData("; WindowLeft=128")]
    public void ABoundPagedAttentionWindowLeftRaisesNoWindowDiagnostic(string window)
    {
        var r = Run(Model(Attention(window)));
        Assert.DoesNotContain(r.Generator, d => d.Id == "ROOST015");
    }

    [Fact]
    public void AReservedWeightNothingReadsGetsNoWarningButStillHasASlot()
    {
        var r = Run(Model("[Weight(\"orphan\", \"d\", Reserved = true)]"));
        Assert.DoesNotContain(r.Generator, d => d.Id == "ROOST014");
        Assert.Contains(r.Sources, s => s.Contains("\"orphan\"", StringComparison.Ordinal));
        Assert.Empty(r.Compile.Where(d => d.Severity >= DiagnosticSeverity.Warning));
    }

    [Fact]
    public void AWeightNothingReadsIsAWarningAndStillGenerates()
    {
        var r = Run(Model("[Weight(\"orphan\", \"d\")]"));
        var d = Single(r, "ROOST014");
        Assert.Equal(DiagnosticSeverity.Warning, d.Severity);
        Assert.Contains("'orphan'", d.Msg(), StringComparison.Ordinal);
        Assert.Contains(r.Sources, s => s.Contains("\"orphan\"", StringComparison.Ordinal));
    }

    [Fact]
    public void StateGroupsWithoutTheStateContractTypesAreReported()
    {
        // A compilation that has an IRoostOps but not Starlings.Roost.Abstractions' state types.
        string source = Model("[StateGroup(\"g\", \"Full\", Layers = \"0:1\", KvHeads = \"1\", HeadSize = \"8\", BlockSize = \"16\")]");
        var d = Single(Run(source, withOps: false, extra: "namespace Starlings.Roost.Ops { public interface IRoostOps { } }"), "ROOST013");
        Assert.Equal(DiagnosticSeverity.Error, d.Severity);
        Assert.Contains("StateGroupDescriptor", d.Msg(), StringComparison.Ordinal);
    }

    [Fact]
    public void ACompilationWithoutIRoostOpsIsReportedOnce()
    {
        var r = Run(Model(), withOps: false);
        Assert.Contains("ROOST005", r.Generator.Select(d => d.Id));
    }

    [Fact]
    public void GeneratedTextIsDeterministicAndUsesUnixLineEndings()
    {
        string a = Assert.Single(Run(Model()).Sources, s => s.Contains("Forward<TOps>", StringComparison.Ordinal));
        string b = Assert.Single(Run(Model()).Sources, s => s.Contains("Forward<TOps>", StringComparison.Ordinal));
        Assert.Equal(a, b);
        Assert.DoesNotContain('\r', a);
    }
}
