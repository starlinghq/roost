using System.Reflection;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Tests.Contracts;

/// <summary>C3: the op vocabulary declares every name of build plan section 5 and matches the C1 op ids.</summary>
public sealed class OpVocabularyTests
{
    // Build plan section 5, C3, as written (cache ops and Argmax/Sample spelled as their M0 members).
    private static readonly string[] s_m0 =
    [
        "RmsNorm", "QkNorm", "Rope", "PagedAttention", "SiluAndMul", "GeluAndMul", "Gemm", "Embedding", "LmHead",
        "Residual", "CacheWrite", "CacheCopy", "Argmax", "Sample",
    ];

    private static readonly string[] s_reserved =
    [
        "GatedDeltaRule", "CausalConv1d", "RecurrentStateCopy", "RecurrentStateSnapshot", "SlidingWindowAttention",
        "RopeScaled", "Quantize", "Dequantize", "GemmW4A16", "GemmFp8", "TreeAttention", "RejectSample",
        "MoeRouteTopK", "MoePermute", "MoeUnpermute", "GroupedGemm", "FusedMoe",
        "LoraShrink", "LoraExpand",
        "AllReduce", "AllGather", "ReduceScatter", "AllToAll", "MlaAttention",
        "PatchEmbed", "BidirectionalAttention", "MelSpectrogram", "AudioConvSubsample", "DequantizeGguf",
    ];

    // Added by contract version 3 for Qwen3.5-0.8B (decision 15), numbered in the hybrid block; not in section 5's table.
    private static readonly string[] s_qwen35 = ["GatedDeltaRuleGates", "GatedRmsNorm", "SigmoidMul"];

    private static readonly string[] s_fused =
        ["AddRmsNorm", "AddRmsNormQuant", "SiluMulQuant", "QkvQkNormRopeCacheWrite", "GemmLoraEpilogue"];

    private static MethodInfo[] OpMethods() =>
        [.. typeof(IRoostOps).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.ReturnType == typeof(void) && m.GetParameters().Length == 1)];

    [Fact]
    public void EveryNameInSection5IsAMember()
    {
        var names = OpMethods().Select(m => m.Name).ToHashSet(StringComparer.Ordinal);
        Assert.All(s_m0.Concat(s_reserved).Concat(s_qwen35).Concat(s_fused), n => Assert.Contains(n, names));
        Assert.Equal(s_m0.Length + s_reserved.Length + s_qwen35.Length + s_fused.Length, names.Count);
    }

    [Fact]
    public void OpIdsAndMembersCorrespondOneToOne()
    {
        var ids = Enum.GetNames<OpId>().Where(n => n != nameof(OpId.None)).Order(StringComparer.Ordinal);
        Assert.Equal(ids, OpMethods().Select(m => m.Name).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void M0IsAbstractAndTheRestHaveDefaults()
    {
        foreach (var method in OpMethods())
        {
            var phase = OpCatalog.PhaseOf(Enum.Parse<OpId>(method.Name));
            Assert.True((phase == OpPhase.M0) == method.IsAbstract, $"{method.Name} ({phase}) abstract={method.IsAbstract}");
        }
    }

    [Fact]
    public void ArgumentsArePassedInAsStructsNamedAfterTheOp()
    {
        foreach (var method in OpMethods())
        {
            var parameter = method.GetParameters()[0];
            Assert.True(parameter.IsIn, method.Name);
            var type = parameter.ParameterType.GetElementType()!;
            Assert.True(type.IsValueType, method.Name);
            Assert.Equal(method.Name + "Args", type.Name);
        }
    }

    [Fact]
    public void ReservedAndFusedOpsThrowUntilImplemented()
    {
        Assert.True(Supports<M0Only>(OpId.RmsNorm));
        Assert.False(Supports<M0Only>(OpId.GatedDeltaRule));
        Assert.False(SupportsAddRmsNorm<M0Only>());
        Assert.Throws<NotSupportedException>(CallGatedDeltaRule<M0Only>);
        Assert.Throws<NotSupportedException>(CallAddRmsNorm<M0Only>);
        CallRmsNorm<M0Only>();
        Assert.Equal(1, M0Only.Calls);
    }

    private static bool Supports<T>(OpId op) where T : struct, IRoostOps => T.Supports(op);

    private static bool SupportsAddRmsNorm<T>() where T : struct, IRoostOps => T.SupportsAddRmsNorm;

    private static void CallGatedDeltaRule<T>() where T : struct, IRoostOps => T.GatedDeltaRule(default);

    private static void CallAddRmsNorm<T>() where T : struct, IRoostOps => T.AddRmsNorm(default);

    private static void CallRmsNorm<T>() where T : struct, IRoostOps => T.RmsNorm(default);

    /// <summary>A backend that implements only M0 compiles: the reserved members have defaults.</summary>
    private struct M0Only : IRoostOps
    {
        public static int Calls;

        public static void RmsNorm(in RmsNormArgs a) => Calls++;
        public static void QkNorm(in QkNormArgs a) => Calls++;
        public static void Rope(in RopeArgs a) => Calls++;
        public static void PagedAttention(in PagedAttentionArgs a) => Calls++;
        public static void SiluAndMul(in SiluAndMulArgs a) => Calls++;
        public static void GeluAndMul(in GeluAndMulArgs a) => Calls++;
        public static void Gemm(in GemmArgs a) => Calls++;
        public static void Embedding(in EmbeddingArgs a) => Calls++;
        public static void LmHead(in LmHeadArgs a) => Calls++;
        public static void Residual(in ResidualArgs a) => Calls++;
        public static void CacheWrite(in CacheWriteArgs a) => Calls++;
        public static void CacheCopy(in CacheCopyArgs a) => Calls++;
        public static void Argmax(in ArgmaxArgs a) => Calls++;
        public static void Sample(in SampleArgs a) => Calls++;
    }
}
