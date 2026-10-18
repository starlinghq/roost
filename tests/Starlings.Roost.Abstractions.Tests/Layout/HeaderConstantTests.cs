using System.Globalization;
using Starlings.Roost.Abi;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Tests.Layout;

/// <summary>Every <c>#define</c> family in the C headers matches its C# enum, member for member and value for value.</summary>
public sealed class HeaderConstantTests
{
    // Longest prefix wins, so ROOST_CACHE_OP_ is not read as ROOST_OP_.
    private static readonly (string Prefix, Type Enum)[] s_families =
    [
        ("ROOST_STATUS_", typeof(KernelStatus)),
        ("ROOST_DTYPE_", typeof(DType)),
        ("ROOST_LAYOUT_", typeof(TensorLayout)),
        ("ROOST_OP_", typeof(OpId)),
        ("ROOST_FLAG_", typeof(VariantTraits)),
        ("ROOST_CONTEXT_", typeof(ContextOptions)),
        ("ROOST_ROPE_", typeof(RopeKind)),
        ("ROOST_ATTENTION_", typeof(AttentionMode)),
        ("ROOST_GELU_", typeof(GeluApproximation)),
        ("ROOST_MATH_", typeof(MathMode)),
        ("ROOST_PLAN_FLAG_", typeof(PlanOptions)),
        ("ROOST_SEQ_FLAG_", typeof(SequenceOptions)),
        ("ROOST_SAMPLE_FLAG_", typeof(SamplingOptions)),
        ("ROOST_CACHE_OP_", typeof(CacheOpKind)),
        ("ROOST_TIER_", typeof(BlockTier)),
        ("ROOST_KV_TRANSFER_", typeof(KvTransferKind)),
        ("ROOST_HEAD_", typeof(HeadKind)),
        ("ROOST_FAULT_", typeof(FaultClass)),
        ("ROOST_FINISH_", typeof(FinishReason)),
        ("ROOST_RING_SLOT_", typeof(RingSlotKind)),
        ("ROOST_HANDSHAKE_", typeof(HandshakeState)),
        ("ROOST_REJECT_", typeof(HandshakeReject)),
        ("ROOST_WORKER_", typeof(WorkerState)),
    ];

    // Defines that are not members of an enum family.
    private static readonly HashSet<string> s_scalars =
    [
        "ROOST_ABI_MAJOR", "ROOST_ABI_MINOR", "ROOST_DTYPE_COUNT", "ROOST_SHM_LAYOUT_VERSION", "ROOST_SHM_ALIGNMENT",
        "ROOST_SHM_MAGIC_SEGMENT", "ROOST_SHM_MAGIC_PLAN", "ROOST_SHM_MAGIC_RESULT", "ROOST_SHM_MAGIC_RING",
    ];

    public static TheoryData<string> Prefixes() => [.. s_families.Select(f => f.Prefix)];

    [Fact]
    public void EveryDefineBelongsToAFamilyOrIsAKnownScalar()
    {
        var unclaimed = NativeLayout.Defines.Keys
            .Where(name => !s_scalars.Contains(name) && FamilyOf(name) is null)
            .ToList();
        Assert.Empty(unclaimed);
    }

    [Theory]
    [MemberData(nameof(Prefixes))]
    public void EnumMatchesDefines(string prefix)
    {
        var type = s_families.Single(f => f.Prefix == prefix).Enum;
        var defines = NativeLayout.Defines
            .Where(d => !s_scalars.Contains(d.Key) && FamilyOf(d.Key) == prefix)
            .ToDictionary(d => Normalise(d.Key[prefix.Length..]), d => d.Value, StringComparer.Ordinal);
        var members = Enum.GetNames(type)
            .ToDictionary(n => Normalise(n), n => Convert.ToInt64(Enum.Parse(type, n), CultureInfo.InvariantCulture), StringComparer.Ordinal);

        // A zero "None" or "Empty" member may exist in C# only.
        foreach (var extra in members.Keys.Except(defines.Keys).ToList())
        {
            Assert.True(members[extra] == 0 && extra is "NONE", $"{type.Name}.{extra} has no #define {prefix}{extra}");
            members.Remove(extra);
        }

        Assert.Equal(defines.OrderBy(d => d.Key, StringComparer.Ordinal), members.OrderBy(m => m.Key, StringComparer.Ordinal));
    }

    private static string? FamilyOf(string name) =>
        s_families.Select(f => f.Prefix).Where(name.StartsWith).OrderByDescending(p => p.Length).FirstOrDefault();

    private static string Normalise(string name) => name.Replace("_", "", StringComparison.Ordinal).ToUpperInvariant();
}
