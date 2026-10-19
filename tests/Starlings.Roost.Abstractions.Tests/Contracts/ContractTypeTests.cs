using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;
using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.Mesh;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Tests.Contracts;

public sealed class ContractTypeTests
{
    [Fact]
    public void GraphEnumsMirrorAbiEnums()
    {
        Assert.Equal(
            Enum.GetValues<DType>().Select(v => (Name: v.ToString(), Value: (int)v)),
            Enum.GetValues<GraphDType>().Select(v => (Name: v.ToString(), Value: (int)v)));
        Assert.Equal(
            Enum.GetValues<TensorLayout>().Select(v => (Name: v.ToString(), Value: (int)v)),
            Enum.GetValues<GraphLayout>().Select(v => (Name: v.ToString(), Value: (int)v)));
    }

    [Fact]
    public void ErrorCodesHaveDistinctStableNamesAndStatuses()
    {
        var codes = Enum.GetValues<RoostErrorCode>();
        Assert.Equal(9, codes.Length);
        Assert.Equal(codes.Length, codes.Select(RoostError.NameOf).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(429, RoostError.StatusOf(RoostErrorCode.QueueFull));
        Assert.Equal(400, RoostError.StatusOf(RoostErrorCode.InvalidOption));
        Assert.Equal(503, RoostError.StatusOf(RoostErrorCode.WorkerRestarted));
        Assert.Equal(500, RoostError.StatusOf(RoostErrorCode.Internal));
    }

    [Fact]
    public void ErrorBodyIsStable()
    {
        var buffer = new ArrayBufferWriter<byte>();
        new RoostError(RoostErrorCode.InvalidOption, "Unimplemented keys: uniqueItems", "response_format").WriteHttpBody(buffer);
        Assert.Equal(
            """{"error":{"message":"Unimplemented keys: uniqueItems","type":"roost_error","param":"response_format","code":"invalid_option"}}""",
            Encoding.UTF8.GetString(buffer.WrittenSpan));
    }

    [Fact]
    public void FaultRulesFollowS8()
    {
        Assert.Equal(RecoveryAction.RetryStep, FaultRules.ActionFor(FaultClass.Recoverable));
        Assert.All(
            new[] { FaultClass.Sticky, FaultClass.Hang, FaultClass.Stalled, FaultClass.Exited },
            f => Assert.Equal(RecoveryAction.Respawn, FaultRules.ActionFor(f)));
        Assert.Equal(FaultClass.Hang, FaultRules.ClassifyMissedDeadline(heartbeatAdvanced: true));
        Assert.Equal(FaultClass.Stalled, FaultRules.ClassifyMissedDeadline(heartbeatAdvanced: false));
    }

    [Fact]
    public void SingleRankMesh()
    {
        var mesh = DeviceMesh.SingleRank(3);
        Assert.Equal(1, mesh.Descriptor.WorldSize);
        Assert.Equal(3, Assert.Single(mesh.Ranks).DeviceOrdinal);
        Assert.Equal(128, Unsafe.SizeOf<NcclUniqueId>());
    }

    [Fact]
    public void OpPhasesFollowTheNumbering()
    {
        Assert.Equal(OpPhase.M0, OpCatalog.PhaseOf(OpId.Sample));
        Assert.Equal(OpPhase.Fused, OpCatalog.PhaseOf(OpId.AddRmsNorm));
        Assert.Equal(OpPhase.M1, OpCatalog.PhaseOf(OpId.GatedDeltaRule));
        Assert.Equal(OpPhase.M1, OpCatalog.PhaseOf(OpId.GatedDeltaRuleGates));
        Assert.Equal(OpPhase.M1, OpCatalog.PhaseOf(OpId.SigmoidMul));
        Assert.Equal(OpPhase.M2, OpCatalog.PhaseOf(OpId.FusedMoe));
        Assert.Equal(OpPhase.M3, OpCatalog.PhaseOf(OpId.LoraExpand));
        Assert.Equal(OpPhase.M4, OpCatalog.PhaseOf(OpId.MlaAttention));
        Assert.Equal(OpPhase.M5, OpCatalog.PhaseOf(OpId.DequantizeGguf));
        Assert.Equal(OpPhase.None, OpCatalog.PhaseOf((OpId)999));
    }
}
