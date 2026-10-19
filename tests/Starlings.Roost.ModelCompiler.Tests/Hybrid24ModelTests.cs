using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.ModelCompiler.Tests.Models;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>The 24-layer hybrid (6 attention, 18 recurrent layers) at the depth and layer layout of Qwen3.5-0.8B.</summary>
public class Hybrid24ModelTests
{
    private const nint ArenaBase = 0x4000_0000;
    private const nint PlanBase = 0x6000_0000;

    private static readonly Hybrid24Model.ForwardLayout Layout = new(maxTokens: 4096, maxSequences: 16, maxKvBlocks: 512);

    private static void Forward(DeviceSpan[] weights, DeviceSpan[] state)
    {
        var arena = new DeviceSpan { Address = ArenaBase, Length = (ulong)Layout.ArenaBytes, Dtype = DType.U8 };
        var plan = Hybrid24Model.PlanInputs.Bind(new DeviceSpan { Address = PlanBase, Length = (ulong)Layout.PlanBytes, Dtype = DType.U8 }, Layout, 7, 2, 2, 5);
        var ctx = new Hybrid24Model.StepContext(Layout, arena, weights, state, plan, 7, 2, 2, 5);
        Hybrid24Model.Forward<RecordingOps>(in ctx);
    }

    [Fact]
    public void LayerCountsSlotCountsAndCallsMatchTheLayout()
    {
        Assert.Equal(1 + (18 * 11) + (6 * 9) + 2, Hybrid24Model.WeightSlots.Length);
        // Per recurrent layer a state and a conv slot, per attention layer K and V.
        Assert.Equal((18 * 2) + (6 * 2), Hybrid24Model.StateSlots.Length);

        var g = Hybrid24Model.BuildGraph();
        Assert.Equal(18, g.Nodes.Count(n => n.Op == "GatedDeltaRule"));
        Assert.Equal(6, g.Nodes.Count(n => n.Op == "PagedAttention"));
        Assert.Equal(18, g.Nodes.Count(n => n.Op == "CausalConv1d"));
        Assert.Equal([3, 7, 11, 15, 19, 23], Hybrid24Model.StateSlots.Where(s => s.Role == "k").Select(s => s.Layer).ToArray());

        var weights = new DeviceSpan[Hybrid24Model.WeightSlots.Length];
        var state = new DeviceSpan[Hybrid24Model.StateSlots.Length];
        RecordingOps.Reset();
        Forward(weights, state);
        Assert.Equal(g.Nodes.Count, RecordingOps.Count);
        Assert.Equal(g.Nodes.Select(n => Enum.Parse<OpId>(n.Op)).ToArray(), RecordingOps.Ops.Take(RecordingOps.Count).ToArray());
    }

    [Fact]
    public void ForwardAllocatesNothingOnceWarm()
    {
        var weights = new DeviceSpan[Hybrid24Model.WeightSlots.Length];
        var state = new DeviceSpan[Hybrid24Model.StateSlots.Length];
        void Step()
        {
            RecordingOps.Reset();
            Forward(weights, state);
        }

        AllocationProbe.AssertSteady(Step);
    }
}
