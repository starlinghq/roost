using Starlings.Roost.Graph;
using Starlings.Roost.Ops;
using Starlings.Roost.ModelCompiler.Tests.Models;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>The fused AddRmsNorm: the generated forward calls it when the backend has it and falls back to Residual + RmsNorm when not (S4 finding 1).</summary>
public sealed unsafe class FusedModelTests : IDisposable
{
    private const int Tokens = 3;
    private static readonly int[] Ids = [1, 60, 9];
    private static readonly int[] Rows = [2];

    private readonly Native _native = new();
    private readonly DeviceSpan[] _weights;
    private readonly float[][] _weightData;
    private readonly FusedModel.ForwardLayout _layout = new(8, 1, 0);
    private readonly DeviceSpan _arena;
    private readonly FusedModel.PlanInputs _plan;

    public FusedModelTests()
    {
        var rng = new Random(11);
        _weights = new DeviceSpan[FusedModel.WeightSlots.Length];
        _weightData = new float[_weights.Length][];
        for (int i = 0; i < _weights.Length; i++)
        {
            var slot = FusedModel.WeightSlots[i];
            _weights[i] = _native.Alloc(slot.Elements, slot.Dtype);
            _weightData[i] = new float[slot.Elements];
            for (int j = 0; j < slot.Elements; j++)
            {
                _weightData[i][j] = slot.Name.StartsWith("norm", StringComparison.Ordinal) ? 1f + (((float)rng.NextDouble() - 0.5f) * 0.2f) : ((float)rng.NextDouble() - 0.5f) * 0.6f;
                ((float*)_weights[i].Address)[j] = _weightData[i][j];
            }
        }

        _arena = _native.AllocBytes(_layout.ArenaBytes);
        _plan = FusedModel.PlanInputs.Bind(_native.AllocBytes(_layout.PlanBytes), _layout, Tokens, Rows.Length, 1, 0);
        for (int i = 0; i < Ids.Length; i++)
        {
            ((int*)_plan.Ids.Address)[i] = Ids[i];
        }

        ((int*)_plan.Rows.Address)[0] = Rows[0];
    }

    public void Dispose() => _native.Dispose();

    private float[] Run<TOps>()
        where TOps : struct, IRoostOps
    {
        var ctx = new FusedModel.StepContext(_layout, _arena, _weights, default, _plan, Tokens, Rows.Length, 1, 0);
        FusedModel.Forward<TOps>(in ctx);
        var o = FusedModel.Output_logits(in ctx);
        var r = new float[o.Length];
        new ReadOnlySpan<float>((void*)o.Address, r.Length).CopyTo(r);
        return r;
    }

    [Fact]
    public void FusedAndFallbackLoweringsGiveTheSameLogits()
    {
        float[] fallback = Run<StubCpuOps>();
        float[] fused = Run<StubFusedOps>();
        Assert.Equal(64, fallback.Length);
        Assert.Contains(fallback, x => Math.Abs(x) > 1e-3f);
        for (int i = 0; i < fused.Length; i++)
        {
            Assert.Equal(fused[i], fallback[i], 1e-5f);
        }
    }

    [Fact]
    public void BothLoweringsAllocateNothing()
    {
        void Fallback()
        {
            var ctx = new FusedModel.StepContext(_layout, _arena, _weights, default, _plan, Tokens, Rows.Length, 1, 0);
            FusedModel.Forward<StubCpuOps>(in ctx);
        }

        void Fused()
        {
            var ctx = new FusedModel.StepContext(_layout, _arena, _weights, default, _plan, Tokens, Rows.Length, 1, 0);
            FusedModel.Forward<StubFusedOps>(in ctx);
        }

        AllocationProbe.AssertSteady(Fallback);
        AllocationProbe.AssertSteady(Fused);
    }

    [Fact]
    public void AFusedNodeKeepsBothOverwrittenValuesAsOutputs()
    {
        var g = FusedModel.BuildGraph();
        var node = g.Nodes.Single(n => n.Op == "AddRmsNorm");
        Assert.Equal("Input,Residual,Weight", node.Attributes["in"]);
        Assert.Equal("Input,Residual", node.Attributes["out"]);
        Assert.Equal(["normed", "h1"], node.Outputs.Select(o => g.Values[o].Name).ToArray());
        Assert.Equal(["mlp", "h0", "norm1"], node.Inputs.Select(o => g.Values[o].Name).ToArray());
    }

    [Fact]
    public void TheGraphRunOnTheFusedBackendEqualsTheGeneratedForward()
    {
        float[] got = Run<StubFusedOps>();
        var g = FusedModel.BuildGraph();
        var inputs = new Dictionary<int, double[]>();
        for (int i = 0; i < _weightData.Length; i++)
        {
            inputs[i] = _weightData[i].Select(x => (double)x).ToArray();
        }

        inputs[g.Values.Single(v => v.Name == "plan.Ids").Id] = Ids.Select(x => (double)x).ToArray();
        inputs[g.Values.Single(v => v.Name == "plan.Rows").Id] = Rows.Select(x => (double)x).ToArray();
        var ev = new GraphEvaluator<StubFusedOps>(new Dictionary<string, long> { ["tokens"] = Tokens, ["selected"] = 1, ["seqs"] = 1, ["kv_blocks"] = 0 });
        Assert.Equal(got.Select(x => (double)x).ToArray(), ev.Evaluate(g, inputs)[g.Values.Single(v => v.Name == "logits").Id]);
    }

    [Fact]
    public void TheGraphEvaluatorRefusesAFusedOpTheBackendLacks()
    {
        var g = FusedModel.BuildGraph();
        var ev = new GraphEvaluator<StubCpuOps>(new Dictionary<string, long> { ["tokens"] = Tokens, ["selected"] = 1, ["seqs"] = 1, ["kv_blocks"] = 0 });
        var inputs = new Dictionary<int, double[]>();
        for (int i = 0; i < _weightData.Length; i++)
        {
            inputs[i] = _weightData[i].Select(x => (double)x).ToArray();
        }

        inputs[g.Values.Single(v => v.Name == "plan.Ids").Id] = Ids.Select(x => (double)x).ToArray();
        inputs[g.Values.Single(v => v.Name == "plan.Rows").Id] = Rows.Select(x => (double)x).ToArray();
        Assert.Throws<NotSupportedException>(() => ev.Evaluate(g, inputs));
    }
}
