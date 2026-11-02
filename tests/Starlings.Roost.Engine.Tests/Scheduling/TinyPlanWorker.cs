using System.Collections.Immutable;
using Starlings.Roost.Abi;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.ModelCompiler.Tests;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Scheduling;

/// <summary>
/// A CPU worker for the planner's own plans: roost-tiny or roost-tiny-hybrid (R-021), weights generated from their seeds, one
/// generated forward per plan on <see cref="CpuOps"/> over the plan's own block ids and recurrent slots, greedy sampling (first
/// maximum). It does what a C2 worker does for async scheduling (R-032): it keeps the ids it sampled in the step it ran last,
/// writes them into the plan's <see cref="SequenceOptions.SampledInput"/> rows before the forward and refuses a plan that names
/// another input step; it sets the stop-id and length finish flags as <c>StepResultWriter</c> does; and it treats
/// <c>position_offset == 0</c> as a zero recurrent state (a preempted sequence recomputes from scratch). Host addresses stand
/// in for device addresses.
/// </summary>
internal sealed unsafe class TinyPlanWorker : IOrderedWorker, IDisposable
{
    private readonly Native _native = new();
    private readonly bool _hybrid;
    private readonly TinyConfig _config;
    private readonly DeviceSpan[] _weights;
    private readonly DeviceSpan[] _state;
    private readonly int[] _stateGroups;
    private readonly long[] _stateElements;
    private readonly int _attentionGroup;
    private readonly int _recurrentGroup = -1;
    private readonly int _maxTokens;
    private readonly int _maxSequences;
    private readonly int _blocks;
    private readonly DeviceSpan _region;
    private readonly DeviceSpan _arena;
    private readonly Dictionary<ulong, int> _sampled = [];
    private ulong _lastStep;

    public TinyPlanWorker(bool hybrid, int attentionBlocks, int recurrentSlots, int maxTokens, int maxSequences)
    {
        _hybrid = hybrid;
        _config = hybrid ? TinyConfig.Hybrid : TinyConfig.Dense;
        _maxTokens = maxTokens;
        _maxSequences = maxSequences;
        _blocks = attentionBlocks;
        var groups = StateGroups(hybrid);
        _attentionGroup = groups.Single(g => g.Kind == StateGroupKind.Full).Index;
        if (hybrid)
        {
            _recurrentGroup = groups.Single(g => g.Kind == StateGroupKind.Recurrent).Index;
        }

        var weightSlots = hybrid
            ? RoostTinyHybrid.WeightSlots.Select(w => (w.Name, w.Shape, w.Dtype, w.Elements)).ToArray()
            : RoostTiny.WeightSlots.Select(w => (w.Name, w.Shape, w.Dtype, w.Elements)).ToArray();
        _weights = new DeviceSpan[weightSlots.Length];
        for (var i = 0; i < _weights.Length; i++)
        {
            var slot = weightSlots[i];
            _weights[i] = _native.Alloc(slot.Elements, slot.Dtype);
            TinyWeights.Write(TinyWeights.Generate(_config.Seed, slot.Name, slot.Shape, slot.Dtype), slot.Dtype, _weights[i].Address);
        }

        var stateSlots = hybrid
            ? RoostTinyHybrid.StateSlots.Select(x => (x.Group, x.ElementsPerBlock, x.Dtype)).ToArray()
            : RoostTiny.StateSlots.Select(x => (x.Group, x.ElementsPerBlock, x.Dtype)).ToArray();
        _state = new DeviceSpan[stateSlots.Length];
        _stateGroups = new int[stateSlots.Length];
        _stateElements = new long[stateSlots.Length];
        for (var i = 0; i < _state.Length; i++)
        {
            var slot = stateSlots[i];
            _stateGroups[i] = slot.Group;
            _stateElements[i] = slot.ElementsPerBlock;
            _state[i] = _native.Alloc((slot.Group == _recurrentGroup ? recurrentSlots : attentionBlocks) * slot.ElementsPerBlock, slot.Dtype);
        }

        var (planBytes, arenaBytes) = hybrid
            ? (new RoostTinyHybrid.ForwardLayout(maxTokens, maxSequences, attentionBlocks).PlanBytes, new RoostTinyHybrid.ForwardLayout(maxTokens, maxSequences, attentionBlocks).ArenaBytes)
            : (new RoostTiny.ForwardLayout(maxTokens, maxSequences, attentionBlocks).PlanBytes, new RoostTiny.ForwardLayout(maxTokens, maxSequences, attentionBlocks).ArenaBytes);
        _region = _native.AllocBytes(planBytes);
        _arena = _native.AllocBytes(arenaBytes);
    }

    /// <summary>The model's state groups, as the planner's block manager needs them.</summary>
    public static ImmutableArray<StateGroupDescriptor> StateGroups(bool hybrid) => hybrid ? RoostTinyHybrid.CreateStateGroups() : RoostTiny.CreateStateGroups();

    public int Vocab => _config.Vocab;

    /// <summary>Rows whose input was patched from the previous step.</summary>
    public int Patched { get; private set; }

    public StepOutcome Execute(PlannedStep plan)
    {
        Assert.Empty(plan.CacheOps); // no prefix cache and no forks in these runs
        var tokens = plan.TokenIds.ToArray();
        if (plan.InputStepId != 0 && plan.InputStepId != _lastStep)
        {
            throw new InvalidOperationException($"Plan {plan.StepId} takes its inputs from step {plan.InputStepId}; the worker ran {_lastStep} last.");
        }

        var rows = plan.Sequences;
        foreach (var s in rows)
        {
            if ((s.Flags & SequenceOptions.SampledInput) != 0)
            {
                tokens[(int)(s.TokenStart + s.TokenCount - 1)] = _sampled[s.SequenceHandle];
                Patched++;
            }

            if (s.PositionOffset == 0 && _recurrentGroup >= 0)
            {
                ClearSlot(s.RecurrentSlot);
            }
        }

        var bs = _config.BlockSize;
        var total = tokens.Length;
        var pages = 0;
        for (var r = 0; r < rows.Length; r++)
        {
            pages += (int)((rows[r].PositionOffset + rows[r].TokenCount + bs - 1) / bs);
        }

        var logits = _hybrid ? ForwardHybrid(plan, tokens, total, pages) : ForwardDense(plan, tokens, total, pages);
        _sampled.Clear();
        var results = ImmutableArray.CreateBuilder<SequenceResult>(rows.Length);
        for (var r = 0; r < rows.Length; r++)
        {
            var s = rows[r];
            if ((s.Flags & SequenceOptions.Sample) == 0)
            {
                results.Add(new SequenceResult(s.SequenceHandle, [], FinishReason.None));
                continue;
            }

            var row = new ReadOnlySpan<float>((float*)logits.Address + ((long)r * _config.Vocab), _config.Vocab);
            var best = 0;
            for (var v = 1; v < row.Length; v++)
            {
                if (row[v] > row[best])
                {
                    best = v;
                }
            }

            _sampled[s.SequenceHandle] = best;
            var stop = plan.StopTokenIds.AsSpan((int)s.StopStart, (int)s.StopCount).Contains(best);
            results.Add(new SequenceResult(s.SequenceHandle, [best], stop ? FinishReason.Eos : s.MaxNewTokensLeft == 1 ? FinishReason.Length : FinishReason.None));
        }

        _lastStep = plan.StepId;
        return new StepOutcome(plan.StepId, results.MoveToImmutable());
    }

    public void Dispose() => _native.Dispose();

    private DeviceSpan ForwardDense(PlannedStep plan, int[] tokens, int total, int pages)
    {
        var layout = new RoostTiny.ForwardLayout(_maxTokens, _maxSequences, maxKvBlocks: _blocks);
        var n = plan.Sequences.Length;
        var p = RoostTiny.PlanInputs.Bind(_region, layout, total, n, n, pages);
        Fill(plan, tokens, (int*)p.Ids.Address, (int*)p.Positions.Address, (long*)p.SlotMapping.Address, (int*)p.QIndptr.Address,
            (int*)p.KvIndptr.Address, (int*)p.KvIndices.Address, (int*)p.KvLastPageLength.Address, (int*)p.Rows.Address, null, null);
        var ctx = new RoostTiny.StepContext(layout, _arena, _weights, _state, p, total, n, n, pages);
        RoostTiny.Forward<CpuOps>(in ctx);
        return RoostTiny.Output_logits(in ctx);
    }

    private DeviceSpan ForwardHybrid(PlannedStep plan, int[] tokens, int total, int pages)
    {
        var layout = new RoostTinyHybrid.ForwardLayout(_maxTokens, _maxSequences, maxKvBlocks: _blocks);
        var n = plan.Sequences.Length;
        var p = RoostTinyHybrid.PlanInputs.Bind(_region, layout, total, n, n, pages);
        Fill(plan, tokens, (int*)p.Ids.Address, (int*)p.Positions.Address, (long*)p.SlotMapping.Address, (int*)p.QIndptr.Address,
            (int*)p.KvIndptr.Address, (int*)p.KvIndices.Address, (int*)p.KvLastPageLength.Address, (int*)p.Rows.Address,
            (int*)p.StateSlots.Address, (int*)p.CuSeqlens.Address);
        var ctx = new RoostTinyHybrid.StepContext(layout, _arena, _weights, _state, p, total, n, n, pages);
        RoostTinyHybrid.Forward<CpuOps>(in ctx);
        return RoostTinyHybrid.Output_logits(in ctx);
    }

    // The plan inputs of the generated forward, from the C2 tables: every row is selected (its last token's logits).
    private void Fill(PlannedStep plan, int[] tokens, int* ids, int* positions, long* slots, int* qIndptr, int* kvIndptr, int* kvIndices, int* last, int* rows, int* stateSlots, int* cuSeqlens)
    {
        var bs = _config.BlockSize;
        var page = 0;
        var n = plan.Sequences.Length;
        for (var b = 0; b < n; b++)
        {
            var s = plan.Sequences[b];
            var table = plan.BlocksOf(b, _attentionGroup);
            var start = (int)s.TokenStart;
            qIndptr[b] = start;
            kvIndptr[b] = page;
            if (cuSeqlens is not null)
            {
                cuSeqlens[b] = start;
                stateSlots[b] = s.RecurrentSlot;
            }

            for (var j = 0; j < (int)s.TokenCount; j++)
            {
                var pos = (int)s.PositionOffset + j;
                Assert.Equal(pos, plan.Positions[start + j]);
                ids[start + j] = tokens[start + j];
                positions[start + j] = pos;
                slots[start + j] = ((long)table[pos / bs] * bs) + (pos % bs);
            }

            var after = (int)(s.PositionOffset + s.TokenCount);
            var count = (after + bs - 1) / bs;
            for (var i = 0; i < count; i++)
            {
                kvIndices[page++] = table[i];
            }

            last[b] = ((after - 1) % bs) + 1;
            rows[b] = start + (int)s.TokenCount - 1;
        }

        qIndptr[n] = plan.TokenIds.Length;
        kvIndptr[n] = page;
        if (cuSeqlens is not null)
        {
            cuSeqlens[n] = plan.TokenIds.Length;
        }
    }

    private void ClearSlot(int slot)
    {
        for (var i = 0; i < _state.Length; i++)
        {
            if (_stateGroups[i] == _recurrentGroup)
            {
                var size = Native.Size(_state[i].Dtype);
                new Span<byte>((byte*)_state[i].Address + (slot * _stateElements[i] * size), (int)(_stateElements[i] * size)).Clear();
            }
        }
    }
}
