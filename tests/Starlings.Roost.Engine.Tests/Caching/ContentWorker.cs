using System.Collections.Immutable;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.State;

namespace Starlings.Roost.Engine.Tests.Caching;

/// <summary>
/// A worker that keeps real contents. Each pool slot holds a number, cache ops move those numbers, and a forward
/// computes a token from everything its row can reach: every earlier position through the row's block tables and the
/// recurrent state. A wrong block id, a stale block, an op in the wrong order or a bad snapshot changes a sampled token
/// (or reads a slot nothing wrote, which throws), so cached and uncached runs match only if the cache is exact.
/// </summary>
/// <remarks>
/// The recurrent state is folded token by token, so it does not depend on how a prompt was chunked. The attention
/// value is a function of the token, the position and the recurrent state at that position, like a hybrid layer stack.
/// This is a stand-in for <c>roost-tiny</c> and <c>roost-tiny-hybrid</c> on CpuOps (R-021 is not merged at this base).
/// </remarks>
internal sealed class ContentWorker : IPlanWorker
{
    private readonly ImmutableArray<StateGroupDescriptor> _groups;
    private readonly ulong[][][] _attention; // group, block, slot in block
    private readonly ulong[][] _live; // recurrent group -> slot
    private readonly ulong[][] _snapshots;
    private readonly int _recurrent = -1;

    public ContentWorker(BlockManager blocks)
    {
        _groups = blocks.Groups;
        _attention = new ulong[_groups.Length][][];
        _live = new ulong[_groups.Length][];
        _snapshots = new ulong[_groups.Length][];
        foreach (var g in _groups)
        {
            if (g.Kind == StateGroupKind.Recurrent)
            {
                _recurrent = g.Index;
                _live[g.Index] = new ulong[blocks.GetPool(g.Index).Capacity];
                _snapshots[g.Index] = new ulong[blocks.GetSnapshotPool(g.Index).Capacity];
            }
            else
            {
                _attention[g.Index] = [.. Enumerable.Range(0, blocks.GetPool(g.Index).Capacity).Select(_ => new ulong[g.BlockSize])];
            }
        }
    }

    /// <summary>Tokens the worker computed (rows' token counts), for comparing work done.</summary>
    public long ComputedTokens { get; private set; }

    public StepOutcome Execute(PlannedStep plan)
    {
        foreach (var op in plan.CacheOps)
        {
            Apply(op);
        }

        var rows = ImmutableArray.CreateBuilder<SequenceResult>();
        foreach (var s in plan.Sequences)
        {
            ulong last = 0;
            var start = (int)s.PositionOffset;
            var state = _recurrent >= 0 && start > 0 ? _live[_recurrent][s.RecurrentSlot] : 0UL;
            for (var i = 0; i < (int)s.TokenCount; i++)
            {
                var pos = start + i;
                var token = (ulong)plan.TokenIds[(int)s.TokenStart + i];
                state = _recurrent >= 0 ? Mix(state, token, (ulong)pos, 1) : 0;
                ulong acc = Mix(token, (ulong)pos, state, 2);
                foreach (var g in _groups)
                {
                    if (g.Kind == StateGroupKind.Recurrent)
                    {
                        continue;
                    }

                    var table = Table(plan, s, g.Index);
                    for (var q = 0; q < pos; q++)
                    {
                        acc = Mix(acc, Read(table, g, q), (ulong)g.Index, 3);
                    }
                }

                var kv = Mix(token, (ulong)pos, state, 4);
                foreach (var g in _groups)
                {
                    if (g.Kind != StateGroupKind.Recurrent)
                    {
                        var table = Table(plan, s, g.Index);
                        _attention[g.Index][table[pos / g.BlockSize]][pos % g.BlockSize] = Mix(kv, (ulong)g.Index, 5, 6) | 1;
                    }
                }

                last = Mix(acc, kv, state, 7);
                ComputedTokens++;
            }

            if (_recurrent >= 0)
            {
                _live[_recurrent][s.RecurrentSlot] = state;
            }

            rows.Add(new SequenceResult(
                s.SequenceHandle,
                (s.Flags & SequenceOptions.Sample) != 0 ? [(int)(last % 4999UL) + 1] : [],
                FinishReason.None));
        }

        return new StepOutcome(plan.StepId, rows.ToImmutable());
    }

    private static ulong Mix(ulong a, ulong b, ulong c, ulong d)
    {
        var z = a * 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 29)) + (b * 0xBF58476D1CE4E5B9UL);
        z = (z ^ (z >> 31)) + (c * 0x94D049BB133111EBUL);
        z = (z ^ (z >> 27)) + (d * 0xD6E8FEB86659FD93UL) + 0x2545F4914F6CDD1DUL;
        return z ^ (z >> 32);
    }

    private static ReadOnlySpan<int> Table(PlannedStep plan, PlanSequence s, int group)
    {
        var gb = plan.GroupBlocks[(int)s.GroupBlocksStart + group];
        return plan.BlockIds.AsSpan((int)gb.BlockStart, (int)gb.BlockCount);
    }

    private ulong Read(ReadOnlySpan<int> table, StateGroupDescriptor g, int position)
    {
        var block = table[position / g.BlockSize];
        if (block < 0)
        {
            throw new InvalidOperationException($"Position {position} of group {g.Index} reads a released block.");
        }

        var value = _attention[g.Index][block][position % g.BlockSize];
        if (value == 0)
        {
            throw new InvalidOperationException($"Position {position} of group {g.Index} reads block {block} that nothing wrote.");
        }

        return value;
    }

    private void Apply(PlanCacheOp op)
    {
        var g = (int)op.Group;
        for (var k = 0; k < (int)op.Count; k++)
        {
            var src = (int)op.Src + k;
            var dst = (int)op.Dst + k;
            switch (op.Kind)
            {
                case CacheOpKind.Copy:
                    Array.Copy(_attention[g][src], _attention[g][dst], _attention[g][src].Length);
                    break;
                case CacheOpKind.Fork:
                    _live[g][dst] = _live[g][src];
                    break;
                case CacheOpKind.SnapshotWrite:
                    _snapshots[g][dst] = _live[g][src];
                    break;
                case CacheOpKind.SnapshotRestore:
                    _live[g][dst] = _snapshots[g][src];
                    break;
                default:
                    throw new InvalidOperationException($"Unexpected op {op.Kind}.");
            }
        }
    }
}
