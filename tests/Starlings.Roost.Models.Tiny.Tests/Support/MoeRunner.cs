using Starlings.Roost.Abi;
using Starlings.Roost.ModelCompiler.Tests;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// Runs roost-tiny-moe on <see cref="CpuOps"/> the way <see cref="TinyRunner"/> runs roost-tiny: seeded weights in native memory, KV
/// blocks handed out in a scrambled order, one generated forward per step over several sequences. It adds a fork: the new sequence
/// shares the parent's full blocks and gets a copy (<see cref="CpuOps.CacheCopy"/>, in every layer's K and V pool) of its partial
/// last block, as a prefix cache would.
/// </summary>
public sealed unsafe class MoeRunner : IDisposable
{
    private readonly Native _native = new();
    private readonly TinyConfig _config = TinyConfig.Moe;
    private readonly DeviceSpan[] _weights;
    private readonly DeviceSpan[] _state;
    private readonly RoostTinyMoe.ForwardLayout _layout;
    private readonly DeviceSpan _arena;
    private readonly DeviceSpan _planRegion;
    private readonly DeviceSpan _pairs;
    private readonly Queue<int> _freeBlocks;

    public MoeRunner(int blocks = 32, int maxTokens = 64, int maxSequences = 4)
    {
        _weights = new DeviceSpan[RoostTinyMoe.WeightSlots.Length];
        for (int i = 0; i < _weights.Length; i++)
        {
            var slot = RoostTinyMoe.WeightSlots[i];
            double[] values = TinyWeights.Generate(_config.Seed, slot.Name, slot.Shape, slot.Dtype);
            _weights[i] = _native.Alloc(slot.Elements, slot.Dtype);
            TinyWeights.Write(values, slot.Dtype, _weights[i].Address);
        }

        _state = new DeviceSpan[RoostTinyMoe.StateSlots.Length];
        for (int i = 0; i < _state.Length; i++)
        {
            var slot = RoostTinyMoe.StateSlots[i];
            _state[i] = _native.Alloc(blocks * slot.ElementsPerBlock, slot.Dtype);
        }

        _layout = new RoostTinyMoe.ForwardLayout(maxTokens, maxSequences, maxKvBlocks: blocks);
        _arena = _native.AllocBytes(_layout.ArenaBytes);
        _planRegion = _native.AllocBytes(_layout.PlanBytes);
        _pairs = _native.Alloc(2, DType.I32);

        var order = Enumerable.Range(0, blocks).ToArray();
        for (int i = 0; i < order.Length; i++)
        {
            int j = (int)(((ulong)i * 7 + 3) % (ulong)order.Length);
            (order[i], order[j]) = (order[j], order[i]);
        }

        _freeBlocks = new Queue<int>(order);
    }

    /// <summary>A sequence's block table and length.</summary>
    public sealed class Sequence
    {
        public List<int> Blocks { get; } = [];

        public int Length { get; internal set; }
    }

    /// <summary>The span of a weight slot, so a test can change it in place.</summary>
    public DeviceSpan Weight(string name) => _weights[Array.FindIndex(RoostTinyMoe.WeightSlots, s => s.Name == name)];

    /// <summary>A new sequence with the same cached prefix: full blocks shared, the partial last block copied into a block of its own.</summary>
    public Sequence Fork(Sequence parent)
    {
        var child = new Sequence { Length = parent.Length };
        int bs = _config.BlockSize;
        int full = parent.Length / bs;
        child.Blocks.AddRange(parent.Blocks.Take(full));
        if (parent.Length % bs != 0)
        {
            int src = parent.Blocks[full];
            int dst = _freeBlocks.Dequeue();
            ((int*)_pairs.Address)[0] = src;
            ((int*)_pairs.Address)[1] = dst;
            for (int i = 0; i < _state.Length; i++)
            {
                var slot = RoostTinyMoe.StateSlots[i];
                CpuOps.CacheCopy(new CacheCopyArgs
                {
                    Source = _state[i], Destination = _state[i], Pairs = _pairs, BlockBytes = slot.ElementsPerBlock * Native.Size(slot.Dtype), Count = 1,
                });
            }

            child.Blocks.Add(dst);
        }

        return child;
    }

    /// <summary>One forward over the tokens each sequence contributes; returns the logits of each sequence's last token.</summary>
    public float[][] Step(params (Sequence Seq, int[] Tokens)[] work)
    {
        int bs = _config.BlockSize;
        int total = work.Sum(w => w.Tokens.Length);
        int pages = 0;
        foreach (var (seq, tokens) in work)
        {
            int needed = (seq.Length + tokens.Length + bs - 1) / bs;
            while (seq.Blocks.Count < needed)
            {
                seq.Blocks.Add(_freeBlocks.Dequeue());
            }

            pages += needed;
        }

        var plan = RoostTinyMoe.PlanInputs.Bind(_planRegion, _layout, total, work.Length, work.Length, pages);
        int* ids = (int*)plan.Ids.Address;
        int* positions = (int*)plan.Positions.Address;
        long* slots = (long*)plan.SlotMapping.Address;
        int* qIndptr = (int*)plan.QIndptr.Address;
        int* kvIndptr = (int*)plan.KvIndptr.Address;
        int* kvIndices = (int*)plan.KvIndices.Address;
        int* last = (int*)plan.KvLastPageLength.Address;
        int* rows = (int*)plan.Rows.Address;
        int t = 0;
        int page = 0;
        for (int b = 0; b < work.Length; b++)
        {
            var (seq, tokens) = work[b];
            qIndptr[b] = t;
            kvIndptr[b] = page;
            for (int j = 0; j < tokens.Length; j++, t++)
            {
                int p = seq.Length + j;
                ids[t] = tokens[j];
                positions[t] = p;
                slots[t] = ((long)seq.Blocks[p / bs] * bs) + (p % bs);
            }

            int after = seq.Length + tokens.Length;
            int count = (after + bs - 1) / bs;
            for (int i = 0; i < count; i++)
            {
                kvIndices[page++] = seq.Blocks[i];
            }

            last[b] = ((after - 1) % bs) + 1;
            rows[b] = t - 1;
            seq.Length = after;
        }

        qIndptr[work.Length] = t;
        kvIndptr[work.Length] = page;

        var ctx = new RoostTinyMoe.StepContext(_layout, _arena, _weights, _state, plan, total, work.Length, work.Length, pages);
        RoostTinyMoe.Forward<CpuOps>(in ctx);
        var logits = RoostTinyMoe.Output_logits(in ctx);
        int vocab = _config.Vocab;
        var result = new float[work.Length][];
        for (int b = 0; b < work.Length; b++)
        {
            result[b] = new ReadOnlySpan<float>((float*)logits.Address + (b * vocab), vocab).ToArray();
        }

        return result;
    }

    /// <summary>Greedy decoding: one prefill step, then one token per step; first maximum wins.</summary>
    public int[] Generate(int[] prompt, int count)
    {
        var seq = new Sequence();
        var generated = new List<int>();
        float[] logits = Step((seq, prompt))[0];
        for (int i = 0; i < count; i++)
        {
            int next = TinyRunner.ArgMax(logits);
            generated.Add(next);
            logits = Step((seq, new[] { next }))[0];
        }

        return generated.ToArray();
    }

    public void Dispose() => _native.Dispose();
}
