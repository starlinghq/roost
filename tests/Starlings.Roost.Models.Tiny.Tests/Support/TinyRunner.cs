using Starlings.Roost.Abi;
using Starlings.Roost.ModelCompiler.Tests;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// Runs roost-tiny on <see cref="CpuOps"/> the way a worker would: weights generated from the seed into native memory, a
/// pool of KV blocks handed out in a scrambled order, and one generated forward per step over a plan that holds several
/// sequences, each contributing a prefill chunk or a single decode token. Host addresses stand in for device addresses.
/// </summary>
public sealed unsafe class TinyRunner : IDisposable
{
    private readonly Native _native = new();
    private readonly TinyConfig _config = TinyConfig.Dense;
    private readonly DeviceSpan[] _weights;
    private readonly DeviceSpan[] _state;
    private readonly RoostTiny.ForwardLayout _layout;
    private readonly DeviceSpan _arena;
    private readonly DeviceSpan _planRegion;
    private readonly Queue<int> _freeBlocks;

    public TinyRunner(int blocks = 24, int maxTokens = 64, int maxSequences = 4)
    {
        Weights = new Dictionary<string, double[]>();
        _weights = new DeviceSpan[RoostTiny.WeightSlots.Length];
        for (int i = 0; i < _weights.Length; i++)
        {
            var slot = RoostTiny.WeightSlots[i];
            double[] values = TinyWeights.Generate(_config.Seed, slot.Name, slot.Shape, slot.Dtype);
            Weights[slot.Name] = values;
            _weights[i] = _native.Alloc(slot.Elements, slot.Dtype);
            TinyWeights.Write(values, slot.Dtype, _weights[i].Address);
        }

        _state = new DeviceSpan[RoostTiny.StateSlots.Length];
        for (int i = 0; i < _state.Length; i++)
        {
            var slot = RoostTiny.StateSlots[i];
            _state[i] = _native.Alloc(blocks * slot.ElementsPerBlock, slot.Dtype);
        }

        _layout = new RoostTiny.ForwardLayout(maxTokens, maxSequences, maxKvBlocks: blocks);
        _arena = _native.AllocBytes(_layout.ArenaBytes);
        _planRegion = _native.AllocBytes(_layout.PlanBytes);

        // A fixed scrambled order: consecutive sequences get interleaved, non-adjacent blocks.
        var order = Enumerable.Range(0, blocks).ToArray();
        for (int i = 0; i < order.Length; i++)
        {
            int j = (int)(((ulong)i * 7 + 3) % (ulong)order.Length);
            (order[i], order[j]) = (order[j], order[i]);
        }

        _freeBlocks = new Queue<int>(order.Distinct());
        if (_freeBlocks.Count != blocks)
        {
            throw new InvalidOperationException("The block permutation lost blocks.");
        }
    }

    /// <summary>The generated weights as doubles, by slot name.</summary>
    public Dictionary<string, double[]> Weights { get; }

    /// <summary>A sequence's block table and length.</summary>
    public sealed class Sequence
    {
        public List<int> Blocks { get; } = [];

        public int Length { get; internal set; }
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

        var plan = RoostTiny.PlanInputs.Bind(_planRegion, _layout, total, work.Length, work.Length, pages);
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

        var ctx = new RoostTiny.StepContext(_layout, _arena, _weights, _state, plan, total, work.Length, work.Length, pages);
        RoostTiny.Forward<CpuOps>(in ctx);
        var logits = RoostTiny.Output_logits(in ctx);
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
            int next = ArgMax(logits);
            generated.Add(next);
            logits = Step((seq, new[] { next }))[0];
        }

        return generated.ToArray();
    }

    public static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    public void Dispose() => _native.Dispose();
}
