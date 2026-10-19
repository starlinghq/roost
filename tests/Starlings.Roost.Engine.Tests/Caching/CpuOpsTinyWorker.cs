using System.Collections.Immutable;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Ops;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Engine.Tests.Caching;

/// <summary>
/// A two-layer, Qwen3-shaped attention model (hidden 16, two query heads over one KV head of width 8, vocabulary 257)
/// with seeded weights generated here, whose cache writes, paged attention, residual adds, LM head and argmax all run on
/// <see cref="CpuOps"/> in fp32 against real block pools. It stands in for <c>roost-tiny</c> (R-021 is not merged at this
/// base). The linear maps are plain loops over each token alone, so a token's result never depends on how the plan was
/// batched; any difference between a cached and an uncached run is then the cache's.
/// </summary>
internal sealed unsafe class CpuOpsTinyWorker : IPlanWorker, IDisposable
{
    private const int Dim = 16;
    private const int QHeads = 2;
    private const int KvHeads = 1;
    private const int HeadDim = 8;
    private const int Layers = 2;
    private const int Vocab = 257;

    private readonly int _blockSize;
    private readonly int _blocks;
    private readonly nint[] _k = new nint[Layers];
    private readonly nint[] _v = new nint[Layers];
    private readonly float[][] _wq = new float[Layers][];
    private readonly float[][] _wk = new float[Layers][];
    private readonly float[][] _wv = new float[Layers][];
    private readonly nint _wout;

    public CpuOpsTinyWorker(BlockManager blocks)
    {
        var g = blocks.Groups.Single();
        _blockSize = g.BlockSize;
        _blocks = blocks.GetPool(0).Capacity;
        for (var l = 0; l < Layers; l++)
        {
            _k[l] = Alloc(_blocks * _blockSize * KvHeads * HeadDim);
            _v[l] = Alloc(_blocks * _blockSize * KvHeads * HeadDim);
            _wq[l] = Weights((ulong)(l * 3) + 1, Dim * Dim, Dim);
            _wk[l] = Weights((ulong)(l * 3) + 2, Dim * HeadDim, Dim);
            _wv[l] = Weights((ulong)(l * 3) + 3, Dim * HeadDim, Dim);
        }

        _wout = Alloc(Vocab * Dim);
        var w = Weights(99, Vocab * Dim, 1);
        for (var i = 0; i < w.Length; i++)
        {
            ((float*)_wout)[i] = w[i];
        }
    }

    public long ComputedTokens { get; private set; }

    /// <summary>For each request, the logits (as raw bits) of every row sampled, in order.</summary>
    public Dictionary<ulong, List<uint[]>> Logits { get; } = [];

    public void Dispose()
    {
        foreach (var p in _k.Concat(_v).Append(_wout))
        {
            Marshal.FreeHGlobal(p);
        }
    }

    public StepOutcome Execute(PlannedStep plan)
    {
        foreach (var op in plan.CacheOps)
        {
            if (op.Kind != CacheOpKind.Copy)
            {
                throw new InvalidOperationException($"The attention-only model got a {op.Kind} op.");
            }

            for (var k = 0; k < (int)op.Count; k++)
            {
                int[] pairs = [(int)op.Src + k, (int)op.Dst + k];
                for (var l = 0; l < Layers; l++)
                {
                    CopyBlock(_k[l], pairs);
                    CopyBlock(_v[l], pairs);
                }
            }
        }

        var rows = plan.Sequences;
        var total = plan.TokenIds.Length;
        var h = new float[total * Dim];
        for (var t = 0; t < total; t++)
        {
            for (var d = 0; d < Dim; d++)
            {
                h[(t * Dim) + d] = Embed(plan.TokenIds[t], d);
            }
        }

        var slots = new int[total];
        var qIndptr = new int[rows.Length + 1];
        var kvIndptr = new int[rows.Length + 1];
        var last = new int[rows.Length];
        var indices = new List<int>();
        for (var r = 0; r < rows.Length; r++)
        {
            var s = rows[r];
            var gb = plan.GroupBlocks[(int)s.GroupBlocksStart];
            var table = plan.BlockIds.AsSpan((int)gb.BlockStart, (int)gb.BlockCount);
            for (var i = 0; i < (int)s.TokenCount; i++)
            {
                var pos = (int)s.PositionOffset + i;
                slots[(int)s.TokenStart + i] = (table[pos / _blockSize] * _blockSize) + (pos % _blockSize);
            }

            var length = (int)s.PositionOffset + (int)s.TokenCount;
            var pages = (length + _blockSize - 1) / _blockSize;
            qIndptr[r + 1] = (int)s.TokenStart + (int)s.TokenCount;
            kvIndptr[r + 1] = kvIndptr[r] + pages;
            last[r] = length - ((pages - 1) * _blockSize);
            for (var p = 0; p < pages; p++)
            {
                indices.Add(table[p]);
            }
        }

        for (var l = 0; l < Layers; l++)
        {
            var q = new float[total * Dim];
            var k = new float[total * HeadDim];
            var v = new float[total * HeadDim];
            for (var t = 0; t < total; t++)
            {
                Linear(_wq[l], h, t * Dim, q, t * Dim, Dim);
                Linear(_wk[l], h, t * Dim, k, t * HeadDim, HeadDim);
                Linear(_wv[l], h, t * Dim, v, t * HeadDim, HeadDim);
            }

            fixed (float* qp = q, kp = k, vp = v, hp = h)
            fixed (int* sp = slots, qi = qIndptr, ki = kvIndptr, lp = last, ip = indices.ToArray())
            {
                var output = new float[total * Dim];
                fixed (float* op = output)
                {
                    CpuOps.CacheWrite(new CacheWriteArgs
                    {
                        K = F32(kp, total * HeadDim), V = F32(vp, total * HeadDim), KCache = F32(_k[l], _blocks * _blockSize * HeadDim),
                        VCache = F32(_v[l], _blocks * _blockSize * HeadDim), SlotMapping = I32(sp, total), Tokens = total, KvHeads = KvHeads,
                        HeadDim = HeadDim, BlockSize = _blockSize, KvLayout = TensorLayout.PagedNhd, KScale = 1, VScale = 1, KvDType = DType.F32,
                    });
                    CpuOps.PagedAttention(new PagedAttentionArgs
                    {
                        Q = F32(qp, total * Dim), Output = F32(op, total * Dim), KCache = F32(_k[l], _blocks * _blockSize * HeadDim),
                        VCache = F32(_v[l], _blocks * _blockSize * HeadDim), QIndptr = I32(qi, rows.Length + 1), KvIndptr = I32(ki, rows.Length + 1),
                        KvIndices = I32(ip, indices.Count), KvLastPageLength = I32(lp, rows.Length), Mode = AttentionMode.Prefill, Batch = rows.Length,
                        Tokens = total, QHeads = QHeads, KvHeads = KvHeads, HeadDim = HeadDim, BlockSize = _blockSize, KvLayout = TensorLayout.PagedNhd,
                        Scale = 0, SoftCap = 0, WindowLeft = -1, Causal = true, KvDType = DType.F32, DType = DType.F32,
                    });
                    CpuOps.Residual(new ResidualArgs { A = F32(hp, total * Dim), B = F32(op, total * Dim), Output = F32(hp, total * Dim), Elements = total * Dim, DType = DType.F32 });
                }
            }
        }

        ComputedTokens += total;
        var sampled = new List<int>();
        for (var r = 0; r < rows.Length; r++)
        {
            if ((rows[r].Flags & SequenceOptions.Sample) != 0)
            {
                sampled.Add((int)rows[r].TokenStart + (int)rows[r].TokenCount - 1);
            }
        }

        var ids = new int[sampled.Count];
        if (sampled.Count > 0)
        {
            var logits = new float[sampled.Count * Vocab];
            var rowIdx = sampled.ToArray();
            fixed (float* hp = h, lg = logits)
            fixed (int* rp = rowIdx, idp = ids)
            {
                CpuOps.LmHead(new LmHeadArgs
                {
                    Hidden = F32(hp, total * Dim), Rows = I32(rp, rowIdx.Length), Weight = F32((float*)_wout, Vocab * Dim), Logits = F32(lg, logits.Length),
                    Selected = rowIdx.Length, Dim = Dim, Vocab = Vocab, VocabPadded = Vocab, DType = DType.F32,
                });
                CpuOps.Argmax(new ArgmaxArgs { Logits = F32(lg, logits.Length), Ids = I32(idp, ids.Length), Rows = rowIdx.Length, Vocab = Vocab });
            }

            var n = 0;
            for (var r = 0; r < rows.Length; r++)
            {
                if ((rows[r].Flags & SequenceOptions.Sample) != 0)
                {
                    var bits = new uint[Vocab];
                    for (var i = 0; i < Vocab; i++)
                    {
                        bits[i] = BitConverter.SingleToUInt32Bits(logits[(n * Vocab) + i]);
                    }

                    if (!Logits.TryGetValue(rows[r].SequenceHandle, out var list))
                    {
                        Logits[rows[r].SequenceHandle] = list = [];
                    }

                    list.Add(bits);
                    n++;
                }
            }
        }

        var results = ImmutableArray.CreateBuilder<SequenceResult>();
        var next = 0;
        foreach (var s in rows)
        {
            results.Add(new SequenceResult(s.SequenceHandle, (s.Flags & SequenceOptions.Sample) != 0 ? [ids[next++] + 1] : [], FinishReason.None));
        }

        return new StepOutcome(plan.StepId, results.ToImmutable());
    }

    private static float Embed(int token, int d)
    {
        var z = ((ulong)token * 0x9E3779B97F4A7C15UL) ^ ((ulong)d * 0xBF58476D1CE4E5B9UL);
        z = (z ^ (z >> 30)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return ((z >> 40) / (float)(1 << 24) * 2f) - 1f;
    }

    private static float[] Weights(ulong seed, int count, int fanIn)
    {
        var w = new float[count];
        for (var i = 0; i < count; i++)
        {
            w[i] = Embed((int)seed, i) / MathF.Sqrt(fanIn);
        }

        return w;
    }

    private static void Linear(float[] w, float[] x, int xo, float[] y, int yo, int outputs)
    {
        for (var o = 0; o < outputs; o++)
        {
            var acc = 0f;
            for (var d = 0; d < Dim; d++)
            {
                acc += w[(o * Dim) + d] * x[xo + d];
            }

            y[yo + o] = acc;
        }
    }

    private static nint Alloc(int floats)
    {
        var p = Marshal.AllocHGlobal(floats * sizeof(float));
        new Span<byte>((void*)p, floats * sizeof(float)).Clear();
        return p;
    }

    private static DeviceSpan F32(float* p, int n) => new() { Address = (nint)p, Length = (ulong)n, Dtype = DType.F32 };

    private static DeviceSpan F32(nint p, int n) => new() { Address = p, Length = (ulong)n, Dtype = DType.F32 };

    private static DeviceSpan I32(int* p, int n) => new() { Address = (nint)p, Length = (ulong)n, Dtype = DType.I32 };

    private void CopyBlock(nint pool, int[] pair)
    {
        var bytes = _blockSize * KvHeads * HeadDim * sizeof(float);
        Buffer.MemoryCopy((byte*)pool + ((long)pair[0] * bytes), (byte*)pool + ((long)pair[1] * bytes), bytes, bytes);
    }
}
