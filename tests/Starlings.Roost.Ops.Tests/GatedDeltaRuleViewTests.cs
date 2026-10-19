using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

// GatedDeltaRule reading q, k and v as row-strided column slices of one fused buffer (contract version 2): the layout
// Qwen3.5 has after its causal conv1d over in_proj_qkv, split by [k_heads * k_dim, k_heads * k_dim, v_heads * v_dim]
// (transformers 5.20.1 Qwen3_5GatedDeltaNet.forward). Every case compares the views with contiguous copies of the same
// values; CpuOps does the same arithmetic on both, so the comparison is exact.
public sealed class GatedDeltaRuleViewTests
{
    // Unequal key and value widths (2 key heads x 3, 4 value heads x 2), so a split into equal thirds would be wrong.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StridedViewsOfAFusedBufferEqualContiguousCopies(bool l2)
    {
        const int kh = 2;
        const int vh = 4;
        const int kd = 3;
        const int vd = 2;
        const int width = (2 * kh * kd) + (vh * vd);
        int[] lens = [5, 3];
        int tokens = lens.Sum();
        var rng = new Random(91);
        double[] fused = Enumerable.Range(0, tokens * width).Select(_ => (rng.NextDouble() * 2) - 1).ToArray();
        double[] g = Enumerable.Range(0, tokens * vh).Select(_ => -rng.NextDouble()).ToArray();
        double[] beta = Enumerable.Range(0, tokens * vh).Select(_ => rng.NextDouble()).ToArray();
        float[] pool = Enumerable.Range(0, 4 * vh * kd * vd).Select(_ => (float)((rng.NextDouble() * 2) - 1)).ToArray();

        using var f = new Bag();
        var qkv = f.Add(Mem.F64(fused));
        var gm = f.Add(Mem.F64(g));
        var bm = f.Add(Mem.F64(beta));
        var slots = f.Add(Mem.I32(3, 1));
        var cu = f.Add(Mem.I32(0, 5, 8));

        var viewOut = f.Add(Mem.Zeros(DType.F64, tokens * vh * vd));
        var viewState = f.Add(Mem.F32(pool));
        var views = Args(View(qkv, 0), View(qkv, kh * kd), View(qkv, 2 * kh * kd), gm, bm, viewOut, viewState, slots, cu, 2, tokens, kh, vh, kd, vd, l2);
        views.QStride = width;
        views.KStride = width;
        views.VStride = width;
        CpuOps.GatedDeltaRule(views);

        var q = f.Add(Mem.F64(Columns(fused, tokens, width, 0, kh * kd)));
        var k = f.Add(Mem.F64(Columns(fused, tokens, width, kh * kd, kh * kd)));
        var v = f.Add(Mem.F64(Columns(fused, tokens, width, 2 * kh * kd, vh * vd)));
        var copyOut = f.Add(Mem.Zeros(DType.F64, tokens * vh * vd));
        var copyState = f.Add(Mem.F32(pool));
        CpuOps.GatedDeltaRule(Args(q.Span, k.Span, v.Span, gm, bm, copyOut, copyState, slots, cu, 2, tokens, kh, vh, kd, vd, l2));

        Assert.Equal(copyOut.ReadF64(), viewOut.ReadF64());
        Assert.Equal(copyState.ReadF32(), viewState.ReadF32());
        Assert.Contains(viewOut.ReadF64(), x => x != 0);
    }

    // The Qwen3.5-0.8B linear-attention shape: CausalConv1d over all 6144 channels of the fused projection, then
    // GatedDeltaRule on views of the conv output (stride 6144) against the same op on split copies.
    [Fact]
    public void Qwen35ConvThenViewsEqualsConvThenSplit()
    {
        const int heads = 16;
        const int dim = 128;
        const int keyDim = heads * dim;
        const int valueDim = heads * dim;
        const int channels = (2 * keyDim) + valueDim;
        const int tokens = 2;
        Assert.Equal(6144, channels);
        var rng = new Random(5);
        float[] Rand(int n, double s) => Enumerable.Range(0, n).Select(_ => (float)(((rng.NextDouble() * 2) - 1) * s)).ToArray();

        using var f = new Bag();
        var slots = f.Add(Mem.I32(0));
        var cu = f.Add(Mem.I32(0, tokens));
        var x = f.Add(Mem.F32(Rand(tokens * channels, 1)));
        var w = f.Add(Mem.F32(Rand(channels * 4, 0.5)));
        var conv = f.Add(Mem.Zeros(DType.F32, tokens * channels));
        var convState = f.Add(Mem.Zeros(DType.F32, channels * 3));
        CpuOps.CausalConv1d(new CausalConv1dArgs
        {
            Input = x.Span, Weight = w.Span, Output = conv.Span, ConvState = convState.Span, StateSlots = slots.Span, CuSeqlens = cu.Span,
            Batch = 1, Tokens = tokens, Channels = channels, Width = 4, Silu = true, DType = DType.F32,
        });
        var g = f.Add(Mem.F32(Rand(tokens * heads, 0.5).Select(v => -Math.Abs(v)).ToArray()));
        var beta = f.Add(Mem.F32(Rand(tokens * heads, 1).Select(Math.Abs).ToArray()));

        var viewOut = f.Add(Mem.Zeros(DType.F32, tokens * valueDim));
        var viewState = f.Add(Mem.Zeros(DType.F32, heads * dim * dim));
        var views = Args(View(conv, 0), View(conv, keyDim), View(conv, 2 * keyDim), g, beta, viewOut, viewState, slots, cu, 1, tokens, heads, heads, dim, dim, true);
        views.QStride = channels;
        views.KStride = channels;
        views.VStride = channels;
        views.DType = DType.F32;
        CpuOps.GatedDeltaRule(views);

        float[] y = conv.ReadF32();
        var q = f.Add(Mem.F32(Columns(y, tokens, channels, 0, keyDim)));
        var k = f.Add(Mem.F32(Columns(y, tokens, channels, keyDim, keyDim)));
        var v = f.Add(Mem.F32(Columns(y, tokens, channels, 2 * keyDim, valueDim)));
        var copyOut = f.Add(Mem.Zeros(DType.F32, tokens * valueDim));
        var copyState = f.Add(Mem.Zeros(DType.F32, heads * dim * dim));
        var copies = Args(q.Span, k.Span, v.Span, g, beta, copyOut, copyState, slots, cu, 1, tokens, heads, heads, dim, dim, true);
        copies.DType = DType.F32;
        CpuOps.GatedDeltaRule(copies);

        Assert.Equal(copyOut.ReadF32(), viewOut.ReadF32());
        Assert.Equal(copyState.ReadF32(), viewState.ReadF32());
        Assert.Contains(viewOut.ReadF32(), x => x != 0f);
    }

    [Fact]
    public void ZeroStrideMeansContiguous()
    {
        using var f = new Bag();
        var q = f.Add(Mem.F64(0.5, -1, 2, 0.25));
        var g = f.Add(Mem.F64(-0.5, -0.25));
        var b = f.Add(Mem.F64(0.75, 0.5));
        var slots = f.Add(Mem.I32(0));
        var cu = f.Add(Mem.I32(0, 2));
        var o0 = f.Add(Mem.Zeros(DType.F64, 4));
        var s0 = f.Add(Mem.Zeros(DType.F32, 4));
        var o1 = f.Add(Mem.Zeros(DType.F64, 4));
        var s1 = f.Add(Mem.Zeros(DType.F32, 4));
        CpuOps.GatedDeltaRule(Args(q.Span, q.Span, q.Span, g, b, o0, s0, slots, cu, 1, 2, 1, 1, 2, 2, false));
        var explicitStride = Args(q.Span, q.Span, q.Span, g, b, o1, s1, slots, cu, 1, 2, 1, 1, 2, 2, false);
        explicitStride.QStride = 2;
        explicitStride.KStride = 2;
        explicitStride.VStride = 2;
        CpuOps.GatedDeltaRule(explicitStride);
        Assert.Equal(o0.ReadF64(), o1.ReadF64());
        Assert.Equal(s0.ReadF32(), s1.ReadF32());
    }

    // Each view starts at element 32 of a 64-element buffer and runs to its end, long enough for every stride here
    // (contiguous extent 12, strided extent stride + row width), so only the stride check can reject these; ParamName pins
    // which check threw. Starting mid-buffer also keeps a negative stride's reads inside the buffer if that check is missing.
    [Theory]
    [InlineData(5, 0, 0, "QStride")]  // QStride below the row width (KHeads * KDim = 6)
    [InlineData(0, -6, 0, "KStride")] // negative KStride
    [InlineData(0, 0, 3, "VStride")]  // VStride below the row width (VHeads * VDim = 4)
    public void RejectsAStrideBelowTheRowWidth(int qs, int ks, int vs, string field)
    {
        using var f = new Bag();
        var buf = f.Add(Mem.Zeros(DType.F64, 64));
        var span = new DeviceSpan { Address = buf.Address + (32 * sizeof(double)), Length = 32, Dtype = DType.F64 };
        var a = StrideCase(f, span);
        a.QStride = qs;
        a.KStride = ks;
        a.VStride = vs;
        var e = Assert.Throws<ArgumentException>(() => CpuOps.GatedDeltaRule(a));
        Assert.Equal(field, e.ParamName);
    }

    // The same views with strides at the row width run, so the rejections above come from the stride values alone.
    [Fact]
    public void StridesAtTheRowWidthAreAccepted()
    {
        using var f = new Bag();
        var buf = f.Add(Mem.Zeros(DType.F64, 64));
        var span = new DeviceSpan { Address = buf.Address + (32 * sizeof(double)), Length = 32, Dtype = DType.F64 };
        var a = StrideCase(f, span);
        a.QStride = 6;
        a.KStride = 6;
        a.VStride = 4;
        CpuOps.GatedDeltaRule(a);
    }

    [Fact]
    public void RejectsASpanTooShortForItsStride()
    {
        using var f = new Bag();
        var buf = f.Add(Mem.Zeros(DType.F64, 64));
        // (Tokens - 1) * stride + row width = 20 + 6 = 26 elements needed; the span has 25.
        var span = new DeviceSpan { Address = buf.Address, Length = 25, Dtype = DType.F64 };
        var a = StrideCase(f, span);
        a.QStride = 20;
        a.KStride = 20;
        a.VStride = 20;
        var e = Assert.Throws<ArgumentException>(() => CpuOps.GatedDeltaRule(a));
        Assert.StartsWith("Q needs 26 elements", e.Message, StringComparison.Ordinal);
    }

    // Two tokens, one sequence, 2 key heads x 3 and 2 value heads x 2, with q, k and v all reading `span`.
    private static GatedDeltaRuleArgs StrideCase(Bag f, DeviceSpan span)
    {
        var g = f.Add(Mem.Zeros(DType.F64, 4));
        var o = f.Add(Mem.Zeros(DType.F64, 8));
        var st = f.Add(Mem.Zeros(DType.F32, 2 * 3 * 2));
        var slots = f.Add(Mem.I32(0));
        var cu = f.Add(Mem.I32(0, 2));
        return Args(span, span, span, g, g, o, st, slots, cu, 1, 2, 2, 2, 3, 2, false);
    }

    private static GatedDeltaRuleArgs Args(DeviceSpan q, DeviceSpan k, DeviceSpan v, Mem g, Mem beta, Mem o, Mem state, Mem slots, Mem cu, int batch, int tokens, int kh, int vh, int kd, int vd, bool l2) => new()
    {
        Q = q, K = k, V = v, G = g.Span, Beta = beta.Span, Output = o.Span, State = state.Span, StateSlots = slots.Span, CuSeqlens = cu.Span,
        Batch = batch, Tokens = tokens, KHeads = kh, VHeads = vh, KDim = kd, VDim = vd, Scale = 0.6f, UseInitialState = true, L2NormQk = l2, DType = DType.F64,
    };

    // A span that starts at column `column` of a row-major buffer and runs to the buffer's end.
    private static DeviceSpan View(Mem m, int column)
    {
        int size = m.Dtype == DType.F64 ? 8 : 4;
        return new DeviceSpan { Address = m.Address + (column * size), Length = (ulong)(m.Count - column), Dtype = m.Dtype };
    }

    private static T[] Columns<T>(T[] rows, int count, int width, int start, int length)
    {
        var r = new T[count * length];
        for (int t = 0; t < count; t++)
        {
            Array.Copy(rows, (t * width) + start, r, t * length, length);
        }

        return r;
    }
}
