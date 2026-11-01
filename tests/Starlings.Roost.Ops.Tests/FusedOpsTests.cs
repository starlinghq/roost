using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>
/// The fused ops on CpuOps (R-071) are defined as their unfused parts, so each must give the same bytes as the unfused
/// calls in the order the reference graph makes them.
/// </summary>
public sealed class FusedOpsTests
{
    private static Mem Of(DType t, float[] v) => t == DType.BF16 ? Mem.Bf16(v) : Mem.F32(v);

    private static float[] Random(Random rng, int n, float scale = 1f, float offset = 0f) =>
        Enumerable.Range(0, n).Select(_ => offset + (((float)rng.NextDouble() - 0.5f) * 2f * scale)).ToArray();

    [Theory]
    [InlineData(DType.F32)]
    [InlineData(DType.BF16)]
    public void AddRmsNormIsResidualThenRmsNormBitForBit(DType t)
    {
        const int rows = 3, dim = 40;
        var rng = new Random(5);
        float[] x = Random(rng, rows * dim), r = Random(rng, rows * dim), w = Random(rng, dim, 0.1f, 1f);

        // Unfused, as the reference graph runs it: S = Residual(A = r, B = x); Y = RmsNorm(S).
        using var a = Of(t, r);
        using var b = Of(t, x);
        using var s = Mem.Zeros(t, rows * dim);
        using var y = Mem.Zeros(t, rows * dim);
        using var wm = Of(t, w);
        CpuOps.Residual(new ResidualArgs { A = a.Span, B = b.Span, Output = s.Span, Elements = rows * dim, DType = t });
        CpuOps.RmsNorm(new RmsNormArgs { Input = s.Span, Weight = wm.Span, Output = y.Span, Rows = rows, Dim = dim, Eps = 1e-6f, DType = t });

        // Fused: Input = x (becomes Y), Residual = r (becomes S).
        using var input = Of(t, x);
        using var residual = Of(t, r);
        CpuOps.AddRmsNorm(new AddRmsNormArgs { Input = input.Span, Residual = residual.Span, Weight = wm.Span, Rows = rows, Dim = dim, Eps = 1e-6f, DType = t });

        Assert.Equal(s.ReadBytes(), residual.ReadBytes());
        Assert.Equal(y.ReadBytes(), input.ReadBytes());
        Assert.True(CpuOps.SupportsAddRmsNorm);
    }

    [Theory]
    [InlineData(DType.F32, DType.F32)]
    [InlineData(DType.BF16, DType.BF16)]
    [InlineData(DType.BF16, DType.F32)]
    public void QkvQkNormRopeCacheWriteIsItsUnfusedPartsBitForBit(DType t, DType kvType)
    {
        const int tokens = 5, qHeads = 4, kvHeads = 2, hd = 8, blockSize = 4, blocks = 4;
        const int qw = qHeads * hd, kw = kvHeads * hd, row = qw + (2 * kw);
        var rng = new Random(9);
        float[] qkv = Random(rng, tokens * row), qNorm = Random(rng, hd, 0.1f, 1f), kNorm = Random(rng, hd, 0.1f, 1f);
        int[] positions = [7, 8, 9, 0, 1];
        long[] slots = [3, 12, -1, 5, 6];

        // Unfused: the three projections as separate values, then QkNorm, Rope and CacheWrite.
        float[] Columns(int start, int width) =>
            Enumerable.Range(0, tokens).SelectMany(tok => qkv.Skip((tok * row) + start).Take(width)).ToArray();
        using var q = Of(t, Columns(0, qw));
        using var k = Of(t, Columns(qw, kw));
        using var v = Of(t, Columns(qw + kw, kw));
        using var qn = Of(t, qNorm);
        using var kn = Of(t, kNorm);
        using var pos = Mem.I32(positions);
        using var map = Mem.I64(slots);
        using var kc = Mem.Zeros(kvType, blocks * blockSize * kw);
        using var vc = Mem.Zeros(kvType, blocks * blockSize * kw);
        CpuOps.QkNorm(new QkNormArgs { Q = q.Span, K = k.Span, QWeight = qn.Span, KWeight = kn.Span, Tokens = tokens, QHeads = qHeads, KvHeads = kvHeads, HeadDim = hd, Eps = 1e-6f, DType = t });
        CpuOps.Rope(new RopeArgs { Q = q.Span, K = k.Span, Positions = pos.Span, Tokens = tokens, QHeads = qHeads, KvHeads = kvHeads, HeadDim = hd, RotaryDim = hd, Kind = RopeKind.Default, Theta = 1e6f, DType = t });
        CpuOps.CacheWrite(new CacheWriteArgs
        {
            K = k.Span, V = v.Span, KCache = kc.Span, VCache = vc.Span, SlotMapping = map.Span, Tokens = tokens, KvHeads = kvHeads, HeadDim = hd,
            BlockSize = blockSize, KvLayout = TensorLayout.PagedNhd, KScale = 1f, VScale = 1f, KvDType = kvType,
        });

        // Fused, over the concatenated rows.
        using var fused = Of(t, qkv);
        using var qOut = Mem.Zeros(t, tokens * qw);
        using var kc2 = Mem.Zeros(kvType, blocks * blockSize * kw);
        using var vc2 = Mem.Zeros(kvType, blocks * blockSize * kw);
        byte[] before = fused.ReadBytes();
        CpuOps.QkvQkNormRopeCacheWrite(new QkvQkNormRopeCacheWriteArgs
        {
            Qkv = fused.Span, Q = qOut.Span, QWeight = qn.Span, KWeight = kn.Span, Positions = pos.Span, KCache = kc2.Span, VCache = vc2.Span,
            SlotMapping = map.Span, Tokens = tokens, QHeads = qHeads, KvHeads = kvHeads, HeadDim = hd, BlockSize = blockSize, Eps = 1e-6f, Theta = 1e6f,
            DType = t, KvDType = kvType,
        });

        Assert.Equal(q.ReadBytes(), qOut.ReadBytes());
        Assert.Equal(kc.ReadBytes(), kc2.ReadBytes());
        Assert.Equal(vc.ReadBytes(), vc2.ReadBytes());
        Assert.Equal(before, fused.ReadBytes());
        Assert.Contains(kc2.ReadBytes(), b => b != 0);
        Assert.True(CpuOps.SupportsQkvQkNormRopeCacheWrite);
    }

    [Fact]
    public void AnUndersizedQkvSpanIsRefusedBeforeAnyWrite()
    {
        using var qkv = Mem.Zeros(DType.F32, 11);
        using var q = Mem.F32(1, 2, 3, 4);
        using var w = Mem.F32(1, 1);
        using var pos = Mem.I32(0, 1);
        using var map = Mem.I64(0, 1);
        using var cache = Mem.Zeros(DType.F32, 8);
        var args = new QkvQkNormRopeCacheWriteArgs
        {
            Qkv = qkv.Span, Q = q.Span, QWeight = w.Span, KWeight = w.Span, Positions = pos.Span, KCache = cache.Span, VCache = cache.Span,
            SlotMapping = map.Span, Tokens = 2, QHeads = 1, KvHeads = 1, HeadDim = 2, BlockSize = 2, DType = DType.F32, KvDType = DType.F32,
        };
        Assert.Throws<ArgumentException>(() => CpuOps.QkvQkNormRopeCacheWrite(args));
        Assert.Equal([1f, 2f, 3f, 4f], q.ReadF32());
    }
}
