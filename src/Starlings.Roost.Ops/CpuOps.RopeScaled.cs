using Starlings.Roost.Abi;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

public readonly partial struct CpuOps
{
    /// <inheritdoc/>
    /// <remarks>
    /// <para>Rotary embedding from a precomputed cache (R-098). Rotary dimensions are R = RotaryDim (0 means HeadDim, as for
    /// <see cref="Rope"/>); CosSinCache holds MaxPosition = <c>CosSinCache.Length / R</c> rows, row p being R / 2 cosines then
    /// R / 2 sines with any attention factor (YaRN's mscale, longrope's scaling) already multiplied in, as
    /// <see cref="RopeScaling"/> builds them. For token t at p = Positions[t], pair i joins element i with i + R / 2 (the NeoX
    /// halves, transformers' <c>rotate_half</c>; the arguments have no interleaved form) and becomes
    /// <c>(x1 c - x2 s, x2 c + x1 s)</c> with <c>c = cache[p, i]</c>, <c>s = cache[p, R/2 + i]</c>. Elements at and beyond R are
    /// unchanged. Q and K are rotated in place in DType; the cache is read as its span's Dtype (DType when it has none).</para>
    /// <para>The scaling itself is entirely in the cache: ScalingKind (1 YaRN, 2 longrope, 3 dynamic NTK) and Factor say how it was
    /// built and are checked (a known kind, a positive finite factor) but not otherwise read. For dynamic NTK and longrope, whose
    /// frequencies depend on the sequence length, the caller passes the cache built for the length being served.</para>
    /// <para>Every position is checked to lie in [0, MaxPosition) before anything is written.</para>
    /// </remarks>
    public static void RopeScaled(in RopeScaledArgs a)
    {
        int hd = a.HeadDim;
        int rot = a.RotaryDim == 0 ? hd : a.RotaryDim;
        if (hd < 1 || rot < 2 || rot > hd || (rot & 1) != 0)
        {
            throw new ArgumentException("RotaryDim must be even, positive and at most HeadDim.", nameof(a));
        }

        if (a.ScalingKind is < 1 or > 3)
        {
            throw new ArgumentException($"ScalingKind {a.ScalingKind} is not 1 (YaRN), 2 (longrope) or 3 (dynamic NTK).", nameof(a));
        }

        if (!(a.Factor > 0) || float.IsInfinity(a.Factor))
        {
            throw new ArgumentException("Factor must be positive and finite.", nameof(a));
        }

        Need(a.Q, (long)a.Tokens * a.QHeads * hd, nameof(a.Q));
        Need(a.K, (long)a.Tokens * a.KvHeads * hd, nameof(a.K));
        Need(a.Positions, a.Tokens, nameof(a.Positions));
        Need(a.CosSinCache, rot, nameof(a.CosSinCache));
        long rows = (long)(a.CosSinCache.Length / (ulong)rot);
        for (int tok = 0; tok < a.Tokens; tok++)
        {
            long pos = LoadInt(a.Positions.Address, a.Positions.Dtype, tok);
            if (pos < 0 || pos >= rows)
            {
                throw new ArgumentException($"Position {pos} of token {tok} is outside the cache's {rows} rows.", nameof(a));
            }
        }

        DType t = a.DType;
        DType ct = SpanType(a.CosSinCache, t);
        int half = rot / 2;
        for (int tok = 0; tok < a.Tokens; tok++)
        {
            long row = LoadInt(a.Positions.Address, a.Positions.Dtype, tok) * rot;
            for (int i = 0; i < half; i++)
            {
                double c = Load(a.CosSinCache.Address, ct, row + i);
                double s = Load(a.CosSinCache.Address, ct, row + half + i);
                RotatePair(a.Q.Address, t, (long)tok * a.QHeads * hd, a.QHeads, hd, i, i + half, c, s);
                RotatePair(a.K.Address, t, (long)tok * a.KvHeads * hd, a.KvHeads, hd, i, i + half, c, s);
            }
        }
    }
}
