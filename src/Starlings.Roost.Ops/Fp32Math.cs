namespace Starlings.Roost.Ops;

/// <summary>
/// The float32 arithmetic of <see cref="CpuOps.Sample"/> under <see cref="Abi.SampleMath.Fp32"/>: exp, log and a row sum built
/// only from correctly rounded IEEE binary32 add, multiply, divide, fused multiply-add, round-to-nearest-even and exact
/// power-of-two scaling, so a CUDA kernel that performs the same operations in the same order (<c>__fadd_rn</c>,
/// <c>__fmul_rn</c>, <c>__fdiv_rn</c>, <c>__fmaf_rn</c>, <c>rintf</c>, <c>ldexpf</c>; no <c>--use_fast_math</c>, no
/// <c>-fmad</c> contraction) produces the same bits. Platform <c>expf</c> and <c>logf</c> are not used: neither .NET's nor
/// CUDA's is specified to the bit.
/// </summary>
/// <remarks>
/// Both functions use textbook schemes with series coefficients, not tuned minimax constants: exp reduces x = k ln 2 + r with
/// a two-part ln 2 and sums the Taylor series of e^r to r^7 (|r| &lt;= ln 2 / 2, truncation below 2^-27); log writes
/// x = 2^e m with m in (sqrt(2)/2, sqrt(2)], s = (m - 1) / (m + 1), and sums 2 atanh(s) to s^11. Their accuracy (a few ulp)
/// is checked in the tests; what the sampler needs is that the definition is exact.
/// </remarks>
internal static class Fp32Math
{
    // ln 2 split so that k * Ln2Hi is exact for |k| < 2^15: Ln2Hi = 355 / 512 has 9 significant bits.
    private const float Ln2Hi = 0.693359375f;
    private const float Ln2Lo = -2.12194440054690583e-4f;
    private const float Log2E = 1.44269504088896341f;

    /// <summary>exp(x) in binary32: +inf above 88.7228, +0 below -103.9721, NaN for NaN.</summary>
    /// <param name="x">The argument.</param>
    /// <returns>The value.</returns>
    public static float Exp(float x)
    {
        if (float.IsNaN(x))
        {
            return x;
        }

        if (x > 88.72283935546875f)
        {
            return float.PositiveInfinity;
        }

        if (x < -103.97208404541015625f)
        {
            return 0f;
        }

        float k = MathF.Round(x * Log2E, MidpointRounding.ToEven);
        float r = MathF.FusedMultiplyAdd(-k, Ln2Hi, x);
        r = MathF.FusedMultiplyAdd(-k, Ln2Lo, r);
        float p = 1f / 5040f;
        p = MathF.FusedMultiplyAdd(p, r, 1f / 720f);
        p = MathF.FusedMultiplyAdd(p, r, 1f / 120f);
        p = MathF.FusedMultiplyAdd(p, r, 1f / 24f);
        p = MathF.FusedMultiplyAdd(p, r, 1f / 6f);
        p = MathF.FusedMultiplyAdd(p, r, 0.5f);
        p = MathF.FusedMultiplyAdd(p, r, 1f);
        p = MathF.FusedMultiplyAdd(p, r, 1f);
        return MathF.ScaleB(p, (int)k);
    }

    /// <summary>ln(x) in binary32 for a positive finite x (the sampler calls it on a softmax denominator, which is at least 1); NaN otherwise, -inf for 0.</summary>
    /// <param name="x">The argument.</param>
    /// <returns>The value.</returns>
    public static float Log(float x)
    {
        if (float.IsNaN(x) || x < 0f)
        {
            return float.NaN;
        }

        if (x == 0f)
        {
            return float.NegativeInfinity;
        }

        if (float.IsPositiveInfinity(x))
        {
            return x;
        }

        int e = 0;
        if (x < 1.17549435e-38f)
        {
            x *= 16777216f; // 2^24, exact: a subnormal becomes normal
            e = -24;
        }

        int bits = BitConverter.SingleToInt32Bits(x);
        e += ((bits >> 23) & 0xFF) - 127;
        float m = BitConverter.Int32BitsToSingle((bits & 0x007FFFFF) | 0x3F800000); // [1, 2), exact
        if (m > 1.41421356f)
        {
            m *= 0.5f; // exact
            e++;
        }

        float f = m - 1f; // exact (Sterbenz)
        float s = f / (2f + f);
        float z = s * s;
        float q = 2f / 11f;
        q = MathF.FusedMultiplyAdd(q, z, 2f / 9f);
        q = MathF.FusedMultiplyAdd(q, z, 2f / 7f);
        q = MathF.FusedMultiplyAdd(q, z, 2f / 5f);
        q = MathF.FusedMultiplyAdd(q, z, 2f / 3f);
        float r = z * q; // 2 atanh(s) = 2s + s * r
        float hfsq = 0.5f * f * f;
        float dk = e;

        // ln m = f - hfsq + s * (hfsq + r), since 2s = f - s * f; the small terms are added first.
        float lnm = f - (hfsq - (s * (hfsq + r)));
        return MathF.FusedMultiplyAdd(dk, Ln2Hi, MathF.FusedMultiplyAdd(dk, Ln2Lo, lnm));
    }

    /// <summary>
    /// The sampler's row sum: 32 lanes, lane j adding the values at indices j, j + 32, j + 64, ... in increasing order from +0,
    /// then lane j += lane j + w for w = 16, 8, 4, 2, 1 (j &lt; w), and lane 0 is the sum. One warp per row computes exactly this
    /// with a strided loop and a <c>__shfl_down_sync</c> tree.
    /// </summary>
    /// <param name="lanes">32 lanes, filled by <see cref="AddToLane"/> from zero.</param>
    /// <returns>The sum.</returns>
    public static float ReduceLanes(Span<float> lanes)
    {
        for (int w = 16; w >= 1; w >>= 1)
        {
            for (int j = 0; j < w; j++)
            {
                lanes[j] += lanes[j + w];
            }
        }

        return lanes[0];
    }

    /// <summary>Adds the value at index <paramref name="index"/> to its lane (index mod 32). Values must arrive in increasing index order.</summary>
    /// <param name="lanes">The 32 lanes.</param>
    /// <param name="index">The element's index in the row.</param>
    /// <param name="value">The element.</param>
    public static void AddToLane(Span<float> lanes, int index, float value) => lanes[index & 31] += value;
}
