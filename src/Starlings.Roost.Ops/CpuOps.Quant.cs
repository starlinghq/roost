using Starlings.Roost.Abi;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

// The quantisation ops reserved for M2 (R-077, R-043), as CPU references. FP8 follows Fp8 (OFP8 E4M3 and E5M2, saturating);
// 4-bit types are packed two per byte, low nibble first, and a span's length counts logical elements (C1). Scales and zero
// points are read in their span's own Dtype. GPU kernels for these ops do not exist yet; their status is untested.
public readonly partial struct CpuOps
{
    /// <inheritdoc/>
    /// <remarks>
    /// <para>Symmetric quantisation of each group of GroupSize consecutive elements of a row (0 means the whole row, so a
    /// per-tensor scale is one row spanning the tensor and a per-channel weight scale is one scale per row). OutputDType is
    /// F8E4M3, F8E5M2, I8 or I4 (packed). Scales is [Rows, Dim / GroupSize] in its span's Dtype (F32, F16, BF16 or F64).</para>
    /// <para>Per group: <c>amax</c> is the largest finite <c>|x|</c>; the scale <c>amax / qmax</c> (qmax 448, 57344, 127 or 7)
    /// is rounded into Scales and read back as <c>s</c>, and each element becomes <c>x / s</c> rounded to nearest even into the
    /// output: saturating for FP8 (NaN stays NaN), clamped to the type's range for integers (NaN gives 0). A group with
    /// <c>s == 0</c> (all zeros) stores scale 0 and zeros. <see cref="Dequantize"/> with the same Scales and no Zeros inverts it.</para>
    /// </remarks>
    public static void Quantize(in QuantizeArgs a)
    {
        int g = GroupOf(a.GroupSize, a.Dim, nameof(a));
        double qmax = a.OutputDType switch
        {
            DType.F8E4M3 or DType.F8E5M2 => Fp8.MaxOf(a.OutputDType),
            DType.I8 => 127.0,
            DType.I4 => 7.0,
            _ => throw new ArgumentException($"Quantize: OutputDType {a.OutputDType} is not F8E4M3, F8E5M2, I8 or I4.", nameof(a)),
        };
        DType st = ScaleType(a.Scales, "Quantize", nameof(a));
        long n = (long)a.Rows * a.Dim;
        int groups = a.Dim / g;
        Need(a.Input, n, nameof(a.Input));
        Need(a.Output, n, nameof(a.Output));
        Need(a.Scales, (long)a.Rows * groups, nameof(a.Scales));
        bool fp8 = Fp8.Is(a.OutputDType);
        long lo = a.OutputDType == DType.I8 ? -128 : -8;
        long hi = a.OutputDType == DType.I8 ? 127 : 7;
        for (long r = 0; r < a.Rows; r++)
        {
            for (int gi = 0; gi < groups; gi++)
            {
                long o = (r * a.Dim) + ((long)gi * g);
                double amax = 0;
                for (int i = 0; i < g; i++)
                {
                    double x = Math.Abs(Load(a.Input.Address, a.DType, o + i));
                    if (double.IsFinite(x) && x > amax)
                    {
                        amax = x;
                    }
                }

                long si = (r * groups) + gi;
                Store(a.Scales.Address, st, si, amax / qmax);
                double s = Load(a.Scales.Address, st, si);
                for (int i = 0; i < g; i++)
                {
                    double y = s == 0 ? 0.0 : Load(a.Input.Address, a.DType, o + i) / s;
                    if (fp8)
                    {
                        Store(a.Output.Address, a.OutputDType, o + i, y);
                    }
                    else
                    {
                        long q = double.IsNaN(y) ? 0 : (long)Math.Clamp(Math.Round(y, MidpointRounding.ToEven), lo, hi);
                        StoreInt(a.Output.Address, a.OutputDType, o + i, q);
                    }
                }
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para><c>Output = (q - z) * s</c> in double, rounded once into DType, per group of GroupSize consecutive elements of a
    /// row (0 means the whole row). InputDType is F8E4M3, F8E5M2, I8, U8, I4 or U4 (4-bit packed). Scales and the optional
    /// Zeros are [Rows, Dim / GroupSize] in their spans' own Dtypes; Zeros may be packed U4 or I4, an integer type or a float
    /// type, and must be absent for FP8.</para>
    /// <para>Without Zeros, z is 0 for signed and FP8 inputs and the midpoint for unsigned ones (8 for U4, 128 for U8), the
    /// symmetric GPTQ convention (vLLM's <c>uint4b8</c>). AWQ and GPTQ checkpoints are brought to this form by
    /// <see cref="Int4Weights"/>; GPTQ v1 checkpoints store <c>z - 1</c> and are corrected there, not here.</para>
    /// </remarks>
    public static void Dequantize(in DequantizeArgs a)
    {
        int g = GroupOf(a.GroupSize, a.Dim, nameof(a));
        double defaultZero = a.InputDType switch
        {
            DType.F8E4M3 or DType.F8E5M2 or DType.I8 or DType.I4 => 0.0,
            DType.U8 => 128.0,
            DType.U4 => 8.0,
            _ => throw new ArgumentException($"Dequantize: InputDType {a.InputDType} is not F8E4M3, F8E5M2, I8, U8, I4 or U4.", nameof(a)),
        };
        DType st = ScaleType(a.Scales, "Dequantize", nameof(a));
        bool zeros = a.Zeros.Address != 0;
        if (zeros && Fp8.Is(a.InputDType))
        {
            throw new ArgumentException("Dequantize: FP8 input takes no zero points.", nameof(a));
        }

        long n = (long)a.Rows * a.Dim;
        int groups = a.Dim / g;
        Need(a.Input, n, nameof(a.Input));
        Need(a.Output, n, nameof(a.Output));
        Need(a.Scales, (long)a.Rows * groups, nameof(a.Scales));
        if (zeros)
        {
            Need(a.Zeros, (long)a.Rows * groups, nameof(a.Zeros));
        }

        for (long r = 0; r < a.Rows; r++)
        {
            for (int gi = 0; gi < groups; gi++)
            {
                long si = (r * groups) + gi;
                double s = Load(a.Scales.Address, st, si);
                double z = zeros ? Load(a.Zeros.Address, a.Zeros.Dtype, si) : defaultZero;
                long o = (r * a.Dim) + ((long)gi * g);
                for (int i = 0; i < g; i++)
                {
                    Store(a.Output.Address, a.DType, o + i, (Load(a.Input.Address, a.InputDType, o + i) - z) * s);
                }
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Dequantise, then Gemm: <c>W[n, k] = (q[n, k] - z[n, k / GroupSize]) * s[n, k / GroupSize]</c> rounded once into
    /// DType (what <see cref="Dequantize"/> with Rows = N, Dim = K writes), then <c>D[m, n] = sum_k A[m, k] * W[n, k]</c> in
    /// double, rounded once into DType. GroupSize 0 means K.</para>
    /// <para>The CPU reference reads B in the plain layout: [N, K] row-major, B's Dtype U4 or I4, packed low nibble first;
    /// Scales and the optional Zeros are [N, K / GroupSize] in their spans' Dtypes, with Dequantize's default zero point when
    /// Zeros is absent (8 for U4). The GPU kernel's repacked (Marlin) layout is not defined in C3 yet, so a GPU result is
    /// compared after repacking the same weights, never by sharing a buffer. Act-order (g_idx) checkpoints are not expressible.</para>
    /// </remarks>
    public static void GemmW4A16(in GemmW4A16Args a)
    {
        DType bt = a.B.Dtype;
        double defaultZero = bt switch
        {
            DType.U4 => 8.0,
            DType.I4 => 0.0,
            _ => throw new ArgumentException($"GemmW4A16: B's Dtype {bt} is not U4 or I4.", nameof(a)),
        };
        if (a.K > int.MaxValue)
        {
            throw new ArgumentException("GemmW4A16: K is too large.", nameof(a));
        }

        long g = GroupOf(a.GroupSize, (int)a.K, nameof(a));
        long groups = a.K / g;
        DType st = ScaleType(a.Scales, "GemmW4A16", nameof(a));
        bool zeros = a.Zeros.Address != 0;
        Need(a.A, a.M * a.K, nameof(a.A));
        Need(a.B, a.N * a.K, nameof(a.B));
        Need(a.D, a.M * a.N, nameof(a.D));
        Need(a.Scales, a.N * groups, nameof(a.Scales));
        if (zeros)
        {
            Need(a.Zeros, a.N * groups, nameof(a.Zeros));
        }

        for (long n = 0; n < a.N; n++)
        {
            for (long m = 0; m < a.M; m++)
            {
                double acc = 0;
                for (long k = 0; k < a.K; k++)
                {
                    long si = (n * groups) + (k / g);
                    double z = zeros ? Load(a.Zeros.Address, a.Zeros.Dtype, si) : defaultZero;
                    double w = RoundTo((Load(a.B.Address, bt, (n * a.K) + k) - z) * Load(a.Scales.Address, st, si), a.DType);
                    acc += Load(a.A.Address, a.DType, (m * a.K) + k) * w;
                }

                Store(a.D.Address, a.DType, (m * a.N) + n, acc);
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <c>D[m, n] = (sum_k A[m, k] * B[n, k]) * AScale[m] * BScale[n]</c> in double, rounded once into DType. A and B are FP8
    /// in their spans' Dtypes (E4M3 or E5M2, each its own); the FP8 products are exact. AScale has 1 element (per tensor) or M
    /// (per token); BScale has 1 (per tensor) or N (per output channel); both are read in their spans' Dtypes.
    /// </remarks>
    public static void GemmFp8(in GemmFp8Args a)
    {
        DType at = a.A.Dtype;
        DType bt = a.B.Dtype;
        if (!Fp8.Is(at) || !Fp8.Is(bt))
        {
            throw new ArgumentException($"GemmFp8: A ({at}) and B ({bt}) must be F8E4M3 or F8E5M2.", nameof(a));
        }

        bool perRow = PerTensorOrPer(a.AScale, a.M, "AScale", "M");
        bool perCol = PerTensorOrPer(a.BScale, a.N, "BScale", "N");
        DType ast = ScaleType(a.AScale, "GemmFp8", nameof(a));
        DType bst = ScaleType(a.BScale, "GemmFp8", nameof(a));
        Need(a.A, a.M * a.K, nameof(a.A));
        Need(a.B, a.N * a.K, nameof(a.B));
        Need(a.D, a.M * a.N, nameof(a.D));
        for (long m = 0; m < a.M; m++)
        {
            double sa = Load(a.AScale.Address, ast, perRow ? m : 0);
            for (long n = 0; n < a.N; n++)
            {
                double acc = 0;
                for (long k = 0; k < a.K; k++)
                {
                    acc += Load(a.A.Address, at, (m * a.K) + k) * Load(a.B.Address, bt, (n * a.K) + k);
                }

                Store(a.D.Address, a.DType, (m * a.N) + n, acc * sa * Load(a.BScale.Address, bst, perCol ? n : 0));
            }
        }
    }

    private static int GroupOf(int groupSize, int dim, string what)
    {
        int g = groupSize == 0 ? dim : groupSize;
        if (g <= 0 || dim % g != 0)
        {
            throw new ArgumentException($"GroupSize {groupSize} must be 0 or a positive divisor of {dim}.", what);
        }

        return g;
    }

    private static DType ScaleType(in DeviceSpan s, string op, string what) => s.Dtype is DType.F32 or DType.F16 or DType.BF16 or DType.F64
        ? s.Dtype
        : throw new ArgumentException($"{op}: scales must be F32, F16, BF16 or F64, not {s.Dtype}.", what);

    private static bool PerTensorOrPer(in DeviceSpan s, long count, string name, string dim)
    {
        if (s.Address == 0 || (s.Length != 1 && s.Length != (ulong)count))
        {
            throw new ArgumentException($"GemmFp8: {name} must have 1 element or {dim} ({count}), not {s.Length}.");
        }

        return s.Length != 1;
    }
}
