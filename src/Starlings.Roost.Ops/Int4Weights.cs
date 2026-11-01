using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops;

/// <summary>
/// The AWQ and GPTQ 4-bit weight conventions, as CPU references: how each picks a group's scale and zero point, and how each
/// packs its checkpoint tensors. Everything converts to and from the plain form <see cref="CpuOps.Dequantize"/> and
/// <see cref="CpuOps.GemmW4A16"/> read: q as [N, K] unsigned 4-bit values (one per byte here, packed with <see cref="PackU4"/>),
/// scales and zero points as [N, K / groupSize], <c>w = (q - z) * s</c>. N is the output dimension and K the input dimension
/// of the linear layer, so [N, K] is the layout of a dense weight.
/// </summary>
/// <remarks>
/// <para>Quantisation here is round-to-nearest with each method's scale and zero-point rule, not the GPTQ error-compensating
/// procedure or AWQ's activation-aware search: those choose the weights before this rule applies and run at tool time. The
/// rules are evaluated in double with the scale rounded once to its storage type first, so a value at a rounding boundary can
/// land one step away from the upstream tools, which evaluate in the model dtype (AWQ) or float32 (GPTQ).</para>
/// <para>Checkpoint layouts, as the formats' packers (AutoAWQ <c>WQLinear_GEMM.from_linear</c>, AutoGPTQ <c>QuantLinear.pack</c>)
/// define them; written from the format description, no upstream source is adopted or pinned, and the packing is checked only
/// against hand-computed words, not yet against a real checkpoint: AWQ <c>qweight</c> is int32 [K, N / 8], each word holding
/// eight output columns with nibble i (bits 4i to 4i + 3) taking column <c>8c + {0, 2, 4, 6, 1, 3, 5, 7}[i]</c>, and
/// <c>qzeros</c> is int32 [K / groupSize, N / 8] packed the same way. GPTQ <c>qweight</c> is int32 [K / 8, N], nibble j of
/// row r holding input row <c>8r + j</c>, and <c>qzeros</c> is int32 [K / groupSize, N / 8], nibble j holding column
/// <c>8c + j</c>. Both store scales as [K / groupSize, N]. The GPTQ v1 format stores <c>z - 1</c> in qzeros (so it cannot
/// hold z = 0, and its zero points read back as 1 to 16); the v2 format stores z.</para>
/// <para>Act-order GPTQ checkpoints (a <c>g_idx</c> that is not <c>k / groupSize</c>) are <c>unsupported</c>: C3's GemmW4A16
/// has no input permutation. The Marlin repack the GPU kernel needs is not here: C3 names the layout but does not define it.</para>
/// </remarks>
public static class Int4Weights
{
    /// <summary>Nibble i of an AWQ word holds column <c>AwqOrder[i]</c> of its group of eight.</summary>
    private static ReadOnlySpan<int> AwqOrder => [0, 2, 4, 6, 1, 3, 5, 7];

    /// <summary>
    /// AWQ's rule (AutoAWQ <c>pseudo_quantize_tensor</c>, zero point on): per group <c>s = max(max - min, 1e-5) / 15</c>,
    /// <c>z = clamp(-round(min / s), 0, 15)</c>, <c>q = clamp(round(w / s) + z, 0, 15)</c>, rounding half to even.
    /// </summary>
    /// <param name="w">[N, K] weights.</param>
    /// <param name="n">Output dimension.</param>
    /// <param name="k">Input dimension.</param>
    /// <param name="groupSize">Weights per group along K; 0 means K.</param>
    /// <param name="scaleType">Storage type of the scales (F16, BF16, F32 or F64); s is rounded to it before use.</param>
    /// <param name="q">[N, K] output values in [0, 15].</param>
    /// <param name="scales">[N, K / groupSize] output scales, already rounded to <paramref name="scaleType"/>.</param>
    /// <param name="zeros">[N, K / groupSize] output zero points in [0, 15].</param>
    public static void QuantizeAwq(ReadOnlySpan<double> w, int n, int k, int groupSize, DType scaleType, Span<byte> q, Span<double> scales, Span<byte> zeros)
    {
        int g = Check(w.Length, n, k, groupSize, q.Length, scales.Length, zeros.Length);
        int groups = k / g;
        for (int r = 0; r < n; r++)
        {
            for (int gi = 0; gi < groups; gi++)
            {
                ReadOnlySpan<double> x = w.Slice((r * k) + (gi * g), g);
                double max = double.NegativeInfinity, min = double.PositiveInfinity;
                foreach (double v in x)
                {
                    max = Math.Max(max, v);
                    min = Math.Min(min, v);
                }

                double s = CpuElements.RoundTo(Math.Max(max - min, 1e-5) / 15.0, scaleType);
                double z = Math.Clamp(-Math.Round(min / s, MidpointRounding.ToEven), 0, 15);
                Emit(x, s, z, q.Slice((r * k) + (gi * g), g));
                scales[(r * groups) + gi] = s;
                zeros[(r * groups) + gi] = (byte)z;
            }
        }
    }

    /// <summary>
    /// GPTQ's rule (the <c>Quantizer.find_params</c> of GPTQ and AutoGPTQ, 4 bits): per group <c>lo = min(min, 0)</c>,
    /// <c>hi = max(max, 0)</c>; symmetric sets <c>hi = max(|lo|, hi)</c> and, only when <c>lo &lt; 0</c>, <c>lo = -hi</c>; a group with
    /// <c>lo == hi == 0</c> uses -1 and 1. Then <c>s = (hi - lo) / 15</c>, <c>z = 8</c> (symmetric) or <c>round(-lo / s)</c>,
    /// <c>q = clamp(round(w / s) + z, 0, 15)</c>, rounding half to even.
    /// </summary>
    /// <remarks>Faithful to the rule's quirk: a symmetric group with no negative weight keeps <c>lo = 0</c>, so its upper half clips at 15.</remarks>
    /// <param name="w">[N, K] weights.</param>
    /// <param name="n">Output dimension.</param>
    /// <param name="k">Input dimension.</param>
    /// <param name="groupSize">Weights per group along K; 0 means K (GPTQ's group size -1).</param>
    /// <param name="symmetric">GPTQ's <c>sym</c>.</param>
    /// <param name="scaleType">Storage type of the scales (F16, BF16, F32 or F64); s is rounded to it before use.</param>
    /// <param name="q">[N, K] output values in [0, 15].</param>
    /// <param name="scales">[N, K / groupSize] output scales, already rounded to <paramref name="scaleType"/>.</param>
    /// <param name="zeros">[N, K / groupSize] output zero points in [0, 15].</param>
    public static void QuantizeGptq(ReadOnlySpan<double> w, int n, int k, int groupSize, bool symmetric, DType scaleType, Span<byte> q, Span<double> scales, Span<byte> zeros)
    {
        int g = Check(w.Length, n, k, groupSize, q.Length, scales.Length, zeros.Length);
        int groups = k / g;
        for (int r = 0; r < n; r++)
        {
            for (int gi = 0; gi < groups; gi++)
            {
                ReadOnlySpan<double> x = w.Slice((r * k) + (gi * g), g);
                double lo = 0, hi = 0;
                foreach (double v in x)
                {
                    lo = Math.Min(lo, v);
                    hi = Math.Max(hi, v);
                }

                if (symmetric)
                {
                    hi = Math.Max(Math.Abs(lo), hi);
                    if (lo < 0)
                    {
                        lo = -hi;
                    }
                }

                if (lo == 0 && hi == 0)
                {
                    lo = -1;
                    hi = 1;
                }

                double s = CpuElements.RoundTo((hi - lo) / 15.0, scaleType);
                double z = symmetric ? 8 : Math.Clamp(Math.Round(-lo / s, MidpointRounding.ToEven), 0, 15);
                Emit(x, s, z, q.Slice((r * k) + (gi * g), g));
                scales[(r * groups) + gi] = s;
                zeros[(r * groups) + gi] = (byte)z;
            }
        }
    }

    /// <summary>Packs unsigned 4-bit values two per byte, low nibble first (C1's U4).</summary>
    /// <param name="values">Values in [0, 15].</param>
    /// <param name="packed">At least <c>(values.Length + 1) / 2</c> bytes.</param>
    public static void PackU4(ReadOnlySpan<byte> values, Span<byte> packed)
    {
        Need(packed.Length, (values.Length + 1) / 2, nameof(packed));
        packed[..((values.Length + 1) / 2)].Clear();
        for (int i = 0; i < values.Length; i++)
        {
            Nibble15(values[i]);
            packed[i >> 1] |= (byte)(values[i] << (4 * (i & 1)));
        }
    }

    /// <summary>Unpacks C1's U4: element i is the low nibble of byte i / 2 when i is even, the high nibble otherwise.</summary>
    /// <param name="packed">Packed bytes.</param>
    /// <param name="values">Output values; its length is the element count.</param>
    public static void UnpackU4(ReadOnlySpan<byte> packed, Span<byte> values)
    {
        Need(packed.Length, (values.Length + 1) / 2, nameof(packed));
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (byte)((packed[i >> 1] >> (4 * (i & 1))) & 0xF);
        }
    }

    /// <summary>Packs [N, K] values and [N, K / groupSize] zero points into AWQ's <c>qweight</c> [K, N / 8] and <c>qzeros</c> [K / groupSize, N / 8].</summary>
    /// <param name="q">[N, K] values in [0, 15].</param>
    /// <param name="zeros">[N, K / groupSize] zero points in [0, 15].</param>
    /// <param name="n">Output dimension, a multiple of 8.</param>
    /// <param name="k">Input dimension.</param>
    /// <param name="groupSize">Weights per group; 0 means K.</param>
    /// <param name="qweight">Output words.</param>
    /// <param name="qzeros">Output words.</param>
    public static void PackAwq(ReadOnlySpan<byte> q, ReadOnlySpan<byte> zeros, int n, int k, int groupSize, Span<int> qweight, Span<int> qzeros)
    {
        int groups = Groups(n, k, groupSize, multipleOf8N: true);
        Need(q.Length, n * k, nameof(q));
        Need(zeros.Length, n * groups, nameof(zeros));
        Need(qweight.Length, k * (n / 8), nameof(qweight));
        Need(qzeros.Length, groups * (n / 8), nameof(qzeros));
        PackColumnsAwq(q, k, n, k, qweight);
        PackColumnsAwq(zeros, groups, n, groups, qzeros);
    }

    /// <summary>Unpacks AWQ's <c>qweight</c> and <c>qzeros</c> into [N, K] values and [N, K / groupSize] zero points.</summary>
    /// <param name="qweight">[K, N / 8] words.</param>
    /// <param name="qzeros">[K / groupSize, N / 8] words.</param>
    /// <param name="n">Output dimension, a multiple of 8.</param>
    /// <param name="k">Input dimension.</param>
    /// <param name="groupSize">Weights per group; 0 means K.</param>
    /// <param name="q">[N, K] output values.</param>
    /// <param name="zeros">[N, K / groupSize] output zero points.</param>
    public static void UnpackAwq(ReadOnlySpan<int> qweight, ReadOnlySpan<int> qzeros, int n, int k, int groupSize, Span<byte> q, Span<byte> zeros)
    {
        int groups = Groups(n, k, groupSize, multipleOf8N: true);
        Need(qweight.Length, k * (n / 8), nameof(qweight));
        Need(qzeros.Length, groups * (n / 8), nameof(qzeros));
        Need(q.Length, n * k, nameof(q));
        Need(zeros.Length, n * groups, nameof(zeros));
        UnpackColumnsAwq(qweight, k, n, q, k);
        UnpackColumnsAwq(qzeros, groups, n, zeros, groups);
    }

    /// <summary>
    /// Packs [N, K] values and [N, K / groupSize] zero points into GPTQ's <c>qweight</c> [K / 8, N] and <c>qzeros</c>
    /// [K / groupSize, N / 8]. The v1 format stores <c>z - 1</c> and refuses z = 0, which it cannot represent.
    /// </summary>
    /// <param name="q">[N, K] values in [0, 15].</param>
    /// <param name="zeros">[N, K / groupSize] zero points in [0, 15] (in [1, 15] for v1).</param>
    /// <param name="n">Output dimension, a multiple of 8.</param>
    /// <param name="k">Input dimension, a multiple of 8.</param>
    /// <param name="groupSize">Weights per group; 0 means K.</param>
    /// <param name="v1">Store <c>z - 1</c> (the GPTQ v1 checkpoint format).</param>
    /// <param name="qweight">Output words.</param>
    /// <param name="qzeros">Output words.</param>
    public static void PackGptq(ReadOnlySpan<byte> q, ReadOnlySpan<byte> zeros, int n, int k, int groupSize, bool v1, Span<int> qweight, Span<int> qzeros)
    {
        int groups = Groups(n, k, groupSize, multipleOf8N: true);
        if (k % 8 != 0)
        {
            throw new ArgumentException($"K ({k}) must be a multiple of 8 for GPTQ packing.", nameof(k));
        }

        Need(q.Length, n * k, nameof(q));
        Need(zeros.Length, n * groups, nameof(zeros));
        Need(qweight.Length, (k / 8) * n, nameof(qweight));
        Need(qzeros.Length, groups * (n / 8), nameof(qzeros));
        for (int row = 0; row < k / 8; row++)
        {
            for (int col = 0; col < n; col++)
            {
                uint word = 0;
                for (int j = 0; j < 8; j++)
                {
                    word |= (uint)Nibble15(q[(col * k) + (8 * row) + j]) << (4 * j);
                }

                qweight[(row * n) + col] = unchecked((int)word);
            }
        }

        for (int gi = 0; gi < groups; gi++)
        {
            for (int c = 0; c < n / 8; c++)
            {
                uint word = 0;
                for (int j = 0; j < 8; j++)
                {
                    int z = Nibble15(zeros[(((8 * c) + j) * groups) + gi]);
                    if (v1 && z == 0)
                    {
                        throw new ArgumentException($"GPTQ v1 cannot store zero point 0 (column {(8 * c) + j}, group {gi}).", nameof(zeros));
                    }

                    word |= (uint)(v1 ? z - 1 : z) << (4 * j);
                }

                qzeros[(gi * (n / 8)) + c] = unchecked((int)word);
            }
        }
    }

    /// <summary>
    /// Unpacks GPTQ's <c>qweight</c> and <c>qzeros</c> into [N, K] values and [N, K / groupSize] zero points. A v1 checkpoint's
    /// zero points read back as the stored nibble plus 1 (1 to 16), so store them in a type wider than U4 (U8) for
    /// <see cref="CpuOps.Dequantize"/>. A <paramref name="gIdx"/> that is not <c>k / groupSize</c> (act order) is refused.
    /// </summary>
    /// <param name="qweight">[K / 8, N] words.</param>
    /// <param name="qzeros">[K / groupSize, N / 8] words.</param>
    /// <param name="gIdx">Optional [K] group index of each input row; empty to skip the check.</param>
    /// <param name="n">Output dimension, a multiple of 8.</param>
    /// <param name="k">Input dimension, a multiple of 8.</param>
    /// <param name="groupSize">Weights per group; 0 means K.</param>
    /// <param name="v1">The checkpoint stores <c>z - 1</c>.</param>
    /// <param name="q">[N, K] output values.</param>
    /// <param name="zeros">[N, K / groupSize] output zero points.</param>
    public static void UnpackGptq(ReadOnlySpan<int> qweight, ReadOnlySpan<int> qzeros, ReadOnlySpan<int> gIdx, int n, int k, int groupSize, bool v1, Span<byte> q, Span<byte> zeros)
    {
        int groups = Groups(n, k, groupSize, multipleOf8N: true);
        if (k % 8 != 0)
        {
            throw new ArgumentException($"K ({k}) must be a multiple of 8 for GPTQ packing.", nameof(k));
        }

        Need(qweight.Length, (k / 8) * n, nameof(qweight));
        Need(qzeros.Length, groups * (n / 8), nameof(qzeros));
        Need(q.Length, n * k, nameof(q));
        Need(zeros.Length, n * groups, nameof(zeros));
        if (!gIdx.IsEmpty)
        {
            Need(gIdx.Length, k, nameof(gIdx));
            int g = k / groups;
            for (int i = 0; i < k; i++)
            {
                if (gIdx[i] != i / g)
                {
                    throw new NotSupportedException($"Act-order GPTQ is unsupported: g_idx[{i}] is {gIdx[i]}, not {i / g}.");
                }
            }
        }

        for (int row = 0; row < k / 8; row++)
        {
            for (int col = 0; col < n; col++)
            {
                uint word = unchecked((uint)qweight[(row * n) + col]);
                for (int j = 0; j < 8; j++)
                {
                    q[(col * k) + (8 * row) + j] = (byte)((word >> (4 * j)) & 0xF);
                }
            }
        }

        for (int gi = 0; gi < groups; gi++)
        {
            for (int c = 0; c < n / 8; c++)
            {
                uint word = unchecked((uint)qzeros[(gi * (n / 8)) + c]);
                for (int j = 0; j < 8; j++)
                {
                    int z = (int)((word >> (4 * j)) & 0xF);
                    zeros[(((8 * c) + j) * groups) + gi] = (byte)(v1 ? z + 1 : z);
                }
            }
        }
    }

    /// <summary>Transposes a checkpoint's [K / groupSize, N] scales into the plain [N, K / groupSize] order.</summary>
    /// <typeparam name="T">Element type (for example the raw half bits).</typeparam>
    /// <param name="source">[groups, N].</param>
    /// <param name="groups">Groups along K.</param>
    /// <param name="n">Output dimension.</param>
    /// <param name="destination">[N, groups].</param>
    public static void TransposeScales<T>(ReadOnlySpan<T> source, int groups, int n, Span<T> destination)
    {
        Need(source.Length, groups * n, nameof(source));
        Need(destination.Length, groups * n, nameof(destination));
        for (int gi = 0; gi < groups; gi++)
        {
            for (int c = 0; c < n; c++)
            {
                destination[(c * groups) + gi] = source[(gi * n) + c];
            }
        }
    }

    private static void Emit(ReadOnlySpan<double> x, double s, double z, Span<byte> q)
    {
        for (int i = 0; i < x.Length; i++)
        {
            q[i] = (byte)Math.Clamp(Math.Round(x[i] / s, MidpointRounding.ToEven) + z, 0, 15);
        }
    }

    // AWQ packs along N: word (row, c) of a [rows, N / 8] tensor holds columns 8c .. 8c + 7 of that row, in AwqOrder.
    private static void PackColumnsAwq(ReadOnlySpan<byte> plain, int rows, int n, int plainStride, Span<int> words)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int c = 0; c < n / 8; c++)
            {
                uint word = 0;
                for (int i = 0; i < 8; i++)
                {
                    word |= (uint)Nibble15(plain[(((8 * c) + AwqOrder[i]) * plainStride) + row]) << (4 * i);
                }

                words[(row * (n / 8)) + c] = unchecked((int)word);
            }
        }
    }

    private static void UnpackColumnsAwq(ReadOnlySpan<int> words, int rows, int n, Span<byte> plain, int plainStride)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int c = 0; c < n / 8; c++)
            {
                uint word = unchecked((uint)words[(row * (n / 8)) + c]);
                for (int i = 0; i < 8; i++)
                {
                    plain[(((8 * c) + AwqOrder[i]) * plainStride) + row] = (byte)((word >> (4 * i)) & 0xF);
                }
            }
        }
    }

    private static int Nibble15(byte v) => v <= 15 ? v : throw new ArgumentOutOfRangeException(nameof(v), v, "A 4-bit value must be in [0, 15].");

    private static int Groups(int n, int k, int groupSize, bool multipleOf8N)
    {
        if (n <= 0 || k <= 0 || (multipleOf8N && n % 8 != 0))
        {
            throw new ArgumentException($"N ({n}) must be a positive multiple of 8 and K ({k}) positive.", nameof(n));
        }

        int g = groupSize == 0 ? k : groupSize;
        if (g <= 0 || k % g != 0)
        {
            throw new ArgumentException($"GroupSize {groupSize} must be 0 or a positive divisor of K ({k}).", nameof(groupSize));
        }

        return k / g;
    }

    private static int Check(int wLength, int n, int k, int groupSize, int qLength, int scalesLength, int zerosLength)
    {
        int groups = Groups(n, k, groupSize, multipleOf8N: false);
        Need(wLength, n * k, "w");
        Need(qLength, n * k, "q");
        Need(scalesLength, n * groups, "scales");
        Need(zerosLength, n * groups, "zeros");
        return k / groups;
    }

    private static void Need(int have, int need, string name)
    {
        if (have < need)
        {
            throw new ArgumentException($"{name} has {have} elements; {need} are needed.", name);
        }
    }
}
