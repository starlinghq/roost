using System.Buffers.Binary;
using System.Globalization;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Models.Tiny;

/// <summary>
/// Deterministic weights for the tiny CI models. Nothing is stored: each slot's values come from the model seed and the
/// slot name through SplitMix64 and IEEE double arithmetic, so a slot has the same values on any host, in any slot order,
/// and the repository carries no weight files. Values are rounded to the slot's element type, so a reference computed from
/// <see cref="Generate"/> sees exactly what the backend reads.
/// </summary>
public static class TinyWeights
{
    private const ulong FnvOffset = 14695981039346656037UL;
    private const ulong FnvPrime = 1099511628211UL;

    /// <summary>The generator state for one slot of one model.</summary>
    /// <param name="modelSeed">The configuration's seed.</param>
    /// <param name="slotName">The weight slot's name.</param>
    /// <returns>The first SplitMix64 state.</returns>
    public static ulong SlotSeed(ulong modelSeed, string slotName)
    {
        ArgumentNullException.ThrowIfNull(slotName);
        ulong h = FnvOffset;
        foreach (char c in slotName)
        {
            h = (h ^ c) * FnvPrime;
        }

        return modelSeed ^ h;
    }

    /// <summary>The values of one weight slot, rounded to <paramref name="dtype"/>.</summary>
    /// <param name="modelSeed">The configuration's seed.</param>
    /// <param name="slotName">The weight slot's name, as in <c>WeightSlots</c>.</param>
    /// <param name="shape">The slot's shape, comma-separated, as in <c>WeightSlots</c>.</param>
    /// <param name="dtype">Stored element type: F32 or BF16.</param>
    /// <returns>The values in row-major order.</returns>
    /// <remarks>
    /// Norm weights (a name ending in <c>norm</c>) are 1 plus up to 10 percent of noise. The embedding table is uniform in
    /// [-1, 1). A_log is the log of a uniform draw from [0.1, 1). Every other slot is uniform in [-a, a) with a = 1.5 / sqrt(fan-in), the last dimension of the shape, so
    /// activations stay near unit size through the layers and the logits are not dominated by one token.
    /// </remarks>
    public static double[] Generate(ulong modelSeed, string slotName, string shape, DType dtype)
    {
        ArgumentNullException.ThrowIfNull(shape);
        int[] dims = shape.Split(',').Select(d => int.Parse(d, CultureInfo.InvariantCulture)).ToArray();
        long elements = dims.Aggregate(1L, (a, d) => a * d);
        double amplitude;
        double offset = 0;
        if (slotName.EndsWith("norm", StringComparison.Ordinal))
        {
            amplitude = 0.1;
            offset = 1;
        }
        else if (slotName == "embed")
        {
            amplitude = 1.0;
        }
        else
        {
            amplitude = 1.5 / Math.Sqrt(dims[^1]);
        }

        // A_log is the log of the decay rate, as in the checkpoint's initialisation (log of a uniform draw), drawn from
        // [0.1, 1) so exp(A_log) is a moderate rate and the recurrent state carries across tokens instead of vanishing.
        bool decayLog = slotName.EndsWith(".A_log", StringComparison.Ordinal);
        ulong state = SlotSeed(modelSeed, slotName);
        var values = new double[elements];
        for (long i = 0; i < elements; i++)
        {
            double u = (NextUInt64(ref state) >> 11) * (1.0 / 9007199254740992.0);
            values[i] = Round(decayLog ? Math.Log(0.1 + (0.9 * u)) : offset + (((2 * u) - 1) * amplitude), dtype);
        }

        return values;
    }

    /// <summary>Rounds a value to an element type, as the backend would read it back.</summary>
    /// <param name="value">The value.</param>
    /// <param name="dtype">F32 or BF16.</param>
    /// <returns>The nearest representable value (ties to even).</returns>
    public static double Round(double value, DType dtype) => dtype switch
    {
        DType.F32 => (float)value,
        DType.BF16 => BFloat16ToSingle(SingleToBFloat16((float)value)),
        _ => throw new NotSupportedException("Tiny weights are F32 or BF16, got " + dtype),
    };

    /// <summary>Writes values into native memory in <paramref name="dtype"/>.</summary>
    /// <param name="values">The values, already rounded or not.</param>
    /// <param name="dtype">F32 or BF16.</param>
    /// <param name="address">Destination with room for the values.</param>
    public static unsafe void Write(ReadOnlySpan<double> values, DType dtype, nint address)
    {
        switch (dtype)
        {
            case DType.F32:
                var f = new Span<float>((void*)address, values.Length);
                for (int i = 0; i < values.Length; i++)
                {
                    f[i] = (float)values[i];
                }

                break;
            case DType.BF16:
                var b = new Span<ushort>((void*)address, values.Length);
                for (int i = 0; i < values.Length; i++)
                {
                    b[i] = SingleToBFloat16((float)values[i]);
                }

                break;
            default:
                throw new NotSupportedException("Tiny weights are F32 or BF16, got " + dtype);
        }
    }

    /// <summary>A 64-bit FNV-1a digest of the little-endian bytes of the values in <paramref name="dtype"/>, for tests that pin the generator.</summary>
    /// <param name="values">The values.</param>
    /// <param name="dtype">F32 or BF16.</param>
    /// <returns>The digest.</returns>
    public static ulong Digest(ReadOnlySpan<double> values, DType dtype)
    {
        ulong h = FnvOffset;
        Span<byte> bytes = stackalloc byte[4];
        foreach (double v in values)
        {
            int n;
            if (dtype == DType.F32)
            {
                BinaryPrimitives.WriteInt32LittleEndian(bytes, BitConverter.SingleToInt32Bits((float)v));
                n = 4;
            }
            else if (dtype == DType.BF16)
            {
                BinaryPrimitives.WriteUInt16LittleEndian(bytes, SingleToBFloat16((float)v));
                n = 2;
            }
            else
            {
                throw new NotSupportedException("Tiny weights are F32 or BF16, got " + dtype);
            }

            for (int i = 0; i < n; i++)
            {
                h = (h ^ bytes[i]) * FnvPrime;
            }
        }

        return h;
    }

    private static ulong NextUInt64(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static ushort SingleToBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value))
        {
            return (ushort)((bits >> 16) | 0x40);
        }

        // Round to nearest even on the 16 dropped bits.
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }

    private static float BFloat16ToSingle(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);
}
