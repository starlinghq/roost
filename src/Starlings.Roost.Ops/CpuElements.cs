using System.Runtime.CompilerServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops;

/// <summary>
/// Element access and scalar math for <see cref="CpuOps"/>: every element is read as a double and every result is
/// rounded once, to nearest even, into the output element type. Nothing here allocates managed memory.
/// </summary>
internal static unsafe class CpuElements
{
    /// <summary>Bytes of one element; packed 4-bit and 8-bit float types are not addressable by element and throw.</summary>
    internal static int SizeOf(DType t) => t switch
    {
        DType.F64 or DType.I64 or DType.U64 => 8,
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F16 or DType.BF16 or DType.I16 => 2,
        DType.I8 or DType.U8 => 1,
        _ => throw new NotSupportedException($"CpuOps does not address {t} elements."),
    };

    /// <summary>Bytes in a span; an untyped span counts its length as bytes.</summary>
    internal static long ByteLength(in DeviceSpan s) =>
        s.Dtype == DType.Undefined ? checked((long)s.Length) : checked((long)s.Length * SizeOf(s.Dtype));

    /// <summary>Reads element <paramref name="i"/> of type <paramref name="t"/> at <paramref name="addr"/>, widened to double.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Load(nint addr, DType t, long i)
    {
        switch (t)
        {
            case DType.F32: return ((float*)addr)[i];
            case DType.F64: return ((double*)addr)[i];
            case DType.BF16: return BitConverter.Int32BitsToSingle(((ushort*)addr)[i] << 16);
            case DType.F16: return (double)BitConverter.UInt16BitsToHalf(((ushort*)addr)[i]);
            case DType.I8: return ((sbyte*)addr)[i];
            case DType.U8: return ((byte*)addr)[i];
            case DType.I16: return ((short*)addr)[i];
            case DType.I32: return ((int*)addr)[i];
            case DType.U32: return ((uint*)addr)[i];
            case DType.I64: return ((long*)addr)[i];
            case DType.U64: return ((ulong*)addr)[i];
            default: throw new NotSupportedException($"CpuOps cannot load {t}.");
        }
    }

    /// <summary>Reads an integer element exactly (no trip through double).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static long LoadInt(nint addr, DType t, long i) => t switch
    {
        DType.I32 => ((int*)addr)[i],
        DType.U32 => ((uint*)addr)[i],
        DType.I64 => ((long*)addr)[i],
        DType.U64 => (long)((ulong*)addr)[i],
        DType.I16 => ((short*)addr)[i],
        DType.I8 => ((sbyte*)addr)[i],
        DType.U8 => ((byte*)addr)[i],
        _ => throw new NotSupportedException($"CpuOps cannot load {t} as an index."),
    };

    /// <summary>Rounds <paramref name="v"/> to nearest even into element <paramref name="i"/> of type <paramref name="t"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Store(nint addr, DType t, long i, double v)
    {
        switch (t)
        {
            case DType.F32: ((float*)addr)[i] = (float)v; break;
            case DType.F64: ((double*)addr)[i] = v; break;
            case DType.BF16: ((ushort*)addr)[i] = ToBFloat16(v); break;
            case DType.F16: ((ushort*)addr)[i] = BitConverter.HalfToUInt16Bits((Half)v); break;
            default: throw new NotSupportedException($"CpuOps cannot store {t}.");
        }
    }

    /// <summary>
    /// double to bfloat16, rounded to nearest even once. The double is first rounded to float with round-to-odd so
    /// the second rounding cannot double-round; NaN stays NaN and overflow gives infinity.
    /// </summary>
    internal static ushort ToBFloat16(double v)
    {
        if (double.IsNaN(v))
        {
            return 0x7FC0;
        }

        if (double.IsInfinity(v))
        {
            return v > 0 ? (ushort)0x7F80 : (ushort)0xFF80;
        }

        float f = (float)v;
        if (!float.IsInfinity(f) && (double)f != v)
        {
            int bits = BitConverter.SingleToInt32Bits(f);
            if ((bits & 1) == 0)
            {
                // Round to odd: take the neighbour of f on the side of v whose last bit is 1.
                bits += Math.Abs((double)f) < Math.Abs(v) ? 1 : -1;
                f = BitConverter.Int32BitsToSingle(bits);
            }
        }

        uint u = BitConverter.SingleToUInt32Bits(f);
        if (float.IsInfinity(f))
        {
            return (ushort)(u >> 16);
        }

        u += 0x7FFFu + ((u >> 16) & 1u);
        return (ushort)(u >> 16);
    }

    /// <summary>Error function in double precision (relative error near 1e-15).</summary>
    internal static double Erf(double x)
    {
        double ax = Math.Abs(x);
        if (ax >= 6.0)
        {
            return Math.CopySign(1.0, x);
        }

        double r = ax < 3.0 ? ErfSeries(ax) : 1.0 - ErfcLarge(ax);
        return Math.CopySign(r, x);
    }

    /// <summary>Complementary error function in double precision, accurate in the tails.</summary>
    internal static double Erfc(double x)
    {
        if (x < 0)
        {
            return 2.0 - Erfc(-x);
        }

        if (x >= 27.0)
        {
            return 0.0;
        }

        return x < 3.0 ? 1.0 - ErfSeries(x) : ErfcLarge(x);
    }

    // erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (1 * 3 * ... * (2n+1)); all terms positive.
    private static double ErfSeries(double x)
    {
        double x2 = x * x;
        double term = x;
        double sum = x;
        for (int n = 1; n < 200; n++)
        {
            term *= 2.0 * x2 / ((2 * n) + 1);
            sum += term;
            if (term < sum * 1e-17)
            {
                break;
            }
        }

        return 2.0 / Math.Sqrt(Math.PI) * Math.Exp(-x2) * sum;
    }

    // erfc(x) = exp(-x^2) / sqrt(pi) / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...)))), evaluated bottom-up.
    private static double ErfcLarge(double x)
    {
        double t = x;
        for (int n = 120; n >= 1; n--)
        {
            t = x + (0.5 * n / t);
        }

        return Math.Exp(-x * x) / Math.Sqrt(Math.PI) / t;
    }

    /// <summary><c>x * sigmoid(x)</c>.</summary>
    internal static double Silu(double x) => x / (1.0 + Math.Exp(-x));
}
