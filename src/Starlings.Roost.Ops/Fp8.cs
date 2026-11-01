using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops;

/// <summary>
/// The two 8-bit float formats of the OCP 8-bit floating point specification (OFP8, rev. 1.0), as CUDA's
/// <c>__nv_fp8_e4m3</c> and <c>__nv_fp8_e5m2</c> store them.
/// </summary>
/// <remarks>
/// <para>
/// E4M3 (<see cref="DType.F8E4M3"/>, "e4m3fn"): bias 7, no infinities, NaN is <c>S.1111.111</c> only, largest finite 448,
/// smallest subnormal 2^-9. E5M2 (<see cref="DType.F8E5M2"/>): bias 15, IEEE-style infinities (<c>S.11111.00</c>) and NaNs
/// (<c>S.11111.xx</c>, xx not 0), largest finite 57344, smallest subnormal 2^-16.
/// </para>
/// <para>
/// Encoding rounds the exact double once, to nearest with ties to the even code (the code whose last mantissa bit is 0),
/// subnormals included. NaN encodes as <c>0x7F</c> in both formats (sign dropped). With <c>saturate</c> (CUDA's
/// <c>__NV_SATFINITE</c>, what <c>Store</c> and every quantising op use), a finite value whose rounded magnitude exceeds the
/// largest finite value, and an infinity, give the largest finite value of the same sign. Without it (OFP8 overflow rules),
/// they give NaN in E4M3 and infinity in E5M2, keeping the sign (<c>S.1111.111</c>, <c>S.11111.00</c>). The rounded
/// magnitude is judged as if the exponent range went on: E4M3's tie at 464 rounds to 448 (even code), E5M2's tie at 61440
/// rounds up and overflows. Negative zero keeps its sign. The non-saturating mode agreed with ml_dtypes 0.5.1
/// (<c>float8_e4m3fn</c>, <c>float8_e5m2</c>) on every binary16 input in a one-off check during R-077 (NaN sign aside).
/// </para>
/// </remarks>
internal static class Fp8
{
    /// <summary>Largest finite E4M3 value.</summary>
    internal const double E4M3Max = 448.0;

    /// <summary>Largest finite E5M2 value.</summary>
    internal const double E5M2Max = 57344.0;

    /// <summary>The NaN code both formats produce.</summary>
    internal const byte NaN = 0x7F;

    /// <summary>Largest finite value of an FP8 type.</summary>
    internal static double MaxOf(DType t) => t switch
    {
        DType.F8E4M3 => E4M3Max,
        DType.F8E5M2 => E5M2Max,
        _ => throw new NotSupportedException($"{t} is not an FP8 type."),
    };

    /// <summary>Whether <paramref name="t"/> is one of the two FP8 types.</summary>
    internal static bool Is(DType t) => t is DType.F8E4M3 or DType.F8E5M2;

    /// <summary>The exact value of an FP8 code.</summary>
    internal static double Decode(byte code, DType t)
    {
        bool e4m3 = t == DType.F8E4M3;
        int mBits = e4m3 ? 3 : 2;
        int bias = e4m3 ? 7 : 15;
        int maxExp = e4m3 ? 15 : 31;
        int e = (code & 0x7F) >> mBits;
        int m = code & ((1 << mBits) - 1);
        double sign = (code & 0x80) != 0 ? -1.0 : 1.0;
        if (e4m3 ? (code & 0x7F) == 0x7F : e == maxExp)
        {
            return !e4m3 && m == 0 ? sign * double.PositiveInfinity : double.NaN;
        }

        return e == 0
            ? sign * Math.ScaleB(m, 1 - bias - mBits)
            : sign * Math.ScaleB((1 << mBits) + m, e - bias - mBits);
    }

    /// <summary>Rounds <paramref name="v"/> to nearest even into an FP8 code of type <paramref name="t"/> (see the type remarks).</summary>
    internal static byte Encode(double v, DType t, bool saturate)
    {
        bool e4m3 = t == DType.F8E4M3;
        if (!e4m3 && t != DType.F8E5M2)
        {
            throw new NotSupportedException($"{t} is not an FP8 type.");
        }

        if (double.IsNaN(v))
        {
            return NaN;
        }

        int mBits = e4m3 ? 3 : 2;
        int bias = e4m3 ? 7 : 15;
        int maxCode = e4m3 ? 0x7E : 0x7B;
        int sign = double.IsNegative(v) ? 0x80 : 0;
        double a = Math.Abs(v);
        int code;
        if (double.IsPositiveInfinity(a) || a >= Math.ScaleB(1.0, 18))
        {
            // Far beyond both formats' ranges (2^18 > 57344 * 4): overflow whatever the rounding.
            code = maxCode + 1;
        }
        else if (a == 0)
        {
            code = 0;
        }
        else
        {
            // Subnormals share the exponent of the smallest normal, so the quantum is never finer than 2^(1 - bias - mBits).
            int e = Math.Max(Math.ILogB(a), 1 - bias);
            double q = Math.Round(Math.ScaleB(a, mBits - e), MidpointRounding.ToEven);

            // q is in [0, 2^(mBits+1)]; the code's bit pattern is monotonic in the value, so it is the biased exponent
            // times 2^mBits plus the significand less the hidden bit. A carry to 2^(mBits+1) lands on the next binade's code.
            int biased = e + bias;
            code = q < (1 << mBits) ? (int)q : ((biased - 1) << mBits) + (int)q;
        }

        if (code > maxCode)
        {
            if (saturate)
            {
                return (byte)(sign | maxCode);
            }

            return (byte)(sign | (e4m3 ? NaN : 0x7C));
        }

        return (byte)(sign | code);
    }
}
