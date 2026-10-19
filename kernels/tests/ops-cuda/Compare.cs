using Starlings.Roost.Abi;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>
/// Proposed tolerance of an op class: an element passes when <c>|gpu - ref| &lt;= AtolRel * rms(ref) + Rtol * |ref|</c>.
/// The reference is CpuOps in double, rounded once to the element type; the GPU rounds once from fp32 (or from bf16
/// intermediates in the attention kernels). Scaling the absolute part by the reference's own RMS keeps the criterion
/// independent of the test data's magnitude; attention passes the largest |V| of the pool instead, because its error comes from
/// rounding the softmax weights to bf16 and so scales with the values being averaged, not with the (smaller) average.
/// Exact ops use <see cref="Exact"/>. These are proposals for the
/// maintainer's protected tolerance file (R-019); nothing here edits one.
/// </summary>
internal readonly record struct Tol(string Class, double Rtol, double AtolRel, bool Exact = false)
{
    public static Tol Exact_ => new("exact", 0, 0, true);

    // One bf16 ulp is at most 2^-7 relative (8 significant bits); fp16 is 2^-10.
    public static Tol Elementwise(DType t) => t == DType.BF16 ? new("elementwise-bf16", Math.Pow(2, -6), Math.Pow(2, -12)) : new("elementwise-f16", Math.Pow(2, -9), Math.Pow(2, -15));
}

internal sealed record CompareResult(string Name, string Class, long Count, double MaxAbs, double MaxUtil, long Violations, long NonFinite, double RefRms, bool Pass)
{
    public override string ToString() =>
        $"{(Pass ? "PASS" : "FAIL")} {Name,-58} class={Class,-18} n={Count,-9} max_abs={MaxAbs:E2} ref_rms={RefRms:E2} utilisation={MaxUtil:F3} violations={Violations} nonfinite={NonFinite}";
}

internal static unsafe class Compare
{
    /// <summary>Compares <paramref name="got"/> with <paramref name="want"/> over [first, first + count) of both.</summary>
    public static CompareResult Close(string name, Buf got, Buf want, Tol tol, long first = 0, long count = -1, double scale = 0)
    {
        if (count < 0)
        {
            count = want.Count - first;
        }

        double ss = 0;
        for (long i = first; i < first + count; i++)
        {
            double r = want.Get(i);
            ss += r * r;
        }

        double rms = count > 0 ? Math.Sqrt(ss / count) : 0;
        double atol = tol.AtolRel * (scale > 0 ? scale : rms);
        double maxAbs = 0, maxUtil = 0;
        long bad = 0, nonFinite = 0;
        for (long i = first; i < first + count; i++)
        {
            double g = got.Get(i), r = want.Get(i);
            if (!double.IsFinite(g))
            {
                nonFinite++;
                continue;
            }

            double d = Math.Abs(g - r);
            double allowed = atol + (tol.Rtol * Math.Abs(r));
            maxAbs = Math.Max(maxAbs, d);
            if (tol.Exact)
            {
                if (d != 0)
                {
                    bad++;
                }

                continue;
            }

            double util = allowed > 0 ? d / allowed : (d == 0 ? 0 : double.PositiveInfinity);
            maxUtil = Math.Max(maxUtil, util);
            if (d > allowed)
            {
                bad++;
            }
        }

        return new CompareResult(name, tol.Class, count, maxAbs, maxUtil, bad, nonFinite, rms, bad == 0 && nonFinite == 0);
    }

    /// <summary>Bitwise equality of two buffers' host bytes over a byte range.</summary>
    public static bool BytesEqual(Buf a, Buf b, long offset = 0, long bytes = -1)
    {
        if (bytes < 0)
        {
            bytes = Math.Min(a.Bytes, b.Bytes) - offset;
        }

        return new ReadOnlySpan<byte>((void*)(a.Host + (nint)offset), (int)bytes).SequenceEqual(new ReadOnlySpan<byte>((void*)(b.Host + (nint)offset), (int)bytes));
    }

    public static CompareResult Bits(string name, Buf got, Buf want, long byteOffset = 0, long bytes = -1)
    {
        bool ok = BytesEqual(got, want, byteOffset, bytes);
        return new CompareResult(name, "exact", bytes < 0 ? want.Bytes : bytes, ok ? 0 : 1, ok ? 0 : 1, ok ? 0 : 1, 0, 0, ok);
    }
}
