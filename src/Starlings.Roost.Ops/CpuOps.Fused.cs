using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

// The optional fused ops on the CPU reference (R-071). Each is its unfused parts called in order on this backend, so a
// fused graph and the unfused graph give bit-identical values on CpuOps: the reference defines what a fused kernel
// must compute, it does not approximate it. AddRmsNormQuant and SiluMulQuant need Quantize, which CpuOps does not
// implement yet (reserved for M2), so their Supports flags stay false.
public readonly partial struct CpuOps
{
    /// <inheritdoc cref="IRoostOps.SupportsAddRmsNorm"/>
    public static bool SupportsAddRmsNorm => true;

    /// <inheritdoc cref="IRoostOps.SupportsQkvQkNormRopeCacheWrite"/>
    public static bool SupportsQkvQkNormRopeCacheWrite => true;

    /// <inheritdoc cref="IRoostOps.AddRmsNorm"/>
    /// <remarks>
    /// The unfused parts in order: <see cref="Residual"/> with A = Residual, B = Input and Output = Residual (the sum
    /// is rounded once into DType; the generated fallback passes the addends the other way round, which gives the same
    /// value because the sum is commutative), then <see cref="RmsNorm"/> with Input = Residual and Output = Input.
    /// Input and Residual must not overlap.
    /// </remarks>
    public static void AddRmsNorm(in AddRmsNormArgs a)
    {
        long n = (long)a.Rows * a.Dim;
        Need(a.Input, n, nameof(a.Input));
        Need(a.Residual, n, nameof(a.Residual));
        Need(a.Weight, a.Dim, nameof(a.Weight));
        Residual(new ResidualArgs { A = a.Residual, B = a.Input, Output = a.Residual, Elements = n, DType = a.DType });
        RmsNorm(new RmsNormArgs { Input = a.Residual, Weight = a.Weight, Output = a.Input, Rows = a.Rows, Dim = a.Dim, Eps = a.Eps, DType = a.DType });
    }

    /// <inheritdoc cref="IRoostOps.QkvQkNormRopeCacheWrite"/>
    /// <remarks>
    /// <para>
    /// Each row of Qkv is <c>[q (QHeads * HeadDim) | k (KvHeads * HeadDim) | v (KvHeads * HeadDim)]</c> in DType. The
    /// unfused parts in order: q is copied into Q and k and v into scratch of DType, then <see cref="QkNorm"/> (Eps) of
    /// Q and k, <see cref="Rope"/> of Q and k, and <see cref="CacheWrite"/> of k and v.
    /// </para>
    /// <para>
    /// The argument struct (C3 v0) names no rotary width, rope kind, pair layout, KV layout or KV scales, so those parts
    /// run with: full rotary width (RotaryDim 0), <see cref="RopeKind.Default"/>, NeoX halves (not interleaved),
    /// <see cref="TensorLayout.PagedNhd"/> pools and KScale = VScale = 1. A graph whose unfused nodes use anything else
    /// is not fused. The rotated k and the v rows exist only in the cache afterwards; there is no K or V output.
    /// </para>
    /// <para>Scratch for k and v scales with Tokens, so it is native memory freed before the call returns.</para>
    /// </remarks>
    public static unsafe void QkvQkNormRopeCacheWrite(in QkvQkNormRopeCacheWriteArgs a)
    {
        int hd = a.HeadDim;
        long qWidth = (long)a.QHeads * hd;
        long kvWidth = (long)a.KvHeads * hd;
        long row = qWidth + (2 * kvWidth);
        Need(a.Qkv, a.Tokens * row, nameof(a.Qkv));
        Need(a.Q, a.Tokens * qWidth, nameof(a.Q));
        int es = SizeOf(a.DType);
        long kvElements = Math.Max(1, a.Tokens * kvWidth);
        nint k = (nint)NativeMemory.Alloc((nuint)kvElements, (nuint)es);
        nint v = (nint)NativeMemory.Alloc((nuint)kvElements, (nuint)es);
        try
        {
            for (long t = 0; t < a.Tokens; t++)
            {
                nint src = a.Qkv.Address + (nint)(t * row * es);
                CopyBytes(src, a.Q.Address + (nint)(t * qWidth * es), qWidth * es);
                CopyBytes(src + (nint)(qWidth * es), k + (nint)(t * kvWidth * es), kvWidth * es);
                CopyBytes(src + (nint)((qWidth + kvWidth) * es), v + (nint)(t * kvWidth * es), kvWidth * es);
            }

            var kSpan = new DeviceSpan { Address = k, Length = (ulong)kvElements, Dtype = a.DType };
            var vSpan = new DeviceSpan { Address = v, Length = (ulong)kvElements, Dtype = a.DType };
            QkNorm(new QkNormArgs
            {
                Q = a.Q, K = kSpan, QWeight = a.QWeight, KWeight = a.KWeight,
                Tokens = a.Tokens, QHeads = a.QHeads, KvHeads = a.KvHeads, HeadDim = hd, Eps = a.Eps, DType = a.DType,
            });
            Rope(new RopeArgs
            {
                Q = a.Q, K = kSpan, Positions = a.Positions, Tokens = a.Tokens, QHeads = a.QHeads, KvHeads = a.KvHeads, HeadDim = hd,
                RotaryDim = 0, Kind = RopeKind.Default, Interleaved = false, Theta = a.Theta, DType = a.DType,
            });
            CacheWrite(new CacheWriteArgs
            {
                K = kSpan, V = vSpan, KCache = a.KCache, VCache = a.VCache, SlotMapping = a.SlotMapping,
                Tokens = a.Tokens, KvHeads = a.KvHeads, HeadDim = hd, BlockSize = a.BlockSize, KvLayout = TensorLayout.PagedNhd,
                KScale = 1f, VScale = 1f, KvDType = a.KvDType,
            });
        }
        finally
        {
            NativeMemory.Free((void*)k);
            NativeMemory.Free((void*)v);
        }
    }
}
