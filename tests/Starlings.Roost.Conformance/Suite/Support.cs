using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// Deterministic generator (xoshiro256** seeded through SplitMix64). <see cref="Random"/> is not used: its sequence for a
/// given seed is not guaranteed across .NET releases, and a conformance case must reproduce from its recorded seed.
/// </summary>
public sealed class Rng
{
    private ulong _s0, _s1, _s2, _s3;

    /// <summary>Creates a generator from a seed.</summary>
    /// <param name="seed">Any 64-bit seed.</param>
    public Rng(ulong seed)
    {
        ulong x = seed;
        _s0 = SplitMix(ref x);
        _s1 = SplitMix(ref x);
        _s2 = SplitMix(ref x);
        _s3 = SplitMix(ref x);
    }

    /// <summary>Derives a seed from a text label, stable across processes.</summary>
    /// <param name="label">The label.</param>
    /// <param name="salt">A number mixed in.</param>
    /// <returns>A 64-bit seed (FNV-1a over the UTF-8 bytes).</returns>
    public static ulong SeedOf(string label, ulong salt = 0)
    {
        ulong h = 14695981039346656037UL ^ salt;
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes(label))
        {
            h = (h ^ b) * 1099511628211UL;
        }

        return h;
    }

    /// <summary>Next 64 random bits.</summary>
    /// <returns>The bits.</returns>
    public ulong NextU64()
    {
        ulong result = Rotl(_s1 * 5, 7) * 9;
        ulong t = _s1 << 17;
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = Rotl(_s3, 45);
        return result;
    }

    /// <summary>Uniform double in [0, 1).</summary>
    /// <returns>The value.</returns>
    public double Uniform() => (NextU64() >> 11) * (1.0 / (1UL << 53));

    /// <summary>Uniform integer in [lo, hi).</summary>
    /// <param name="lo">Inclusive lower bound.</param>
    /// <param name="hi">Exclusive upper bound.</param>
    /// <returns>The value.</returns>
    public int Int(int lo, int hi) => lo + (int)(NextU64() % (ulong)(hi - lo));

    /// <summary>Standard normal (Box-Muller).</summary>
    /// <returns>The value.</returns>
    public double Normal()
    {
        double u1 = 1.0 - Uniform();
        double u2 = Uniform();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>One of the given values.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="values">The values.</param>
    /// <returns>A uniformly chosen value.</returns>
    public T Pick<T>(params T[] values) => values[Int(0, values.Length)];

    private static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));

    private static ulong SplitMix(ref ulong x)
    {
        ulong z = x += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

/// <summary>Element sizes and the 16-bit float conversions the suite needs.</summary>
public static class Elem
{
    /// <summary>Bytes of one element.</summary>
    /// <param name="t">The type.</param>
    /// <returns>The size.</returns>
    public static int Size(DType t) => t switch
    {
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F16 or DType.BF16 or DType.I16 => 2,
        DType.F64 or DType.I64 or DType.U64 => 8,
        _ => 1,
    };

    /// <summary>Rounds a float to bfloat16, to nearest even.</summary>
    /// <param name="f">The value.</param>
    /// <returns>The bits.</returns>
    public static ushort ToBf16(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        if ((u & 0x7fffffffu) > 0x7f800000u)
        {
            return (ushort)((u >> 16) | 0x40);
        }

        u += 0x7fffu + ((u >> 16) & 1u);
        return (ushort)(u >> 16);
    }

    /// <summary>Widens bfloat16 bits.</summary>
    /// <param name="b">The bits.</param>
    /// <returns>The value.</returns>
    public static float FromBf16(ushort b) => BitConverter.UInt32BitsToSingle((uint)b << 16);

    /// <summary>Rounds a float to IEEE half, to nearest even.</summary>
    /// <param name="f">The value.</param>
    /// <returns>The bits.</returns>
    public static ushort ToF16(float f) => BitConverter.HalfToUInt16Bits((Half)f);

    /// <summary>Widens IEEE half bits.</summary>
    /// <param name="h">The bits.</param>
    /// <returns>The value.</returns>
    public static float FromF16(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);

    /// <summary>Mask bit of a dtype in a constraints mask.</summary>
    /// <param name="t">The type.</param>
    /// <returns>1 shifted by the dtype number.</returns>
    public static ulong Bit(DType t) => 1UL << (int)t;
}

/// <summary>A tensor in aligned host memory: what CpuOps reads and writes, and what a device backend copies from and to.</summary>
public sealed unsafe class HostBuf : IDisposable
{
    private nint _host;

    /// <summary>Allocates a zeroed buffer.</summary>
    /// <param name="dtype">Element type.</param>
    /// <param name="count">Element count.</param>
    public HostBuf(DType dtype, long count)
    {
        Dtype = dtype;
        Count = count;
        Bytes = count * Elem.Size(dtype);
        _host = (nint)NativeMemory.AlignedAlloc((nuint)Math.Max(Bytes, 256), 256);
        NativeMemory.Clear((void*)_host, (nuint)Math.Max(Bytes, 256));
    }

    /// <summary>Element type.</summary>
    public DType Dtype { get; }

    /// <summary>Element count.</summary>
    public long Count { get; }

    /// <summary>Size in bytes.</summary>
    public long Bytes { get; }

    /// <summary>Host address (256-byte aligned).</summary>
    public nint Host => _host;

    /// <summary>The buffer as a span for CpuOps.</summary>
    public DeviceSpan Span => new() { Address = _host, Length = (ulong)Count, Dtype = Dtype };

    /// <summary>The bytes.</summary>
    /// <returns>A span over the host memory.</returns>
    public Span<byte> Raw() => new((void*)_host, (int)Bytes);

    /// <summary>The elements as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">An unmanaged type of the element's width.</typeparam>
    /// <returns>A span over the host memory.</returns>
    public Span<T> As<T>() where T : unmanaged => new((void*)_host, (int)(Bytes / sizeof(T)));

    /// <summary>Reads element <paramref name="i"/> as a double.</summary>
    /// <param name="i">The index.</param>
    /// <returns>The value.</returns>
    public double Get(long i) => Dtype switch
    {
        DType.BF16 => Elem.FromBf16(((ushort*)_host)[i]),
        DType.F16 => Elem.FromF16(((ushort*)_host)[i]),
        DType.F32 => ((float*)_host)[i],
        DType.I32 => ((int*)_host)[i],
        DType.I64 => ((long*)_host)[i],
        DType.F64 => ((double*)_host)[i],
        _ => ((byte*)_host)[i],
    };

    /// <summary>Writes element <paramref name="i"/>, rounding once to the element type.</summary>
    /// <param name="i">The index.</param>
    /// <param name="v">The value.</param>
    public void Set(long i, double v)
    {
        switch (Dtype)
        {
            case DType.BF16: ((ushort*)_host)[i] = Elem.ToBf16((float)v); break;
            case DType.F16: ((ushort*)_host)[i] = Elem.ToF16((float)v); break;
            case DType.F32: ((float*)_host)[i] = (float)v; break;
            case DType.F64: ((double*)_host)[i] = v; break;
            case DType.I32: ((int*)_host)[i] = (int)v; break;
            case DType.I64: ((long*)_host)[i] = (long)v; break;
            default: ((byte*)_host)[i] = (byte)v; break;
        }
    }

    /// <summary>Fills with mean + std * N(0,1), rounded to the element type.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="std">Standard deviation.</param>
    /// <param name="mean">Mean.</param>
    /// <returns>This buffer.</returns>
    public HostBuf Randn(Rng rng, double std = 1.0, double mean = 0.0)
    {
        for (long i = 0; i < Count; i++)
        {
            Set(i, mean + (std * rng.Normal()));
        }

        return this;
    }

    /// <summary>Fills every byte with a value.</summary>
    /// <param name="value">The byte.</param>
    /// <returns>This buffer.</returns>
    public HostBuf FillBytes(byte value)
    {
        Raw().Fill(value);
        return this;
    }

    /// <summary>A copy.</summary>
    /// <returns>A new buffer with the same contents.</returns>
    public HostBuf Clone()
    {
        var c = new HostBuf(Dtype, Count);
        Raw().CopyTo(c.Raw());
        return c;
    }

    /// <summary>An int32 buffer.</summary>
    /// <param name="v">The values.</param>
    /// <returns>The buffer.</returns>
    public static HostBuf Ints(params int[] v)
    {
        var b = new HostBuf(DType.I32, v.Length);
        v.CopyTo(b.As<int>());
        return b;
    }

    /// <summary>An int64 buffer.</summary>
    /// <param name="v">The values.</param>
    /// <returns>The buffer.</returns>
    public static HostBuf Longs(params long[] v)
    {
        var b = new HostBuf(DType.I64, v.Length);
        v.CopyTo(b.As<long>());
        return b;
    }

    /// <summary>Wraps raw little-endian bytes as a buffer of a type.</summary>
    /// <param name="dtype">The element type.</param>
    /// <param name="bytes">The bytes (length a multiple of the element size).</param>
    /// <returns>A new buffer.</returns>
    public static HostBuf FromBytes(DType dtype, ReadOnlySpan<byte> bytes)
    {
        var b = new HostBuf(dtype, bytes.Length / Elem.Size(dtype));
        bytes.CopyTo(b.Raw());
        return b;
    }

    /// <summary>Rows of <paramref name="width"/> elements out of a buffer with row stride <paramref name="stride"/>.</summary>
    /// <param name="rows">Row count.</param>
    /// <param name="width">Row width in elements.</param>
    /// <param name="stride">Elements between row starts.</param>
    /// <returns>A new, compact buffer.</returns>
    public HostBuf Compact(long rows, long width, long stride)
    {
        var c = new HostBuf(Dtype, rows * width);
        int es = Elem.Size(Dtype);
        for (long r = 0; r < rows; r++)
        {
            Buffer.MemoryCopy((void*)(_host + (nint)(r * stride * es)), (void*)(c._host + (nint)(r * width * es)), width * es, width * es);
        }

        return c;
    }

    /// <summary>Spreads a compact buffer into a strided one; padding keeps <paramref name="pad"/>.</summary>
    /// <param name="rows">Row count.</param>
    /// <param name="width">Row width in elements.</param>
    /// <param name="stride">Elements between row starts.</param>
    /// <param name="pad">Byte value for the elements between rows.</param>
    /// <returns>A new buffer of rows * stride elements.</returns>
    public HostBuf Strided(long rows, long width, long stride, byte pad = 0)
    {
        var c = new HostBuf(Dtype, rows * stride);
        c.FillBytes(pad);
        int es = Elem.Size(Dtype);
        for (long r = 0; r < rows; r++)
        {
            Buffer.MemoryCopy((void*)(_host + (nint)(r * width * es)), (void*)(c._host + (nint)(r * stride * es)), width * es, width * es);
        }

        return c;
    }

    /// <summary>Largest absolute element value.</summary>
    /// <returns>max |x|, 0 for an empty buffer.</returns>
    public double MaxAbs()
    {
        double m = 0;
        for (long i = 0; i < Count; i++)
        {
            m = Math.Max(m, Math.Abs(Get(i)));
        }

        return m;
    }

    /// <summary>Releases the memory.</summary>
    public void Dispose()
    {
        if (_host != 0)
        {
            NativeMemory.AlignedFree((void*)_host);
            _host = 0;
        }
    }
}
