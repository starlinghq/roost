using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>Seeded generator: uniform and normal doubles.</summary>
internal sealed class Rng
{
    private readonly Random _r;

    public Rng(int seed) => _r = new Random(seed);

    public double Uniform() => _r.NextDouble();

    public int Int(int lo, int hiExclusive) => _r.Next(lo, hiExclusive);

    public double Normal()
    {
        double u1 = 1.0 - _r.NextDouble();
        double u2 = _r.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal static class Elem
{
    public static int Size(DType t) => t switch
    {
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F16 or DType.BF16 or DType.I16 => 2,
        DType.F64 or DType.I64 or DType.U64 => 8,
        _ => 1,
    };

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

    public static float FromBf16(ushort b) => BitConverter.UInt32BitsToSingle((uint)b << 16);

    public static ushort ToF16(float f) => BitConverter.HalfToUInt16Bits((Half)f);

    public static float FromF16(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);
}

/// <summary>A tensor with a host copy (aligned unmanaged memory, the address CpuOps reads) and a device copy.</summary>
internal sealed unsafe class Buf : IDisposable
{
    private nint _dev;

    public Buf(DType dtype, long count)
    {
        Dtype = dtype;
        Count = count;
        Bytes = count * Elem.Size(dtype);
        Host = (nint)NativeMemory.AlignedAlloc((nuint)Math.Max(Bytes, 256), 256);
        NativeMemory.Clear((void*)Host, (nuint)Math.Max(Bytes, 256));
    }

    public DType Dtype { get; }

    public long Count { get; }

    public long Bytes { get; }

    public nint Host { get; }

    public nint Dev
    {
        get
        {
            if (_dev == 0)
            {
                _dev = Cuda.Alloc(Bytes);
            }

            return _dev;
        }
    }

    public DeviceSpan HostSpan => new() { Address = Host, Length = (ulong)Count, Dtype = Dtype };

    public DeviceSpan DevSpan => new() { Address = Dev, Length = (ulong)Count, Dtype = Dtype };

    public static DeviceSpan Null => default;

    public Span<T> As<T>() where T : unmanaged => new((void*)Host, (int)(Bytes / sizeof(T)));

    public void Up() => Cuda.ToDevice(Dev, Host, Bytes);

    public void Down() => Cuda.ToHost(Host, Dev, Bytes);

    public void FillDevice(int value) => Cuda.Fill(Dev, value, Bytes);

    /// <summary>Copies the host bytes of this buffer into <paramref name="other"/> (same size).</summary>
    public void CopyHostTo(Buf other) => Buffer.MemoryCopy((void*)Host, (void*)other.Host, other.Bytes, Bytes);

    public double Get(long i) => Dtype switch
    {
        DType.BF16 => Elem.FromBf16(As<ushort>()[(int)i]),
        DType.F16 => Elem.FromF16(As<ushort>()[(int)i]),
        DType.F32 => As<float>()[(int)i],
        DType.I32 => As<int>()[(int)i],
        DType.I64 => As<long>()[(int)i],
        DType.F64 => As<double>()[(int)i],
        _ => As<byte>()[(int)i],
    };

    public void Set(long i, double v)
    {
        switch (Dtype)
        {
            case DType.BF16: As<ushort>()[(int)i] = Elem.ToBf16((float)v); break;
            case DType.F16: As<ushort>()[(int)i] = Elem.ToF16((float)v); break;
            case DType.F32: As<float>()[(int)i] = (float)v; break;
            case DType.I32: As<int>()[(int)i] = (int)v; break;
            case DType.I64: As<long>()[(int)i] = (long)v; break;
            default: throw new NotSupportedException(Dtype.ToString());
        }
    }

    /// <summary>Fills with mean + std * N(0,1), rounded to the element type.</summary>
    public Buf Randn(Rng rng, double std = 1.0, double mean = 0.0)
    {
        for (long i = 0; i < Count; i++)
        {
            Set(i, mean + (std * rng.Normal()));
        }

        return this;
    }

    public Buf FromInts(params int[] v)
    {
        for (int i = 0; i < v.Length; i++)
        {
            As<int>()[i] = v[i];
        }

        return this;
    }

    public static Buf Ints(params int[] v) => new Buf(DType.I32, v.Length).FromInts(v);

    public static Buf Longs(params long[] v)
    {
        var b = new Buf(DType.I64, v.Length);
        for (int i = 0; i < v.Length; i++)
        {
            b.As<long>()[i] = v[i];
        }

        return b;
    }

    public Buf Clone()
    {
        var c = new Buf(Dtype, Count);
        CopyHostTo(c);
        return c;
    }

    public void Dispose()
    {
        NativeMemory.AlignedFree((void*)Host);
        if (_dev != 0)
        {
            Cuda.Release(_dev);
            _dev = 0;
        }
    }
}
