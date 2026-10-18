using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>Unmanaged host memory that tests hand to CpuOps as a DeviceSpan.</summary>
internal sealed class Mem : IDisposable
{
    private readonly int _elementSize;

    private Mem(DType dtype, int count, int elementSize)
    {
        Dtype = dtype;
        Count = count;
        _elementSize = elementSize;
        Address = Marshal.AllocHGlobal(Math.Max(1, count * elementSize));
        for (int i = 0; i < Math.Max(1, count * elementSize); i++)
        {
            Marshal.WriteByte(Address, i, 0);
        }
    }

    public nint Address { get; }

    public DType Dtype { get; }

    public int Count { get; }

    public DeviceSpan Span => new() { Address = Address, Length = (ulong)Count, Dtype = Dtype };

    public static Mem Zeros(DType dtype, int count) => new(dtype, count, dtype switch
    {
        DType.F64 or DType.I64 or DType.U64 => 8,
        DType.F32 or DType.I32 or DType.U32 => 4,
        DType.F16 or DType.BF16 or DType.I16 => 2,
        _ => 1,
    });

    public static Mem F64(params double[] v)
    {
        var m = Zeros(DType.F64, v.Length);
        Marshal.Copy(v, 0, m.Address, v.Length);
        return m;
    }

    public static Mem F32(params float[] v)
    {
        var m = Zeros(DType.F32, v.Length);
        Marshal.Copy(v, 0, m.Address, v.Length);
        return m;
    }

    public static Mem F32(IEnumerable<double> v) => F32(v.Select(x => (float)x).ToArray());

    /// <summary>bfloat16 by truncating the float; callers pass values that are exact in bfloat16.</summary>
    public static Mem Bf16(params float[] v)
    {
        var m = Zeros(DType.BF16, v.Length);
        Marshal.Copy(v.Select(x => unchecked((short)(BitConverter.SingleToInt32Bits(x) >> 16))).ToArray(), 0, m.Address, v.Length);
        return m;
    }

    public static Mem I32(params int[] v)
    {
        var m = Zeros(DType.I32, v.Length);
        Marshal.Copy(v, 0, m.Address, v.Length);
        return m;
    }

    public static Mem I64(params long[] v)
    {
        var m = Zeros(DType.I64, v.Length);
        Marshal.Copy(v, 0, m.Address, v.Length);
        return m;
    }

    public static Mem U64(params ulong[] v)
    {
        var m = Zeros(DType.U64, v.Length);
        Marshal.Copy(v.Select(x => unchecked((long)x)).ToArray(), 0, m.Address, v.Length);
        return m;
    }

    public static Mem U32(params uint[] v)
    {
        var m = Zeros(DType.U32, v.Length);
        Marshal.Copy(v.Select(x => unchecked((int)x)).ToArray(), 0, m.Address, v.Length);
        return m;
    }

    public double[] ReadF64()
    {
        var r = new double[Count];
        Marshal.Copy(Address, r, 0, Count);
        return r;
    }

    public float[] ReadF32()
    {
        var r = new float[Count];
        Marshal.Copy(Address, r, 0, Count);
        return r;
    }

    public int[] ReadI32()
    {
        var r = new int[Count];
        Marshal.Copy(Address, r, 0, Count);
        return r;
    }

    public ushort[] ReadU16()
    {
        var s = new short[Count];
        Marshal.Copy(Address, s, 0, Count);
        return s.Select(x => unchecked((ushort)x)).ToArray();
    }

    public float[] ReadBf16() => ReadU16().Select(x => BitConverter.Int32BitsToSingle(x << 16)).ToArray();

    public byte[] ReadBytes()
    {
        var r = new byte[Count * _elementSize];
        Marshal.Copy(Address, r, 0, r.Length);
        return r;
    }

    public void Dispose() => Marshal.FreeHGlobal(Address);
}

/// <summary>Disposes a set of buffers together.</summary>
internal sealed class Bag : IDisposable
{
    private readonly List<Mem> _all = [];

    public Mem Add(Mem m)
    {
        _all.Add(m);
        return m;
    }

    public void Dispose()
    {
        foreach (var m in _all)
        {
            m.Dispose();
        }
    }
}

/// <summary>Element-wise comparison with an absolute tolerance.</summary>
internal static class Near
{
    public static void Eq(double[] expected, double[] actual, double tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"[{i}] expected {expected[i]:R} actual {actual[i]:R}");
        }
    }
}
