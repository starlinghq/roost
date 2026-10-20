using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;

namespace Roost.HybridConformance;

/// <summary>The few CUDA runtime calls the harness needs, resolved from ROOST_CUDART (default /opt/cuda/lib64/libcudart.so).</summary>
internal static unsafe partial class Cuda
{
    private const string Lib = "roost-cudart";

    static Cuda()
    {
        NativeLibrary.SetDllImportResolver(typeof(Cuda).Assembly, (name, _, _) =>
            name == Lib
                ? NativeLibrary.Load(Environment.GetEnvironmentVariable("ROOST_CUDART") ?? "/opt/cuda/lib64/libcudart.so")
                : IntPtr.Zero);
    }

    [LibraryImport(Lib)] private static partial int cudaMalloc(out nint ptr, nuint bytes);
    [LibraryImport(Lib)] private static partial int cudaFree(nint ptr);
    [LibraryImport(Lib)] private static partial int cudaMallocHost(out nint ptr, nuint bytes);
    [LibraryImport(Lib)] private static partial int cudaFreeHost(nint ptr);
    [LibraryImport(Lib)] private static partial int cudaMemcpy(nint dst, nint src, nuint bytes, int kind);
    [LibraryImport(Lib)] private static partial int cudaMemset(nint ptr, int value, nuint bytes);
    [LibraryImport(Lib)] private static partial int cudaStreamCreate(out nint stream);
    [LibraryImport(Lib)] private static partial int cudaStreamDestroy(nint stream);
    [LibraryImport(Lib)] private static partial int cudaStreamSynchronize(nint stream);
    [LibraryImport(Lib)] private static partial int cudaSetDevice(int device);
    [LibraryImport(Lib)] private static partial int cudaDeviceSynchronize();
    [LibraryImport(Lib)] private static partial int cudaGetLastError();
    [LibraryImport(Lib)] private static partial int cudaStreamBeginCapture(nint stream, int mode);
    [LibraryImport(Lib)] private static partial int cudaStreamEndCapture(nint stream, out nint graph);
    [LibraryImport(Lib)] private static partial int cudaGraphInstantiate(out nint exec, nint graph, ulong flags);
    [LibraryImport(Lib)] private static partial int cudaGraphLaunch(nint exec, nint stream);
    [LibraryImport(Lib)] private static partial int cudaGraphExecDestroy(nint exec);
    [LibraryImport(Lib)] private static partial int cudaGraphDestroy(nint graph);
    [LibraryImport(Lib)] private static partial nint cudaGetErrorString(int error);

    private static void Check(int rc, string what)
    {
        if (rc != 0)
        {
            throw new InvalidOperationException($"{what}: CUDA error {rc} ({Marshal.PtrToStringAnsi(cudaGetErrorString(rc))})");
        }
    }

    public static void SetDevice(int device) => Check(cudaSetDevice(device), "cudaSetDevice");

    public static nint Malloc(long bytes)
    {
        Check(cudaMalloc(out var p, (nuint)Math.Max(bytes, 16)), "cudaMalloc");
        return p;
    }

    public static void Free(nint p) => Check(cudaFree(p), "cudaFree");

    public static nint MallocHost(long bytes)
    {
        Check(cudaMallocHost(out var p, (nuint)Math.Max(bytes, 16)), "cudaMallocHost");
        return p;
    }

    public static void FreeHost(nint p) => Check(cudaFreeHost(p), "cudaFreeHost");

    public static void Upload(nint device, nint host, long bytes) => Check(cudaMemcpy(device, host, (nuint)bytes, 1), "memcpy H2D");

    public static void Download(nint host, nint device, long bytes) => Check(cudaMemcpy(host, device, (nuint)bytes, 2), "memcpy D2H");

    public static void Fill(nint device, byte value, long bytes) => Check(cudaMemset(device, value, (nuint)bytes), "cudaMemset");

    public static nint CreateStream()
    {
        Check(cudaStreamCreate(out var s), "cudaStreamCreate");
        return s;
    }

    public static void DestroyStream(nint s) => Check(cudaStreamDestroy(s), "cudaStreamDestroy");

    public static void Sync(nint stream) => Check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /// <summary>Captures what <paramref name="enqueue"/> queues on the stream into a CUDA graph, instantiates it, launches it once
    /// and waits. Nothing runs during capture, so the op's result appears only if the capture holds the whole op. Returns the
    /// status the enqueue produced.</summary>
    public static int RunGraphed(nint stream, Func<int> enqueue)
    {
        Check(cudaStreamBeginCapture(stream, 1), "cudaStreamBeginCapture"); // cudaStreamCaptureModeThreadLocal
        var rc = enqueue();
        var end = cudaStreamEndCapture(stream, out var graph);
        if (end != 0 || graph == 0)
        {
            throw new InvalidOperationException($"cudaStreamEndCapture failed ({end}): the op's launches were not capturable");
        }

        var inst = cudaGraphInstantiate(out var exec, graph, 0);
        var launch = inst == 0 ? cudaGraphLaunch(exec, stream) : inst;
        var sync = launch == 0 ? cudaStreamSynchronize(stream) : launch;
        var freed = (inst == 0 ? cudaGraphExecDestroy(exec) : 0) | cudaGraphDestroy(graph);
        Check(sync, "graph instantiate, launch and sync");
        Check(freed, "graph destroy");
        return rc;
    }

    public static void SyncDevice() => Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

/// <summary>A buffer that exists twice: host memory (what CpuOps reads and writes) and a device copy (what the library gets).</summary>
internal sealed unsafe class Buf : IDisposable
{
    public Buf(DType dtype, long elements)
    {
        DType = dtype;
        Elements = elements;
        Bytes = elements * Size(dtype);
        Host = (nint)NativeMemory.AllocZeroed((nuint)Math.Max(Bytes, 16));
        Device = Cuda.Malloc(Bytes);
    }

    public DType DType { get; }

    public long Elements { get; }

    public long Bytes { get; }

    public nint Host { get; }

    public nint Device { get; }

    public static int Size(DType t) => t switch
    {
        DType.F32 or DType.I32 => 4,
        DType.BF16 or DType.F16 => 2,
        DType.F64 or DType.I64 => 8,
        _ => throw new ArgumentOutOfRangeException(nameof(t)),
    };

    public static Buf FromF32(float[] v)
    {
        var b = new Buf(DType.F32, v.Length);
        v.AsSpan().CopyTo(new Span<float>((void*)b.Host, v.Length));
        return b;
    }

    public static Buf FromBf16(float[] v)
    {
        var b = new Buf(DType.BF16, v.Length);
        var s = new Span<ushort>((void*)b.Host, v.Length);
        for (var i = 0; i < v.Length; i++)
        {
            s[i] = Bf16.From(v[i]);
        }

        return b;
    }

    public static Buf FromI32(params int[] v)
    {
        var b = new Buf(DType.I32, v.Length);
        v.AsSpan().CopyTo(new Span<int>((void*)b.Host, v.Length));
        return b;
    }

    /// <summary>A host copy of this buffer's host contents, with its own device memory.</summary>
    public Buf Clone()
    {
        var c = new Buf(DType, Elements);
        Buffer.MemoryCopy((void*)Host, (void*)c.Host, c.Bytes, Bytes);
        return c;
    }

    public void ToDevice() => Cuda.Upload(Device, Host, Bytes);

    /// <summary>Copy the device contents into a fresh host array (the buffer's own host copy is the CPU reference's).</summary>
    public byte[] FromDevice()
    {
        var a = new byte[Bytes];
        fixed (byte* p = a)
        {
            Cuda.Download((nint)p, Device, Bytes);
        }

        return a;
    }

    public byte[] HostBytes() => new ReadOnlySpan<byte>((void*)Host, (int)Bytes).ToArray();

    public DeviceSpan HostSpan() => new() { Address = Host, Length = (ulong)Elements, Dtype = DType };

    public DeviceSpan DevSpan() => new() { Address = Device, Length = (ulong)Elements, Dtype = DType };

    /// <summary>Element i as a double, from a host-side byte array of this buffer's type.</summary>
    public double At(byte[] bytes, long i) => DType switch
    {
        DType.F32 => BitConverter.ToSingle(bytes, (int)(i * 4)),
        DType.BF16 => Bf16.ToFloat(BitConverter.ToUInt16(bytes, (int)(i * 2))),
        DType.I32 => BitConverter.ToInt32(bytes, (int)(i * 4)),
        _ => throw new NotSupportedException(),
    };

    public void Dispose()
    {
        NativeMemory.Free((void*)Host);
        Cuda.Free(Device);
    }
}

internal static class Bf16
{
    public static ushort From(float f)
    {
        var bits = BitConverter.SingleToUInt32Bits(f);
        if ((bits & 0x7fffffffu) > 0x7f800000u)
        {
            return (ushort)((bits >> 16) | 0x40);
        }

        bits += 0x7fffu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }

    public static float ToFloat(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    public static float Round(float f) => ToFloat(From(f));
}
