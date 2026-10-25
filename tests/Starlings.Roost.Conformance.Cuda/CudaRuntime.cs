using System.Runtime.InteropServices;

// Origin: kernels/tests/ops-cuda/Cuda.cs (R-017, stream C): the few CUDA runtime calls resolved from libcudart by name.
// Graph capture is left out; the conformance runner does not capture.
namespace Starlings.Roost.Conformance.Cuda;

/// <summary>The few CUDA runtime calls the harness needs, resolved from libcudart by name (the kernel library shares it).</summary>
internal static unsafe class CudaRuntime
{
    private static readonly nint Lib = Load();

    private static readonly delegate* unmanaged[Cdecl]<int, int> SetDevice = (delegate* unmanaged[Cdecl]<int, int>)Sym("cudaSetDevice");
    private static readonly delegate* unmanaged[Cdecl]<nint*, nuint, int> Malloc = (delegate* unmanaged[Cdecl]<nint*, nuint, int>)Sym("cudaMalloc");
    private static readonly delegate* unmanaged[Cdecl]<nint, int> Free = (delegate* unmanaged[Cdecl]<nint, int>)Sym("cudaFree");
    private static readonly delegate* unmanaged[Cdecl]<nint*, nuint, int> MallocHost = (delegate* unmanaged[Cdecl]<nint*, nuint, int>)Sym("cudaMallocHost");
    private static readonly delegate* unmanaged[Cdecl]<nint, int> FreeHost = (delegate* unmanaged[Cdecl]<nint, int>)Sym("cudaFreeHost");
    private static readonly delegate* unmanaged[Cdecl]<nint, nint, nuint, int, int> Memcpy = (delegate* unmanaged[Cdecl]<nint, nint, nuint, int, int>)Sym("cudaMemcpy");
    private static readonly delegate* unmanaged[Cdecl]<nint, int, nuint, int> Memset = (delegate* unmanaged[Cdecl]<nint, int, nuint, int>)Sym("cudaMemset");
    private static readonly delegate* unmanaged[Cdecl]<nint*, int> StreamCreate = (delegate* unmanaged[Cdecl]<nint*, int>)Sym("cudaStreamCreate");
    private static readonly delegate* unmanaged[Cdecl]<nint, int> StreamDestroy = (delegate* unmanaged[Cdecl]<nint, int>)Sym("cudaStreamDestroy");
    private static readonly delegate* unmanaged[Cdecl]<nint, int> StreamSynchronize = (delegate* unmanaged[Cdecl]<nint, int>)Sym("cudaStreamSynchronize");
    private static readonly delegate* unmanaged[Cdecl]<int> DeviceSynchronize = (delegate* unmanaged[Cdecl]<int>)Sym("cudaDeviceSynchronize");
    private static readonly delegate* unmanaged[Cdecl]<int> GetLastError = (delegate* unmanaged[Cdecl]<int>)Sym("cudaGetLastError");
    private static readonly delegate* unmanaged[Cdecl]<int, byte*> GetErrorString = (delegate* unmanaged[Cdecl]<int, byte*>)Sym("cudaGetErrorString");
    private static readonly delegate* unmanaged[Cdecl]<nuint*, nuint*, int> MemGetInfo = (delegate* unmanaged[Cdecl]<nuint*, nuint*, int>)Sym("cudaMemGetInfo");
    private static readonly delegate* unmanaged[Cdecl]<int*, int> GetDeviceCount = (delegate* unmanaged[Cdecl]<int*, int>)Sym("cudaGetDeviceCount");
    private static readonly delegate* unmanaged[Cdecl]<int*, int> DriverGetVersion = (delegate* unmanaged[Cdecl]<int*, int>)Sym("cudaDriverGetVersion");
    private static readonly delegate* unmanaged[Cdecl]<int*, int> RuntimeGetVersion = (delegate* unmanaged[Cdecl]<int*, int>)Sym("cudaRuntimeGetVersion");
    private static readonly delegate* unmanaged[Cdecl]<int*, int, int, int> DeviceGetAttribute = (delegate* unmanaged[Cdecl]<int*, int, int, int>)Sym("cudaDeviceGetAttribute");

    private static nint Load()
    {
        string? env = Environment.GetEnvironmentVariable("ROOST_CUDART");
        foreach (string candidate in new[] { env ?? string.Empty, "libcudart.so.13", "/opt/cuda/lib64/libcudart.so.13" })
        {
            if (candidate.Length > 0 && NativeLibrary.TryLoad(candidate, out nint handle))
            {
                return handle;
            }
        }

        throw new InvalidOperationException("libcudart.so.13 not found; set ROOST_CUDART.");
    }

    private static nint Sym(string name) => NativeLibrary.GetExport(Lib, name);

    public static void Check(int code, string what)
    {
        if (code != 0)
        {
            throw new InvalidOperationException($"{what}: CUDA error {code} ({Marshal.PtrToStringAnsi((nint)GetErrorString(code))})");
        }
    }

    public static void Device(int ordinal) => Check(SetDevice(ordinal), "cudaSetDevice");

    public static int DeviceCount()
    {
        int n = 0;
        Check(GetDeviceCount(&n), "cudaGetDeviceCount");
        return n;
    }

    public static (int Driver, int Runtime) Versions()
    {
        int d = 0, r = 0;
        Check(DriverGetVersion(&d), "cudaDriverGetVersion");
        Check(RuntimeGetVersion(&r), "cudaRuntimeGetVersion");
        return (d, r);
    }

    public static int ComputeCapability(int device)
    {
        int major = 0, minor = 0;
        Check(DeviceGetAttribute(&major, 75, device), "cudaDevAttrComputeCapabilityMajor");
        Check(DeviceGetAttribute(&minor, 76, device), "cudaDevAttrComputeCapabilityMinor");
        return (major * 10) + minor;
    }

    public static nint Alloc(long bytes)
    {
        nint p = 0;
        Check(Malloc(&p, (nuint)Math.Max(bytes, 16)), "cudaMalloc");
        return p;
    }

    public static void Release(nint p) => Check(Free(p), "cudaFree");

    public static nint AllocPinned(long bytes)
    {
        nint p = 0;
        Check(MallocHost(&p, (nuint)Math.Max(bytes, 16)), "cudaMallocHost");
        return p;
    }

    public static void ReleasePinned(nint p) => Check(FreeHost(p), "cudaFreeHost");

    public static void ToDevice(nint dst, nint src, long bytes) => Check(Memcpy(dst, src, (nuint)bytes, 1), "cudaMemcpy H2D");

    public static void ToHost(nint dst, nint src, long bytes) => Check(Memcpy(dst, src, (nuint)bytes, 2), "cudaMemcpy D2H");

    public static void Fill(nint dst, int value, long bytes) => Check(Memset(dst, value, (nuint)bytes), "cudaMemset");

    public static nint NewStream()
    {
        nint s = 0;
        Check(StreamCreate(&s), "cudaStreamCreate");
        return s;
    }

    public static void DestroyStream(nint s) => Check(StreamDestroy(s), "cudaStreamDestroy");

    public static void Sync(nint stream) => Check(StreamSynchronize(stream), "cudaStreamSynchronize");

    public static void SyncDevice() => Check(DeviceSynchronize(), "cudaDeviceSynchronize");

    public static void ClearError() => _ = GetLastError();

    public static (long Free, long Total) MemInfo()
    {
        nuint f = 0, t = 0;
        Check(MemGetInfo(&f, &t), "cudaMemGetInfo");
        return ((long)f, (long)t);
    }
}
