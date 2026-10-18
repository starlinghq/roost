using System.Runtime.InteropServices;

namespace S1;

// Minimal hand-written CUDA bindings for spike S1 (the real generator is R-023).
internal static unsafe partial class Cuda
{
    public const string Rt = "cudart";
    public const string Drv = "cuda";
    public const string Kern = "s1kernels";

    public static void Init()
    {
        string here = AppContext.BaseDirectory;
        NativeLibrary.SetDllImportResolver(typeof(Cuda).Assembly, (name, _, _) => name switch
        {
            Rt => NativeLibrary.Load(Environment.GetEnvironmentVariable("S1_CUDART") ?? "/opt/cuda/lib64/libcudart.so.13"),
            Drv => NativeLibrary.Load("libcuda.so.1"),
            Kern => NativeLibrary.Load(Environment.GetEnvironmentVariable("S1_KERNELS") ?? Path.Combine(here, "libs1kernels.so")),
            _ => 0,
        });
    }

    public static void Check(int rc, string what)
    {
        if (rc != 0) throw new InvalidOperationException($"{what} failed: CUDA error {rc}");
    }

    // Runtime API
    [LibraryImport(Rt, EntryPoint = "cudaMalloc")] public static partial int Malloc(out void* p, nuint size);
    [LibraryImport(Rt, EntryPoint = "cudaFree")] public static partial int Free(void* p);
    [LibraryImport(Rt, EntryPoint = "cudaMemcpy")] public static partial int Memcpy(void* dst, void* src, nuint n, int kind);
    [LibraryImport(Rt, EntryPoint = "cudaMemset")] public static partial int Memset(void* p, int v, nuint n);
    [LibraryImport(Rt, EntryPoint = "cudaStreamCreateWithFlags")] public static partial int StreamCreate(out nint s, uint flags);
    [LibraryImport(Rt, EntryPoint = "cudaStreamDestroy")] public static partial int StreamDestroy(nint s);
    [LibraryImport(Rt, EntryPoint = "cudaStreamSynchronize")] public static partial int StreamSync(nint s);
    [LibraryImport(Rt, EntryPoint = "cudaStreamBeginCapture")] public static partial int BeginCapture(nint s, int mode);
    [LibraryImport(Rt, EntryPoint = "cudaStreamEndCapture")] public static partial int EndCapture(nint s, out nint graph);
    [LibraryImport(Rt, EntryPoint = "cudaGraphInstantiate")] public static partial int GraphInstantiate(out nint exec, nint graph, ulong flags);
    [LibraryImport(Rt, EntryPoint = "cudaGraphLaunch")] public static partial int GraphLaunch(nint exec, nint s);
    [LibraryImport(Rt, EntryPoint = "cudaGraphExecDestroy")] public static partial int GraphExecDestroy(nint exec);
    [LibraryImport(Rt, EntryPoint = "cudaGraphDestroy")] public static partial int GraphDestroy(nint g);
    [LibraryImport(Rt, EntryPoint = "cudaGetLastError")] public static partial int GetLastError();
    [LibraryImport(Rt, EntryPoint = "cudaGetDevice")] public static partial int GetDevice(out int dev);
    [LibraryImport(Rt, EntryPoint = "cudaMemGetInfo")] public static partial int MemGetInfo(out nuint free, out nuint total);
    [LibraryImport(Rt, EntryPoint = "cudaRuntimeGetVersion")] public static partial int RuntimeVersion(out int v);
    [LibraryImport(Rt, EntryPoint = "cudaDeviceReset")] public static partial int DeviceReset();

    // Driver API
    [LibraryImport(Drv, EntryPoint = "cuInit")] public static partial int DrvInit(uint flags);
    [LibraryImport(Drv, EntryPoint = "cuDriverGetVersion")] public static partial int DrvVersion(out int v);
    [LibraryImport(Drv, EntryPoint = "cuDeviceGetCount")] public static partial int DrvDeviceCount(out int n);

    // C ABI library built with nvcc
    [LibraryImport(Kern, EntryPoint = "s1_noop")] public static partial int Noop();
    [LibraryImport(Kern, EntryPoint = "s1_noop"), SuppressGCTransition] public static partial int NoopSuppress();
    [LibraryImport(Kern, EntryPoint = "s1_add")] public static partial int Add(int a, int b);
    [LibraryImport(Kern, EntryPoint = "s1_launch_inc")] public static partial int LaunchInc(float* p, int n, nint stream);

    public const int H2D = 1, D2H = 2;
}
