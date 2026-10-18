// Spike S6: host expert GEMV (bf16 weights x f32 activation -> f32, batch 1) with .NET intrinsics.
using System.Runtime.InteropServices;
// y[n] = sum_k W[n,k] * x[k]; W row-major [N,K] as bf16 (ushort). bf16 -> f32 is a 16-bit left shift.
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

static unsafe class Gemv
{
    public static void Scalar(ushort* w, float* x, float* y, int n0, int n1, int k)
    {
        for (int n = n0; n < n1; n++)
        {
            double acc = 0;
            ushort* r = w + (long)n * k;
            for (int i = 0; i < k; i++) acc += BitConverter.UInt32BitsToSingle((uint)r[i] << 16) * (double)x[i];
            y[n] = (float)acc;
        }
    }

    // Vector256: 8 bf16 per load, widen, shift, FMA; four rows at once share the x loads.
    public static void V256(ushort* w, float* x, float* y, int n0, int n1, int k)
    {
        int n = n0;
        for (; n + 4 <= n1; n += 4)
        {
            ushort* r0 = w + (long)n * k, r1 = r0 + k, r2 = r1 + k, r3 = r2 + k;
            Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
            for (int i = 0; i < k; i += 8)
            {
                var xv = Vector256.Load(x + i);
                a0 = Fma.MultiplyAdd(Widen256(r0 + i), xv, a0);
                a1 = Fma.MultiplyAdd(Widen256(r1 + i), xv, a1);
                a2 = Fma.MultiplyAdd(Widen256(r2 + i), xv, a2);
                a3 = Fma.MultiplyAdd(Widen256(r3 + i), xv, a3);
            }
            y[n] = Vector256.Sum(a0); y[n + 1] = Vector256.Sum(a1); y[n + 2] = Vector256.Sum(a2); y[n + 3] = Vector256.Sum(a3);
        }
        for (; n < n1; n++) Scalar(w, x, y, n, n + 1, k);
    }

    static Vector256<float> Widen256(ushort* p) =>
        Vector256.ShiftLeft(Avx2.ConvertToVector256Int32(Vector128.Load(p)).AsUInt32(), 16).AsSingle();

    // Vector512 using the portable API; on a CPU without AVX-512 the JIT emits a software fallback.
    public static void V512(ushort* w, float* x, float* y, int n0, int n1, int k)
    {
        int n = n0;
        for (; n + 4 <= n1; n += 4)
        {
            ushort* r0 = w + (long)n * k, r1 = r0 + k, r2 = r1 + k, r3 = r2 + k;
            Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default;
            for (int i = 0; i < k; i += 16)
            {
                var xv = Vector512.Load(x + i);
                a0 = Vector512.FusedMultiplyAdd(Widen512(r0 + i), xv, a0);
                a1 = Vector512.FusedMultiplyAdd(Widen512(r1 + i), xv, a1);
                a2 = Vector512.FusedMultiplyAdd(Widen512(r2 + i), xv, a2);
                a3 = Vector512.FusedMultiplyAdd(Widen512(r3 + i), xv, a3);
            }
            y[n] = Vector512.Sum(a0); y[n + 1] = Vector512.Sum(a1); y[n + 2] = Vector512.Sum(a2); y[n + 3] = Vector512.Sum(a3);
        }
        for (; n < n1; n++) Scalar(w, x, y, n, n + 1, k);
    }

    static Vector512<float> Widen512(ushort* p) =>
        Vector512.ShiftLeft(Vector512.WidenLower(Vector512.Create(Vector256.Load(p), Vector256<ushort>.Zero)), 16).AsSingle();
}

static unsafe class Program
{
    delegate void Kernel(ushort* w, float* x, float* y, int n0, int n1, int k);

    static int Main(string[] args)
    {
        int threads = args.Length > 0 ? int.Parse(args[0]) : 1;
        int reps = args.Length > 1 ? int.Parse(args[1]) : 5;
        Console.WriteLine($"dotnet {Environment.Version}; Avx2={Avx2.IsSupported} Fma={Fma.IsSupported} Avx512F={Avx512F.IsSupported}; " +
            $"Vector256.IsHardwareAccelerated={Vector256.IsHardwareAccelerated} Vector512.IsHardwareAccelerated={Vector512.IsHardwareAccelerated}; threads={threads}");
        // Qwen3-30B-A3B expert: gate_up [2I=1536, H=2048], down [H=2048, I=768]. 128 experts rotate so weights stream from DRAM.
        var shapes = new (string name, int n, int k)[] { ("gate_up 1536x2048", 1536, 2048), ("down 2048x768", 2048, 768) };
        var kernels = new (string name, Kernel f)[] { ("scalar-f64", Gemv.Scalar), ("Vector256", Gemv.V256), ("Vector512(sw)", Gemv.V512) };
        foreach (var (sname, n, k) in shapes)
        {
            foreach (int experts in new[] { 1, 128 })
            {
                long bytesPer = (long)n * k * 2;
                var rng = new Random(7);
                ushort* w = (ushort*)NativeMemory.AlignedAlloc((nuint)(bytesPer * experts), 64);
                for (long i = 0; i < (long)n * k * experts; i++) w[i] = (ushort)(((uint)BitConverter.SingleToUInt32Bits(((float)rng.NextDouble() - 0.5f) * 0.05f)) >> 16);
                float* x = (float*)NativeMemory.AlignedAlloc((nuint)(k * 4), 64);
                float* y = (float*)NativeMemory.AlignedAlloc((nuint)(n * 4), 64);
                float* yref = (float*)NativeMemory.AlignedAlloc((nuint)(n * 4), 64);
                for (int i = 0; i < k; i++) x[i] = (float)rng.NextDouble() - 0.5f;
                foreach (var (kname, f) in kernels)
                {
                    int iters = kname == "scalar-f64" ? 4 : kname.StartsWith("Vector512") ? 60 : 400;
                    double best = double.MaxValue;
                    for (int rep = 0; rep < reps; rep++)
                    {
                        var sw = Stopwatch.StartNew();
                        for (int it = 0; it < iters; it++) Run(f, w + (long)(it % experts) * n * k, x, y, n, k, threads);
                        best = Math.Min(best, sw.Elapsed.TotalSeconds / iters);
                    }
                    Gemv.Scalar(w + (long)((iters - 1) % experts) * n * k, x, yref, 0, n, k);  // reference for the expert the last call used
                    double maxErr = 0; for (int i = 0; i < n; i++) maxErr = Math.Max(maxErr, Math.Abs(y[i] - yref[i]));
                    Console.WriteLine($"RESULT impl=dotnet-{kname} shape=\"{sname}\" experts_rotated={experts} threads={threads} " +
                        $"time_us={best * 1e6:F1} GB/s={bytesPer / best / 1e9:F2} GFLOP/s={2.0 * n * k / best / 1e9:F2} max_abs_err_vs_f64={maxErr:E2}");
                }
                NativeMemory.AlignedFree(w); NativeMemory.AlignedFree(x); NativeMemory.AlignedFree(y); NativeMemory.AlignedFree(yref);
            }
        }
        return 0;
    }

    static void Run(Kernel f, ushort* w, float* x, float* y, int n, int k, int threads)
    {
        if (threads == 1) { f(w, x, y, 0, n, k); return; }
        int chunk = ((n + threads - 1) / threads + 3) & ~3;
        nuint wp = (nuint)w, xp = (nuint)x, yp = (nuint)y;
        Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, t =>
        {
            int n0 = t * chunk, n1 = Math.Min(n, n0 + chunk);
            if (n0 < n1) f((ushort*)wp, (float*)xp, (float*)yp, n0, n1, k);
        });
    }
}
