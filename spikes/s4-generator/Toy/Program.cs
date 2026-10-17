using System;
using System.Linq;
using System.Runtime.CompilerServices;
using Spike.Ops;
using Spike.Toy;

namespace Spike.Toy;

internal static unsafe class Program
{
    private const int Tokens = 4;
    private const int Iterations = 20000;

    private static int Main()
    {
        string mode = System.IO.Path.GetFileNameWithoutExtension(Environment.ProcessPath) == "dotnet" ? "JIT" : "AOT";
        Console.WriteLine($"mode={mode} weights={string.Join(",", ToyModel.WeightSlots)} passes={string.Join(",", ToyModel.Passes)}");

        using var w = new ToyModel.Workspace(8);
        int* ids = (int*)w.TokenIds.Ptr;
        for (int i = 0; i < Tokens; i++) ids[i] = (i * 7 + 3) % 64;

        int failures = 0;
        failures += Measure("Forward<CpuOps> (unfused identity)", w, static x => ToyModel.Forward<CpuOps>(x, Tokens));
        float[] reference = Snapshot(w);
        failures += Measure("ForwardFused<CpuOps> (fallback to unfused parts)", w, static x => ToyModel.ForwardFused<CpuOps>(x, Tokens));
        failures += Compare("fused-fallback vs unfused", reference, Snapshot(w));
        failures += Measure("ForwardFused<FusedCpuOps> (fused op)", w, static x => ToyModel.ForwardFused<FusedCpuOps>(x, Tokens));
        failures += Compare("fused-op vs unfused", reference, Snapshot(w));
        failures += CheckAgainstManagedReference(w, reference);
        // Negative control: the instrument must see an allocation when there is one.
        long c0 = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) Sink(new float[16]);
        long ctl = GC.GetAllocatedBytesForCurrentThread() - c0;
        Console.WriteLine($"control allocating loop: allocatedBytes={ctl} (must be > 0)");
        if (ctl == 0) failures++;
        Console.WriteLine(failures == 0 ? "RESULT PASS" : $"RESULT FAIL ({failures})");
        return failures == 0 ? 0 : 1;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Sink(float[] a) => a[0] = 1;

    // Independent managed-code transcription of the toy graph in double precision, reading the workspace weights. It
    // guards against the generator wiring the graph wrongly (equal fused and unfused outputs would not catch that).
    private static int CheckAgainstManagedReference(ToyModel.Workspace w, float[] got)
    {
        const int H = 16, I = 32, V = 64;
        double[][] Rows(DeviceSpan s, int cols) => Enumerable.Range(0, s.Length / cols).Select(r => Enumerable.Range(0, cols).Select(c => (double)((float*)s.Ptr)[(r * cols) + c]).ToArray()).ToArray();
        double[] Norm(double[] x, DeviceSpan g)
        {
            double ss = 0;
            foreach (double v in x) ss += v * v;
            double inv = 1.0 / Math.Sqrt((ss / x.Length) + 1e-6);
            return x.Select((v, i) => v * inv * ((float*)g.Ptr)[i]).ToArray();
        }

        double[] Mm(double[] x, DeviceSpan wt, int n)
        {
            var o = new double[n];
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < x.Length; k++) o[j] += x[k] * ((float*)wt.Ptr)[(j * x.Length) + k];
            }

            return o;
        }

        double[] Act(double[] x) => Enumerable.Range(0, x.Length / 2).Select(i => x[i] / (1 + Math.Exp(-x[i])) * x[(x.Length / 2) + i]).ToArray();
        double[] Add(double[] a, double[] b) => a.Select((v, i) => v + b[i]).ToArray();

        var embed = Rows(w.W_embed, H);
        int* ids = (int*)w.TokenIds.Ptr;
        double maxRel = 0;
        for (int t = 0; t < Tokens; t++)
        {
            double[] h = embed[ids[t]];
            h = Add(h, Mm(Act(Mm(Norm(h, w.W_l0_norm1), w.W_l0_gate_up, 2 * I)), w.W_l0_down, H));
            h = Add(h, Mm(Act(Mm(Norm(h, w.W_l1_norm1), w.W_l1_gate_up, 2 * I)), w.W_l1_down, H));
            double[] logits = Mm(Norm(h, w.W_final_norm), w.W_lm_head, V);
            for (int c = 0; c < V; c++)
            {
                double d = Math.Abs(logits[c] - got[(t * V) + c]);
                maxRel = Math.Max(maxRel, d / (Math.Abs(logits[c]) + 1e-3));
            }
        }

        Console.WriteLine($"check generated forward vs managed double reference: max relative diff={maxRel:E3} (tolerance 1e-3)");
        return maxRel < 1e-3 ? 0 : 1;
    }

    private static float[] Snapshot(ToyModel.Workspace w)
    {
        var o = w.Output;
        var r = new float[Tokens * 64];
        new ReadOnlySpan<float>((void*)o.Ptr, r.Length).CopyTo(r);
        return r;
    }

    private static int Compare(string label, float[] a, float[] b)
    {
        float max = 0;
        bool finite = true;
        for (int i = 0; i < a.Length; i++)
        {
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
            finite &= float.IsFinite(a[i]);
        }

        Console.WriteLine($"check {label}: max|diff|={max:E3} finite={finite} logits[0]={a[0]:F6}");
        return max == 0 && finite ? 0 : 1;
    }

    private static int Measure(string label, ToyModel.Workspace w, Action<ToyModel.Workspace> forward)
    {
        // The delegate is created once outside the measured region; the forward itself takes no delegate.
        for (int i = 0; i < 2000; i++) forward(w);
        long b0 = GC.GetAllocatedBytesForCurrentThread();
        int g0 = GC.CollectionCount(0);
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        for (int i = 0; i < Iterations; i++) forward(w);
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        long bytes = GC.GetAllocatedBytesForCurrentThread() - b0;
        int gcs = GC.CollectionCount(0) - g0;
        double us = (t1 - t0) * 1e6 / System.Diagnostics.Stopwatch.Frequency / Iterations;
        Console.WriteLine($"measure {label}: calls={Iterations} allocatedBytes={bytes} gen0GCs={gcs} usPerCall={us:F2}");
        return bytes == 0 && gcs == 0 ? 0 : 1;
    }
}
