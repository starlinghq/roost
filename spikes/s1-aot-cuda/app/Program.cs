using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
using S1;

Cuda.Init();
string mode = args.Length > 0 ? args[0] : "help";
return mode switch
{
    "pinvoke" => Bench.PInvoke(),
    "graph" => Bench.Graph(),
    "rss" => Bench.RssSplit(),
    "serve" => await Host.Serve(args),
    _ => Help(),
};

static int Help() { Console.WriteLine("usage: s1app pinvoke|graph|rss|serve [url]"); return 2; }

internal static unsafe class Bench
{
    const int N = 1_000_000;

    public static int PInvoke()
    {
        Console.WriteLine($"timer: Stopwatch.Frequency={Stopwatch.Frequency} Hz; AOT={!System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported}");
        Cuda.Check(Cuda.Noop(), "noop");
        nint noopPtr = NativeLibrary.GetExport(NativeLibrary.Load(Environment.GetEnvironmentVariable("S1_KERNELS") ?? Path.Combine(AppContext.BaseDirectory, "libs1kernels.so")), "s1_noop");
        var fp = (delegate* unmanaged<int>)noopPtr;
        var fpNoGc = (delegate* unmanaged[SuppressGCTransition]<int>)noopPtr;

        // Timer baseline: back-to-back timestamps.
        var t = new long[N];
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < N; i++) { long a = Stopwatch.GetTimestamp(); t[i] = Stopwatch.GetTimestamp() - a; }
        double baseline = Median(t) * Stats.NsPerTick;
        Console.WriteLine(Stats.Summarize("timer baseline", t));

        int sink = 0;
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < N; i++) { long a = Stopwatch.GetTimestamp(); sink += Cuda.Noop(); t[i] = Stopwatch.GetTimestamp() - a; }
        Console.WriteLine(Stats.Summarize("LibraryImport noop", t, baseline));
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < N; i++) { long a = Stopwatch.GetTimestamp(); sink += Cuda.NoopSuppress(); t[i] = Stopwatch.GetTimestamp() - a; }
        Console.WriteLine(Stats.Summarize("LibraryImport+SuppressGCTransition", t, baseline));
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < N; i++) { long a = Stopwatch.GetTimestamp(); sink += fp(); t[i] = Stopwatch.GetTimestamp() - a; }
        Console.WriteLine(Stats.Summarize("delegate* unmanaged noop", t, baseline));
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < N; i++) { long a = Stopwatch.GetTimestamp(); sink += fpNoGc(); t[i] = Stopwatch.GetTimestamp() - a; }
        Console.WriteLine(Stats.Summarize("delegate* unmanaged+SuppressGC", t, baseline));

        // Amortized: tight loops, no per-call timestamps.
        Console.WriteLine($"amortized (10M calls, loop total/N): LibraryImport={Loop(() => { for (int i = 0; i < 10_000_000; i++) sink += Cuda.Noop(); }, 10_000_000):F2} ns, " +
            $"SuppressGC={Loop(() => { for (int i = 0; i < 10_000_000; i++) sink += Cuda.NoopSuppress(); }, 10_000_000):F2} ns, " +
            $"fnptr={Loop(() => { for (int i = 0; i < 10_000_000; i++) sink += fp(); }, 10_000_000):F2} ns, " +
            $"fnptr+SuppressGC={Loop(() => { for (int i = 0; i < 10_000_000; i++) sink += fpNoGc(); }, 10_000_000):F2} ns");

        // Real CUDA runtime calls for context.
        Cuda.Check(Cuda.Malloc(out void* d, 4 << 20), "malloc");
        Cuda.Check(Cuda.StreamCreate(out nint s, 1), "stream");
        int dev = 0;
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < N; i++) { long a = Stopwatch.GetTimestamp(); sink += Cuda.GetDevice(out dev); t[i] = Stopwatch.GetTimestamp() - a; }
        Console.WriteLine(Stats.Summarize("cudaGetDevice (runtime call)", t, baseline));
        const int K = 100_000; var tk = new long[K];
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < K; i++)
            {
                long a = Stopwatch.GetTimestamp(); sink += Cuda.LaunchInc((float*)d, 1024, s); tk[i] = Stopwatch.GetTimestamp() - a;
                if ((i & 1023) == 1023) Cuda.StreamSync(s);
            }
        Cuda.StreamSync(s);
        Console.WriteLine(Stats.Summarize("eager kernel launch (host side)", tk, baseline));
        Cuda.Free(d); Cuda.StreamDestroy(s);
        Console.WriteLine($"gen0 collections during run: {GC.CollectionCount(0)} (sink={sink})");
        return 0;
    }

    static double Loop(Action a, int n) { long s = Stopwatch.GetTimestamp(); a(); return (Stopwatch.GetTimestamp() - s) * Stats.NsPerTick / n; }
    static double Median(long[] a) { var c = (long[])a.Clone(); Array.Sort(c); return c[c.Length / 2]; }

    public static int Graph()
    {
        Cuda.Check(Cuda.RuntimeVersion(out int rv), "rtver"); Cuda.Check(Cuda.DrvInit(0), "cuInit"); Cuda.Check(Cuda.DrvVersion(out int dv), "drvver");
        Cuda.Check(Cuda.DrvDeviceCount(out int ndev), "devcount");
        Console.WriteLine($"cudart={rv} driver={dv} visible_devices={ndev}");
        int Elems = int.Parse(Environment.GetEnvironmentVariable("S1_ELEMS") ?? "1048576"); Console.WriteLine($"kernel elements per launch: {Elems}");
        Cuda.Check(Cuda.Malloc(out void* d, (nuint)(Elems * 4)), "malloc");
        Cuda.Check(Cuda.Memset(d, 0, (nuint)(Elems * 4)), "memset");
        Cuda.Check(Cuda.StreamCreate(out nint s, 1), "stream");
        long launches = 0;
        foreach (int kcount in new[] { 1, 8, 32, 128 })
        {
            // capture + instantiate cost, repeated
            var cap = new long[20]; nint exec = 0;
            for (int r = 0; r < cap.Length; r++)
            {
                if (exec != 0) Cuda.GraphExecDestroy(exec);
                long a = Stopwatch.GetTimestamp();
                Cuda.Check(Cuda.BeginCapture(s, 0), "begin");
                for (int k = 0; k < kcount; k++) Cuda.Check(Cuda.LaunchInc((float*)d, Elems, s), "launch(capture)");
                Cuda.Check(Cuda.EndCapture(s, out nint g), "end");
                Cuda.Check(Cuda.GraphInstantiate(out exec, g, 0), "instantiate");
                cap[r] = Stopwatch.GetTimestamp() - a;
                Cuda.GraphDestroy(g);
            }
            Console.WriteLine(Stats.Summarize($"capture+instantiate k={kcount}", cap).Replace(" ns", " ns"));
            // replay
            const int R = 5000;
            for (int w = 0; w < 200; w++) { Cuda.GraphLaunch(exec, s); } Cuda.StreamSync(s); launches += 200L * kcount;
            var rep = new long[R];
            for (int i = 0; i < R; i++)
            {
                long a = Stopwatch.GetTimestamp();
                Cuda.Check(Cuda.GraphLaunch(exec, s), "graphlaunch"); Cuda.Check(Cuda.StreamSync(s), "sync");
                rep[i] = Stopwatch.GetTimestamp() - a;
            }
            launches += (long)R * kcount;
            Console.WriteLine(Stats.Summarize($"graph replay+sync k={kcount}", rep));
            var rep2 = new long[R];
            for (int i = 0; i < R; i++)
            {
                long a = Stopwatch.GetTimestamp(); Cuda.GraphLaunch(exec, s); rep2[i] = Stopwatch.GetTimestamp() - a;
                if ((i & 63) == 63) Cuda.StreamSync(s);
            }
            Cuda.StreamSync(s); launches += (long)R * kcount;
            Console.WriteLine(Stats.Summarize($"graph launch call only k={kcount}", rep2));
            var eag = new long[R];
            for (int i = 0; i < R; i++)
            {
                long a = Stopwatch.GetTimestamp();
                for (int k = 0; k < kcount; k++) Cuda.LaunchInc((float*)d, Elems, s);
                Cuda.Check(Cuda.StreamSync(s), "sync");
                eag[i] = Stopwatch.GetTimestamp() - a;
            }
            launches += (long)R * kcount;
            Console.WriteLine(Stats.Summarize($"eager {kcount} launches+sync k={kcount}", eag));
            Cuda.GraphExecDestroy(exec);
        }
        // correctness: every element must equal the number of kernel executions of the replays/eager runs (capture does not execute)
        float h = 0; Cuda.Check(Cuda.Memcpy(&h, d, 4, Cuda.D2H), "memcpy");
        Console.WriteLine($"correctness: element[0]={h} expected={launches} -> {(h == launches ? "OK" : "MISMATCH")}");
        Cuda.Free(d); Cuda.StreamDestroy(s);
        return h == launches ? 0 : 1;
    }

    public static int RssSplit()
    {
        void Cp(string n) { Console.WriteLine($"[{n}] {Rss.Status()}"); }
        Cp("process start (AOT, pre-CUDA)");
        Cuda.Check(Cuda.DrvInit(0), "cuInit"); Cp("after cuInit (driver loaded)");
        Cuda.Check(Cuda.Free(null), "cudaFree(0)"); Cp("after cudaFree(0) (primary context created)");
        Cuda.Check(Cuda.StreamCreate(out nint s, 1), "stream");
        Cuda.Check(Cuda.Malloc(out void* d, 64 << 20), "malloc"); Cp("after 64 MiB device alloc + stream");
        Cuda.Check(Cuda.LaunchInc((float*)d, 1 << 20, s), "launch"); Cuda.StreamSync(s); Cp("after first kernel launch (module load of kernel image)");
        Cuda.BeginCapture(s, 0); for (int i = 0; i < 32; i++) Cuda.LaunchInc((float*)d, 1 << 20, s);
        Cuda.EndCapture(s, out nint g); Cuda.GraphInstantiate(out nint e, g, 0); for (int i = 0; i < 100; i++) Cuda.GraphLaunch(e, s); Cuda.StreamSync(s);
        Cp("after graph capture + 100 replays");
        GC.Collect(); Cp("after GC.Collect");
        Console.WriteLine("smaps (RSS by mapping):\n" + Rss.Smaps());
        return 0;
    }
}

[JsonSerializable(typeof(Health))]
internal sealed partial class S1Json : JsonSerializerContext { }
internal sealed record Health(string Status, long RssKb, int Steps);

internal static class Host
{
    public static async Task<int> Serve(string[] args)
    {
        Console.Error.WriteLine($"[serve start] {Rss.Status()}");
        Setup(out nint s, out nint d);
        Console.Error.WriteLine($"[cuda ready] {Rss.Status()}");
        var b = WebApplication.CreateSlimBuilder(args.Skip(1).ToArray());
        b.WebHost.UseUrls(Environment.GetEnvironmentVariable("S1_URL") ?? "http://127.0.0.1:5199");
        b.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.TypeInfoResolverChain.Insert(0, S1Json.Default));
        var app = b.Build();
        int steps = 0;
        app.MapGet("/health", () => new Health("ok", 0, steps));
        app.MapGet("/rss", () => Rss.Status() + "\n" + Rss.Smaps());
        // Each SSE event is one GPU step: launch + sync + copy back one value.
        app.MapGet("/sse", (int? n) => TypedResults.ServerSentEvents(Tokens(s, d, n ?? 16)));
        Console.Error.WriteLine($"[host built] {Rss.Status()}");
        app.Lifetime.ApplicationStarted.Register(() => Console.Error.WriteLine($"[listening] {Rss.Status()}"));
        await app.RunAsync();
        return 0;

        async IAsyncEnumerable<string> Tokens(nint stream, nint dev, int count)
        {
            for (int i = 0; i < count; i++)
            {
                float v = Step(stream, dev);
                steps++;
                yield return $"{{\"i\":{i},\"gpu\":{v}}}";
                await Task.Delay(5);
            }
        }
    }

    static unsafe void Setup(out nint s, out nint d)
    {
        Cuda.Check(Cuda.Free(null), "ctx"); Cuda.Check(Cuda.StreamCreate(out s, 1), "stream"); Cuda.Check(Cuda.Malloc(out void* dp, 1 << 20), "malloc");
        Cuda.Check(Cuda.Memset(dp, 0, 1 << 20), "memset");
        d = (nint)dp;
    }

    static unsafe float Step(nint stream, nint dev)
    {
        float v;
        Cuda.Check(Cuda.LaunchInc((float*)dev, 1024, stream), "launch");
        Cuda.Check(Cuda.StreamSync(stream), "sync");
        Cuda.Check(Cuda.Memcpy(&v, (void*)dev, 4, Cuda.D2H), "copy");
        return v;
    }
}
