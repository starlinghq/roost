using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;
using Starlings.Roost.WorkerHost.Tests.Cuda;

// R-069 / decision 21: the S8 per-step measurement repeated with the .NET host.
//   bench --mode direct|inproc|proc --steps N [--warmup N] [--gpu-us N] [--depth 1|2] [--raw file]
//   pool  <mode> <file>... [<mode> <file>...]      pooled percentiles from raw files, and process minus in-process
// A worker process is this same executable started with --roost-worker-segment (WorkerEntry).
if (WorkerEntry.IsWorkerInvocation(args))
{
    return WorkerEntry.Run(args, _ => CudaSpinExecutor.Create());
}

return args.Length > 0 && args[0] == "pool"
    ? Bench.Pool(args[1..])
    : await Bench.RunAsync(args);

internal static class Bench
{
    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] != "bench")
        {
            Console.Error.WriteLine("usage: bench --mode direct|inproc|proc --steps N [--warmup N] [--gpu-us N] [--depth 1|2] [--raw file] | pool <mode> <file>...");
            return 2;
        }

        string Arg(string name, string fallback)
        {
            var at = Array.IndexOf(args, name);
            return at >= 0 && at + 1 < args.Length ? args[at + 1] : fallback;
        }

        var mode = Arg("--mode", "proc");
        var steps = int.Parse(Arg("--steps", "8000"), CultureInfo.InvariantCulture);
        var warmup = int.Parse(Arg("--warmup", "200"), CultureInfo.InvariantCulture);
        var gpuUs = double.Parse(Arg("--gpu-us", "4500"), CultureInfo.InvariantCulture);
        var depth = int.Parse(Arg("--depth", "1"), CultureInfo.InvariantCulture);
        var raw = Arg("--raw", string.Empty);
        var gpuNs = (ulong)(gpuUs * 1000);
        var load = File.ReadAllText("/proc/loadavg").Split(' ')[0];
        Console.WriteLine($"bench mode={mode} steps={steps} gpu_us={gpuUs:0} depth={depth} loadavg={load} runtime={RuntimeInformation.FrameworkDescription}");

        var latency = new long[steps];
        var seen = new long[steps];
        Result result;
        if (mode == "direct")
        {
            result = RunDirect(steps, warmup, gpuNs, latency);
        }
        else
        {
            var kind = mode == "proc" ? TransportKind.Process : TransportKind.InProcess;
            var transport = WorkerTransports.Create(
                kind,
                kind == TransportKind.Process ? WorkerCommand.ForCurrentProcess() : null,
                CudaSpinExecutor.Create);
            await using (transport)
            {
                result = await RunTransportAsync((WorkerTransportBase)transport, steps, warmup, gpuNs, depth, latency, seen);
            }
        }

        Print("step latency", latency);
        if (mode != "direct")
        {
            Print("worker done->host seen", seen);
        }

        Console.WriteLine($"  wall per step (throughput) = {result.WallNs / 1000.0 / steps:0.0} us");
        Console.WriteLine($"  host thread allocated bytes in the measured loop = {result.AllocatedBytes}");
        Console.WriteLine($"  gc collections during the measured loop: gen0={result.Gen0} gen1={result.Gen1} gen2={result.Gen2}");
        if (raw.Length > 0)
        {
            File.WriteAllLines(raw, latency.Select(v => v.ToString(CultureInfo.InvariantCulture)));
        }

        return 0;
    }

    public static int Pool(string[] args)
    {
        var groups = new Dictionary<string, List<long>>(StringComparer.Ordinal);
        for (var i = 0; i + 1 < args.Length; i += 2)
        {
            if (!groups.TryGetValue(args[i], out var list))
            {
                groups[args[i]] = list = [];
            }

            list.AddRange(File.ReadLines(args[i + 1]).Where(l => l.Length > 0).Select(l => long.Parse(l, CultureInfo.InvariantCulture)));
        }

        var stats = new Dictionary<string, (double P50, double P99, double P999)>(StringComparer.Ordinal);
        foreach (var (mode, values) in groups)
        {
            var sorted = values.Order().ToArray();
            stats[mode] = (At(sorted, 0.5), At(sorted, 0.99), At(sorted, 0.999));
            Console.WriteLine($"{mode,-8} n={sorted.Length} p50={stats[mode].P50:0.0} p99={stats[mode].P99:0.0} p99.9={stats[mode].P999:0.0} (us)");
        }

        if (stats.TryGetValue("proc", out var proc) && stats.TryGetValue("inproc", out var inproc))
        {
            Console.WriteLine($"process minus in-process: p50 {proc.P50 - inproc.P50:+0.0;-0.0} us, p99 {proc.P99 - inproc.P99:+0.0;-0.0} us");
        }

        if (stats.TryGetValue("direct", out var direct) && stats.TryGetValue("inproc", out inproc))
        {
            Console.WriteLine($"in-process minus direct:  p50 {inproc.P50 - direct.P50:+0.0;-0.0} us, p99 {inproc.P99 - direct.P99:+0.0;-0.0} us");
        }

        return 0;
    }

    private static double At(long[] sorted, double q) => sorted[Math.Max(0, (int)Math.Ceiling(q * sorted.Length) - 1)] / 1000.0;

    private static void Print(string label, long[] ns)
    {
        var sorted = ns.Order().ToArray();
        Console.WriteLine($"  {label,-24} n={sorted.Length} min={sorted[0] / 1000.0:0.0} p50={At(sorted, 0.5):0.0} p90={At(sorted, 0.9):0.0} p99={At(sorted, 0.99):0.0} p99.9={At(sorted, 0.999):0.0} max={sorted[^1] / 1000.0:0.0} mean={sorted.Average() / 1000.0:0.0} (us)");
    }

    private static unsafe Result RunDirect(int steps, int warmup, ulong gpuNs, long[] latency)
    {
        var plan = GC.AllocateArray<byte>(CudaSpinExecutor.PlanBytes, pinned: true);
        var body = GC.AllocateArray<byte>(CudaSpinExecutor.ResultBytes, pinned: true);
        fixed (byte* p = plan)
        fixed (byte* b = body)
        {
            var open = CudaSpinExecutor.OpenDirect(0, (nint)p, plan.Length);
            if (open != 0 || CudaSpinExecutor.RegisterHost((nint)b, body.Length) != 0)
            {
                throw new InvalidOperationException($"device open or registration failed: {open}");
            }

            for (var i = 0; i < warmup + steps; i++)
            {
                CudaSpinExecutor.FillPlan(plan, (ulong)i + 1, gpuNs);
                var t0 = Stopwatch.GetTimestamp();
                var error = CudaSpinExecutor.StepDirect(p, b, out _);
                var t1 = Stopwatch.GetTimestamp();
                if (error != 0)
                {
                    throw new InvalidOperationException($"step failed: {error}");
                }

                if (i >= warmup)
                {
                    latency[i - warmup] = t1 - t0;
                }
            }
        }

        return new Result(latency.Sum(), 0, 0, 0, 0);
    }

    private static async Task<Result> RunTransportAsync(WorkerTransportBase transport, int steps, int warmup, ulong gpuNs, int depth, long[] latency, long[] seen)
    {
        var launch = new WorkerLaunch
        {
            DeviceOrdinal = 0,
            Mesh = (0, 1),
            ManifestDigest = CudaSpinExecutor.ManifestDigest,
            WeightsDigest = CudaSpinExecutor.WeightsDigest,
            StepDeadline = TimeSpan.FromSeconds(30),
            HeartbeatPeriod = TimeSpan.FromMilliseconds(1),
            PlanBufferCount = 4,
            ResultBufferCount = 4,
            PlanBufferBytes = CudaSpinExecutor.PlanBytes,
            ResultBufferBytes = CudaSpinExecutor.ResultBytes,
        };
        var started = await transport.StartAsync(launch, CancellationToken.None);
        if (started.State != HandshakeState.Ready)
        {
            throw new InvalidOperationException($"handshake {started.State}/{started.Reject}");
        }

        var total = warmup + steps;
        var published = 0;
        var done = 0;
        var startNs = new long[total];
        long wallStart = 0;
        long allocBefore = 0;
        int gc0 = 0, gc1 = 0, gc2 = 0;
        var wait = new SpinWait();
        while (done < total)
        {
            while (published < total && published - done < depth && transport.CanPublish())
            {
                var index = transport.NextPlanBufferIndex;
                CudaSpinExecutor.FillPlan(transport.PlanBuffer(index), (ulong)published + 1, gpuNs);
                if (published == warmup)
                {
                    wallStart = Stopwatch.GetTimestamp();
                    allocBefore = GC.GetAllocatedBytesForCurrentThread();
                    gc0 = GC.CollectionCount(0);
                    gc1 = GC.CollectionCount(1);
                    gc2 = GC.CollectionCount(2);
                }

                startNs[published] = Stopwatch.GetTimestamp();
                if (!transport.TryPublish(new RingSlot { StepId = (ulong)published + 1, Kind = RingSlotKind.Step, BufferIndex = (uint)index }))
                {
                    throw new InvalidOperationException("publish refused");
                }

                published++;
            }

            if (transport.TryTakeResult(out var slot))
            {
                var now = Stopwatch.GetTimestamp();
                var workerDone = (long)slot.PublishedNs;
                var header = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<StepResultHeader>(
                    ref System.Runtime.CompilerServices.Unsafe.AsRef(in transport.ResultBuffer((int)slot.BufferIndex)[0]));
                if (header.Fault != FaultClass.None || header.StepId != slot.StepId)
                {
                    throw new InvalidOperationException($"step {slot.StepId} faulted: {header.Fault} {header.ErrorCode}");
                }

                transport.ReleaseResult((int)slot.BufferIndex);
                if (done >= warmup)
                {
                    latency[done - warmup] = now - startNs[done];
                    seen[done - warmup] = Math.Max(0, now - workerDone);
                }

                done++;
                wait.Reset();
                continue;
            }

            var fault = transport.Poll();
            if (fault != FaultClass.None)
            {
                throw new InvalidOperationException($"fault {fault} at step {done}");
            }

            wait.SpinOnce(-1);
        }

        var wall = Stopwatch.GetTimestamp() - wallStart;
        var allocated = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
        return new Result(wall, allocated, GC.CollectionCount(0) - gc0, GC.CollectionCount(1) - gc1, GC.CollectionCount(2) - gc2);
    }

    private readonly record struct Result(long WallNs, long AllocatedBytes, int Gen0, int Gen1, int Gen2);
}
