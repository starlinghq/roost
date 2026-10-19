using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Starlings.Roost;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;
using Starlings.Roost.Worker.Weights;

// R-025 CPU phase: replays the GPU phase's prompts on CpuOps (double precision) through the same generated forward,
// teacher-forced with the GPU's tokens so every step is compared on the same inputs, and writes a text report: greedy
// agreement, step-0 and per-step logits differences, and per-op max-abs differences for the traced steps.
//   usage: cpu --ckpt <dir> --lock <lock.toml> --dump <dump> --report <text file> [--keep-window-zero] [--only <prompt index>] [--max-steps N]
internal static unsafe class DecodeCpu
{
    private const string Weights = "model.safetensors-00001-of-00001.safetensors";

    public static int Run(string[] args)
    {
        string Opt(string name) => args.SkipWhile(a => a != name).Skip(1).FirstOrDefault() ?? throw new ArgumentException($"missing {name}");
        var dump = Dump.Load(Opt("--dump"));
        var model = new Qwen35EagerModel();
        var lockFile = WeightLock.Load(Opt("--lock"));
        var clock = Stopwatch.StartNew();
        using var weights = ModelWeightLoader.LoadHost(Path.Combine(Opt("--ckpt"), Weights), lockFile.Get(Weights), model.WeightSlots);
        var report = new StringBuilder();
        void Line(string s = "")
        {
            report.AppendLine(s);
            Console.WriteLine(s);
        }

        ProbeState.KeepWindowZero = args.Contains("--keep-window-zero");
        Line($"CPU phase: CpuOps (double precision), weights {weights.Bytes / 1048576.0:F0} MiB verified sha256 {weights.Verification.Sha256} in {clock.Elapsed.TotalSeconds:F1} s");
        Line($"GPU dump: {dump.Device}, kernel library sha256 {dump.LibrarySha256}");
        Line($"WindowLeft handling: {(ProbeState.KeepWindowZero ? "generated default 0 kept (CpuOps attends to the current key only)" : "-1 (no window), both backends")}");
        Line();

        const int MaxTokens = 64;
        var layout = model.Layout(MaxTokens, 1, 8);
        int agree = 0;
        int total = 0;
        double worstStep0 = 0;
        var perOp = new Dictionary<string, Agg>();
        var layerRows = new List<(int Step, int Layer, double MaxAbs, double Scale, double Cos)>();
        var stateRows = new List<(int Step, int Layer, string What, double MaxAbs, double Scale)>();
        int only = args.Contains("--only") ? int.Parse(Opt("--only"), CultureInfo.InvariantCulture) : -1;
        int maxSteps = args.Contains("--max-steps") ? int.Parse(Opt("--max-steps"), CultureInfo.InvariantCulture) : int.MaxValue;
        for (int promptIndex = 0; promptIndex < dump.Prompts.Count; promptIndex++)
        {
            if (only >= 0 && promptIndex != only)
            {
                continue;
            }

            var gpu = dump.Prompts[promptIndex];
            Line($"== prompt {gpu.Name}: ids [{string.Join(' ', gpu.Ids)}], GPU greedy tokens [{string.Join(' ', gpu.Tokens)}]");
            using var run = new HostRun(model, layout, weights, MaxTokens);
            var cpuTokens = new List<int>();
            int firstDivergence = -1;
            for (int step = 0; step < Math.Min(gpu.Tokens.Length, maxSteps); step++)
            {
                int[] stepTokens = step == 0 ? gpu.Ids : [gpu.Tokens[step - 1]]; // teacher forcing: the GPU's token
                int offset = step == 0 ? 0 : gpu.Ids.Length + step - 1;
                bool traced = gpu.Traced.ContainsKey(step);
                var recorder = traced ? new Recorder { Read = HostReader.Read } : null;
                ProbeState.Recorder = recorder;
                var sw = Stopwatch.StartNew();
                var logits = run.Step(stepTokens, offset);
                ProbeState.Recorder = null;
                int cpuToken = ArgMax(logits);
                cpuTokens.Add(cpuToken);
                var g = gpu.Logits[step];
                double maxAbs = 0;
                double dot = 0;
                double na = 0;
                double nb = 0;
                for (int i = 0; i < logits.Length; i++)
                {
                    maxAbs = Math.Max(maxAbs, Math.Abs(logits[i] - g[i]));
                    dot += (double)logits[i] * g[i];
                    na += (double)logits[i] * logits[i];
                    nb += (double)g[i] * g[i];
                }

                var (c1, c2) = Top2(logits);
                var (g1, g2) = Top2(g);
                bool same = cpuToken == gpu.Tokens[step];
                agree += same ? 1 : 0;
                total++;
                if (!same && firstDivergence < 0)
                {
                    firstDivergence = step;
                }

                if (step == 0)
                {
                    worstStep0 = Math.Max(worstStep0, maxAbs);
                }

                Line(string.Create(CultureInfo.InvariantCulture, $"  step {step,2} ({stepTokens.Length,2} tok, {sw.Elapsed.TotalSeconds:F1} s): GPU {gpu.Tokens[step],6} CPU {cpuToken,6} {(same ? "agree   " : "DIFFER  ")} logits max|d| {maxAbs:F4} (logit range {logits.Max() - logits.Min():F1}) cos {dot / Math.Sqrt(na * nb):F7}; top-2 margin GPU {g1 - g2:F3} CPU {c1 - c2:F3}"));
                if (traced)
                {
                    Compare(gpu.Traced[step], recorder!.Snapshots, step, perOp, layerRows, stateRows);
                }
            }

            Line($"  greedy: {gpu.Tokens.Zip(cpuTokens).Count(static p => p.First == p.Second)} of {cpuTokens.Count} tokens agree; first divergence {(firstDivergence < 0 ? "none" : firstDivergence.ToString(CultureInfo.InvariantCulture))}");
            Line();
        }

        Line($"== summary: {agree} of {total} greedy tokens agree across {dump.Prompts.Count} prompts; worst step-0 logits max|d| {worstStep0:F4}");
        Line();
        Line("== per-layer hidden state (Residual 'layer' output h{i+1}), traced steps; scale = max|CPU|, rel = max|d| / scale");
        Line("   step layer   max|d|   scale     rel     cos");
        foreach (var r in layerRows)
        {
            Line(string.Create(CultureInfo.InvariantCulture, $"   {r.Step,4} {r.Layer,5} {r.MaxAbs,8:G4} {r.Scale,8:G4} {r.MaxAbs / Math.Max(r.Scale, 1e-30),8:G3} {r.Cos,9:F7}"));
        }

        Line();
        Line("== recurrent and convolution state after the step (first slot), per linear layer, traced steps");
        Line("   step layer what    max|d|   scale     rel");
        foreach (var r in stateRows)
        {
            Line(string.Create(CultureInfo.InvariantCulture, $"   {r.Step,4} {r.Layer,5} {r.What,-5} {r.MaxAbs,8:G4} {r.Scale,8:G4} {r.MaxAbs / Math.Max(r.Scale, 1e-30),8:G3}"));
        }

        Line();
        Line("== per op class over all traced steps and layers: worst max|d|, worst rel (max|d| / max|CPU| of that output), worst mean|d| / mean|CPU|");
        Line("   op.output                      outputs   worst max|d|  worst rel   worst meanrel");
        foreach (var (key, a) in perOp.OrderBy(static k => k.Key, StringComparer.Ordinal))
        {
            Line(string.Create(CultureInfo.InvariantCulture, $"   {key,-30} {a.Count,7} {a.MaxAbs,14:G4} {a.MaxRel,10:G3} {a.MaxMeanRel,13:G3}"));
        }

        File.WriteAllText(Opt("--report"), report.ToString());
        return 0;
    }

    private static void Compare(
        List<Snapshot> gpu,
        List<Snapshot> cpu,
        int step,
        Dictionary<string, Agg> perOp,
        List<(int Step, int Layer, double MaxAbs, double Scale, double Cos)> layerRows,
        List<(int Step, int Layer, string What, double MaxAbs, double Scale)> stateRows)
    {
        if (gpu.Count != cpu.Count)
        {
            throw new InvalidOperationException($"step {step}: GPU recorded {gpu.Count} outputs, CPU {cpu.Count}");
        }

        for (int i = 0; i < gpu.Count; i++)
        {
            var g = gpu[i];
            var c = cpu[i];
            if (g.Op != c.Op || g.What != c.What || g.Values.Length != c.Values.Length)
            {
                throw new InvalidOperationException($"step {step} output {i}: GPU {g.Op}.{g.What}[{g.Values.Length}] vs CPU {c.Op}.{c.What}[{c.Values.Length}]");
            }

            double maxAbs = 0;
            double scale = 0;
            double sumAbs = 0;
            double sumRef = 0;
            double dot = 0;
            double ng = 0;
            double nc = 0;
            for (int k = 0; k < g.Values.Length; k++)
            {
                double d = Math.Abs((double)g.Values[k] - c.Values[k]);
                maxAbs = Math.Max(maxAbs, d);
                scale = Math.Max(scale, Math.Abs(c.Values[k]));
                sumAbs += d;
                sumRef += Math.Abs(c.Values[k]);
                dot += (double)g.Values[k] * c.Values[k];
                ng += (double)g.Values[k] * g.Values[k];
                nc += (double)c.Values[k] * c.Values[k];
            }

            string key = $"{g.Op}.{g.What}";
            if (!perOp.TryGetValue(key, out var agg))
            {
                perOp[key] = agg = new Agg();
            }

            agg.Add(maxAbs, scale, sumAbs / Math.Max(sumRef, 1e-30));
            if (g.Op == "Residual" && g.What == "layer")
            {
                layerRows.Add((step, g.Layer, maxAbs, scale, dot / Math.Sqrt(Math.Max(ng * nc, 1e-300))));
            }

            if (g.What == "state")
            {
                stateRows.Add((step, g.Layer, g.Op == "GatedDeltaRule" ? "delta" : "conv", maxAbs, scale));
            }
        }
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static (float First, float Second) Top2(float[] v)
    {
        float a = float.NegativeInfinity;
        float b = float.NegativeInfinity;
        foreach (float x in v)
        {
            if (x > a)
            {
                b = a;
                a = x;
            }
            else if (x > b)
            {
                b = x;
            }
        }

        return (a, b);
    }

    private sealed class Agg
    {
        public int Count { get; private set; }

        public double MaxAbs { get; private set; }

        public double MaxRel { get; private set; }

        public double MaxMeanRel { get; private set; }

        public void Add(double maxAbs, double scale, double meanRel)
        {
            Count++;
            MaxAbs = Math.Max(MaxAbs, maxAbs);
            MaxRel = Math.Max(MaxRel, maxAbs / Math.Max(scale, 1e-30));
            MaxMeanRel = Math.Max(MaxMeanRel, meanRel);
        }
    }

    /// <summary>One sequence on CpuOps: host arena, plan region and state pools, the same plan writer and forward as the worker.</summary>
    private sealed class HostRun : IDisposable
    {
        private readonly Qwen35EagerModel _model;
        private readonly EagerLayout _layout;
        private readonly HostModelWeights _weights;
        private readonly int _maxTokens;
        private readonly List<nint> _blocks = [];
        private readonly DeviceSpan _arena;
        private readonly DeviceSpan _plan;
        private readonly DeviceSpan[] _state;

        public HostRun(Qwen35EagerModel model, EagerLayout layout, HostModelWeights weights, int maxTokens)
        {
            _model = model;
            _layout = layout;
            _weights = weights;
            _maxTokens = maxTokens;
            _arena = Alloc(layout.ArenaBytes);
            _plan = Alloc(layout.PlanBytes);
            _state = new DeviceSpan[model.StateSlots.Count];
            for (int i = 0; i < _state.Length; i++)
            {
                var slot = model.StateSlots[i];
                long count = slot.Recurrent ? 1 : 8;
                var raw = Alloc(count * slot.ElementsPerBlock * (slot.Dtype == DType.F32 ? 4 : 2));
                _state[i] = new DeviceSpan { Address = raw.Address, Length = (ulong)(count * slot.ElementsPerBlock), Dtype = slot.Dtype };
            }
        }

        public float[] Step(int[] tokens, int offset)
        {
            int pages = (offset + tokens.Length + 15) / 16;
            var bytes = PlanWriter.Write(1, 2, new TestSequence(1, tokens, offset, Enumerable.Range(0, pages).ToArray(), 0));
            var limits = new PlanLimits(_maxTokens, 1, 8, 1, _model.VocabSize, 16, 8, 1, 0, 2);
            var plan = StepPlanReader.Parse(bytes, limits);
            var host = new Span<byte>((void*)_plan.Address, checked((int)_layout.PlanBytes));
            PlanRegionWriter.Fill(host, _layout, plan, 16);
            if (offset == 0)
            {
                for (int i = 0; i < _state.Length; i++)
                {
                    if (_model.StateSlots[i].Recurrent)
                    {
                        new Span<byte>((void*)_state[i].Address, checked((int)(_state[i].Length * (ulong)(_state[i].Dtype == DType.F32 ? 4 : 2)))).Clear();
                    }
                }
            }

            var input = new EagerForward(_arena, _plan, [.. _weights.Spans], _state, _maxTokens, 1, 8, tokens.Length, 1, 1, plan.KvBlocks);
            _model.Forward<CpuOps>(in input);
            var logits = _model.Logits(in input);
            return new ReadOnlySpan<float>((void*)logits.Address, _model.VocabSize).ToArray();
        }

        public void Dispose()
        {
            foreach (nint p in _blocks)
            {
                NativeMemory.AlignedFree((void*)p);
            }
        }

        private DeviceSpan Alloc(long bytes)
        {
            bytes = Math.Max(bytes, 64);
            void* p = NativeMemory.AlignedAlloc((nuint)bytes, 64);
            NativeMemory.Clear(p, (nuint)bytes);
            _blocks.Add((nint)p);
            return new DeviceSpan { Address = (nint)p, Length = (ulong)bytes, Dtype = DType.U8 };
        }
    }
}
