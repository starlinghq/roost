using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.HybridConformance;

/// <summary>GatedDeltaRule (chunked and recurrent variants) against CpuOps on random Qwen3.5-0.8B-shaped inputs.</summary>
internal static unsafe class GdnCases
{
    private const int Heads = 16;
    private const int Dim = 128;
    private const int Row = Heads * Dim;
    private const int Fused = 3 * Row; // the conv1d output row of Qwen3.5-0.8B: q, k, v columns
    private const long StateElems = (long)Heads * Dim * Dim;

    // Development tolerances, relative to the largest reference magnitude of the tensor. Source: S3 (docs/m0/roost-s3-triton-aot-gdn.md)
    // measured the chunked path at 5.0e-4 on o (reference max 8.9e-2, 5.6e-3 relative) and 3.3e-3 on the state (reference max 0.94,
    // 3.5e-3 relative) against an fp32 recurrent reference, from bf16 intermediate rounding inherent to FLA; the limits here are
    // about three times those. The recurrent path keeps everything in float and only rounds o once to bf16 (2^-8 per element).
    // These live in the test, never in a tolerance file; the maintainer sets the real ones (R-019).
    private const double ChunkOutputTol = 1.6e-2;
    private const double ChunkStateTol = 1.0e-2;
    private const double RecurrentOutputTol = 8e-3;
    private const double RecurrentStateTol = 1e-4;

    public static void Run(Library lib, Report r, nint stream)
    {
        var chunk = lib.Find("gated-delta-rule-chunk-aot");
        var recurrent = lib.Find("gated-delta-rule-recurrent-aot");
        r.Check(chunk is not null && recurrent is not null, "gdn/registered", "both GatedDeltaRule variants are in the table");
        if (chunk is null || recurrent is null)
        {
            return;
        }

        r.Check(chunk.Value.ParamsStructSize == (uint)sizeof(GatedDeltaRuleParams) && recurrent.Value.ParamsStructSize == (uint)sizeof(GatedDeltaRuleParams),
            "gdn/params_struct_size", $"sizeof(RoostGatedDeltaRuleParams) = {sizeof(GatedDeltaRuleParams)}");
        r.Check(chunk.Value.Flags.HasFlag(VariantTraits.NeedsPlan) && !recurrent.Value.Flags.HasFlag(VariantTraits.NeedsPlan) && recurrent.Value.Flags.HasFlag(VariantTraits.GraphCapturable),
            "gdn/flags", $"chunk={chunk.Value.Flags} recurrent={recurrent.Value.Flags}");

        var c = chunk.Value;
        var x = recurrent.Value;
        Case(lib, r, stream, c, "chunk/packed [300,77,1500,64,1] initial state, strided views", [300, 77, 1500, 64, 1], strided: true, init: true, decay: 0.2, seed: 11);
        Case(lib, r, stream, c, "chunk/packed [300,77,1500] no initial state, strided views", [300, 77, 1500], strided: true, init: false, decay: 0.2, seed: 12);
        Case(lib, r, stream, c, "chunk/one sequence 1000 contiguous views, initial state", [1000], strided: false, init: true, decay: 0.2, seed: 13);
        Case(lib, r, stream, c, "chunk/chunk boundaries [64,128,65,1,0,63] incl. an empty sequence", [64, 128, 65, 1, 0, 63], strided: true, init: true, decay: 0.2, seed: 14);
        Case(lib, r, stream, c, "chunk/strong decay (g down to -2.5) [200,40]", [200, 40], strided: true, init: true, decay: 2.5, seed: 15);
        Case(lib, r, stream, x, "recurrent/decode 4 x 1 token, strided views, initial state", [1, 1, 1, 1], strided: true, init: true, decay: 0.2, seed: 21);
        Case(lib, r, stream, x, "recurrent/packed [5,1,1], contiguous views", [5, 1, 1], strided: false, init: true, decay: 0.2, seed: 22);
        Case(lib, r, stream, x, "recurrent/one sequence of 3 tokens, no initial state", [3], strided: true, init: false, decay: 0.2, seed: 23);
        Case(lib, r, stream, x, "recurrent/decode with an empty sequence [1,0,1]", [1, 0, 1], strided: true, init: true, decay: 0.2, seed: 24);
        Case(lib, r, stream, x, "recurrent/one sequence of 200 tokens, strided, strong decay", [200], strided: true, init: true, decay: 2.5, seed: 25);
        Case(lib, r, stream, x, "recurrent/32 sequences x 1 token (serving-size decode batch)", Enumerable.Repeat(1, 32).ToArray(), strided: true, init: true, decay: 0.2, seed: 26);
        BatchInvariance(lib, r, stream, c, "chunk");
        BatchInvariance(lib, r, stream, x, "recurrent");
        NegativeSlot(lib, r, stream, c, "chunk");
        NegativeSlot(lib, r, stream, x, "recurrent");
        Rejections(lib, r, stream, c, x);
    }

    /// <summary>The recurrent variant is GRAPH_CAPTURABLE: its cases again with run() captured into a graph.</summary>
    public static void RunGraph(Library lib, Report r, nint stream)
    {
        var recurrent = lib.Find("gated-delta-rule-recurrent-aot");
        if (recurrent is null)
        {
            r.Check(false, "gdn/registered", "recurrent variant missing");
            return;
        }

        var x = recurrent.Value;
        Case(lib, r, stream, x, "recurrent/decode 4 x 1 token, strided views, initial state", [1, 1, 1, 1], strided: true, init: true, decay: 0.2, seed: 21);
        Case(lib, r, stream, x, "recurrent/packed [5,1,1], contiguous views", [5, 1, 1], strided: false, init: true, decay: 0.2, seed: 22);
        Case(lib, r, stream, x, "recurrent/32 sequences x 1 token (serving-size decode batch)", Enumerable.Repeat(1, 32).ToArray(), strided: true, init: true, decay: 0.2, seed: 26);
    }

    // A negative slot with tokens is outside the contract (CpuOps throws), but the library must stay deterministic: the
    // sequence runs from a zero state whatever use_initial_state says, and its state is not written back. Compare the
    // sequence with slot -1 and use_initial_state = 1 against the same inputs with use_initial_state = 0, bit for bit.
    private static void NegativeSlot(Library lib, Report r, nint stream, in OpEntry e, string label)
    {
        const int PoolSlots = 9;
        int[] lens = label == "chunk" ? [100, 70] : [1, 1];
        using var neg = Make(lens, true, 0.2, 51, PoolSlots, [-1, 3]);
        using var zero = Make(lens, true, 0.2, 51, PoolSlots, [-1, 3]);
        var poolBefore = neg.Pool.HostBytes();
        foreach (var (d, init) in new[] { (neg, true), (zero, false) })
        {
            foreach (var b in new[] { d.Fused, d.G, d.Beta, d.Pool, d.Slots, d.Cu })
            {
                b.ToDevice();
            }

            Cuda.Fill(d.Output.Device, 0xFF, d.Output.Bytes);
            Execute(lib, e, GpuParams(d, init), stream);
        }

        int n0 = lens[0];
        bool same = neg.Output.FromDevice().AsSpan(0, n0 * Row * 2).SequenceEqual(zero.Output.FromDevice().AsSpan(0, n0 * Row * 2));
        bool unwritten = true;
        var after = neg.Pool.FromDevice();
        for (int s = 0; s < PoolSlots; s++)
        {
            if (s != 3)
            {
                unwritten &= after.AsSpan((int)(s * StateElems * 4), (int)(StateElems * 4)).SequenceEqual(poolBefore.AsSpan((int)(s * StateElems * 4), (int)(StateElems * 4)));
            }
        }

        r.Check(same && unwritten, $"gdn/negative-slot/{label}", $"slot -1 with initial state runs from zero state, output bit-identical to use_initial_state=0: {same}; no other slot touched: {unwritten}");
    }

    private sealed class Data : IDisposable
    {
        public required Buf Fused { get; init; }
        public required Buf? Q { get; init; }
        public required Buf? K { get; init; }
        public required Buf? V { get; init; }
        public required Buf G { get; init; }
        public required Buf Beta { get; init; }
        public required Buf Output { get; init; }
        public required Buf Pool { get; init; }
        public required Buf Slots { get; init; }
        public required Buf Cu { get; init; }
        public required int Tokens { get; init; }
        public required int Batch { get; init; }
        public required bool Strided { get; init; }

        public void Dispose()
        {
            foreach (var b in new[] { Fused, Q, K, V, G, Beta, Output, Pool, Slots, Cu })
            {
                b?.Dispose();
            }
        }
    }

    private static Data Make(int[] lens, bool strided, double decay, int seed, int poolSlots, int[]? order = null)
    {
        var rng = new Random(seed);
        int tokens = lens.Sum();
        int batch = lens.Length;
        float Normal() => (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        float[] Gauss(long n, double s) => Enumerable.Range(0, (int)n).Select(_ => Normal() * (float)s).ToArray();

        // The fused buffer is the conv1d output [tokens, 6144]; a contiguous case splits it into three tensors.
        var fused = Gauss((long)tokens * Fused, 1.0);
        Buf? q = null, k = null, v = null;
        if (!strided)
        {
            float[] Cols(int c0) => Enumerable.Range(0, tokens * Row).Select(i => fused[((i / Row) * Fused) + c0 + (i % Row)]).ToArray();
            q = Buf.FromBf16(Cols(0));
            k = Buf.FromBf16(Cols(Row));
            v = Buf.FromBf16(Cols(2 * Row));
        }

        var g = Enumerable.Range(0, tokens * Heads).Select(_ => (float)(-rng.NextDouble() * decay)).ToArray();
        var beta = Enumerable.Range(0, tokens * Heads).Select(_ => (float)rng.NextDouble()).ToArray();
        var cu = new int[batch + 1];
        for (int i = 0; i < batch; i++)
        {
            cu[i + 1] = cu[i] + lens[i];
        }

        order ??= Enumerable.Range(0, batch).Select(i => (i * 5 + 3) % poolSlots).ToArray();
        return new Data
        {
            Fused = Buf.FromBf16(fused),
            Q = q,
            K = k,
            V = v,
            G = Buf.FromF32(g),
            Beta = Buf.FromBf16(beta),
            Output = new Buf(DType.BF16, (long)tokens * Row),
            Pool = Buf.FromF32(Gauss(poolSlots * StateElems, 0.1)),
            Slots = Buf.FromI32(order),
            Cu = Buf.FromI32(cu),
            Tokens = tokens,
            Batch = batch,
            Strided = strided,
        };
    }

    private static DeviceSpan View(Buf whole, int col, long tokens, bool device)
    {
        long length = tokens == 0 ? 0 : ((tokens - 1) * Fused) + Row; // exactly the reach of the view
        return new DeviceSpan { Address = (device ? whole.Device : whole.Host) + (col * 2), Length = (ulong)length, Dtype = DType.BF16 };
    }

    private static GatedDeltaRuleParams GpuParams(Data d, bool init)
    {
        var p = new GatedDeltaRuleParams
        {
            StructSize = (uint)sizeof(GatedDeltaRuleParams),
            AbiMinor = 1,
            G = d.G.DevSpan(),
            Beta = d.Beta.DevSpan(),
            Output = d.Output.DevSpan(),
            State = d.Pool.DevSpan(),
            StateSlots = d.Slots.DevSpan(),
            CuSeqlens = d.Cu.DevSpan(),
            CuSeqlensHost = d.Cu.Host,
            Batch = (uint)d.Batch,
            Tokens = (uint)d.Tokens,
            KHeads = Heads,
            VHeads = Heads,
            KDim = Dim,
            VDim = Dim,
            UseInitialState = init ? 1u : 0u,
            L2NormQk = 1,
            Scale = 0,
        };
        if (d.Strided)
        {
            p.Q = View(d.Fused, 0, d.Tokens, true);
            p.K = View(d.Fused, Row, d.Tokens, true);
            p.V = View(d.Fused, 2 * Row, d.Tokens, true);
            p.QStride = p.KStride = p.VStride = Fused;
        }
        else
        {
            p.Q = d.Q!.DevSpan();
            p.K = d.K!.DevSpan();
            p.V = d.V!.DevSpan();
        }

        return p;
    }

    private static GatedDeltaRuleArgs CpuArgs(Data d, bool init)
    {
        var a = new GatedDeltaRuleArgs
        {
            G = d.G.HostSpan(),
            Beta = d.Beta.HostSpan(),
            Output = d.Output.HostSpan(),
            State = d.Pool.HostSpan(),
            StateSlots = d.Slots.HostSpan(),
            CuSeqlens = d.Cu.HostSpan(),
            Batch = d.Batch,
            Tokens = d.Tokens,
            KHeads = Heads,
            VHeads = Heads,
            KDim = Dim,
            VDim = Dim,
            UseInitialState = init,
            L2NormQk = true,
            Scale = 0,
            DType = DType.BF16,
        };
        if (d.Strided)
        {
            a.Q = View(d.Fused, 0, d.Tokens, false);
            a.K = View(d.Fused, Row, d.Tokens, false);
            a.V = View(d.Fused, 2 * Row, d.Tokens, false);
            a.QStride = a.KStride = a.VStride = Fused;
        }
        else
        {
            a.Q = d.Q!.HostSpan();
            a.K = d.K!.HostSpan();
            a.V = d.V!.HostSpan();
        }

        return a;
    }

    /// <summary>Allocates the workspaces the entry asks for, plans (when it needs a plan), runs and synchronises.</summary>
    private static void Execute(Library lib, in OpEntry e, in GatedDeltaRuleParams p, nint stream)
    {
        var sizes = lib.QueryWorkspace(e, p);
        var ws = new Workspace { StructSize = (uint)sizeof(Workspace), AbiMinor = 1 };
        nint dev = 0, pinned = 0, plan = 0;
        try
        {
            if (sizes.DeviceBytes > 0)
            {
                dev = Cuda.Malloc((long)sizes.DeviceBytes);
                ws.Device = dev;
                ws.DeviceBytes = sizes.DeviceBytes;
            }

            if (sizes.PinnedHostBytes > 0)
            {
                pinned = Cuda.MallocHost((long)sizes.PinnedHostBytes);
                ws.PinnedHost = pinned;
                ws.PinnedHostBytes = sizes.PinnedHostBytes;
            }

            if (sizes.PlanBytes > 0)
            {
                plan = (nint)NativeMemory.AllocZeroed((nuint)sizes.PlanBytes);
                ws.Plan = plan;
                ws.PlanBytes = sizes.PlanBytes;
            }

            if (e.Plan != 0)
            {
                lib.Plan(e, p, &ws, stream);
            }

            lib.Run(e, p, &ws, stream);
            Cuda.Sync(stream);
        }
        finally
        {
            Cuda.Sync(stream);
            if (dev != 0)
            {
                Cuda.Free(dev);
            }

            if (pinned != 0)
            {
                Cuda.FreeHost(pinned);
            }

            if (plan != 0)
            {
                NativeMemory.Free((void*)plan);
            }
        }
    }

    private static void Case(Library lib, Report r, nint stream, in OpEntry e, string name, int[] lens, bool strided, bool init, double decay, int seed)
    {
        const int PoolSlots = 41;
        var recurrent = !e.Flags.HasFlag(VariantTraits.NeedsPlan);
        using var d = Make(lens, strided, decay, seed, PoolSlots);
        foreach (var b in new[] { d.Fused, d.Q, d.K, d.V, d.G, d.Beta, d.Pool, d.Slots, d.Cu })
        {
            b?.ToDevice();
        }

        Cuda.Fill(d.Output.Device, 0xFF, d.Output.Bytes); // NaN: an element the kernels do not write shows up as a mismatch
        var before = d.Pool.HostBytes();

        CpuOps.GatedDeltaRule(CpuArgs(d, init));
        Execute(lib, e, GpuParams(d, init), stream);

        var got = d.Output.FromDevice();
        var want = d.Output.HostBytes();
        double refMax = 0;
        for (long i = 0; i < d.Output.Elements; i++)
        {
            refMax = Math.Max(refMax, Math.Abs(d.Output.At(want, i)));
        }

        var eo = Compare.Errors(i => d.Output.At(got, i), i => d.Output.At(want, i), d.Output.Elements, Math.Max(refMax, 1e-30));
        var gotPool = d.Pool.FromDevice();
        var wantPool = d.Pool.HostBytes();
        double stateMax = 0;
        for (long i = 0; i < d.Pool.Elements; i++)
        {
            stateMax = Math.Max(stateMax, Math.Abs(d.Pool.At(wantPool, i)));
        }

        // Slots no sequence owns, and the slots of empty sequences, must be bit-identical to what they were.
        var owned = new HashSet<int>();
        var slots = new int[d.Batch];
        Marshal.Copy(d.Slots.Host, slots, 0, d.Batch);
        var cuHost = new int[d.Batch + 1];
        Marshal.Copy(d.Cu.Host, cuHost, 0, d.Batch + 1);
        for (int b = 0; b < d.Batch; b++)
        {
            if (cuHost[b + 1] > cuHost[b])
            {
                owned.Add(slots[b]);
            }
        }

        bool untouched = true;
        for (int s = 0; s < PoolSlots && untouched; s++)
        {
            if (!owned.Contains(s))
            {
                untouched = gotPool.AsSpan((int)(s * StateElems * 4), (int)(StateElems * 4)).SequenceEqual(before.AsSpan((int)(s * StateElems * 4), (int)(StateElems * 4)));
            }
        }

        var es = Compare.Errors(i => d.Pool.At(gotPool, i), i => d.Pool.At(wantPool, i), d.Pool.Elements, Math.Max(stateMax, 1e-30));
        double otol = recurrent ? RecurrentOutputTol : ChunkOutputTol;
        double stol = recurrent ? RecurrentStateTol : ChunkStateTol;
        bool ok = eo.MaxScaled <= otol && es.MaxScaled <= stol && untouched;
        r.Check(ok, $"gdn/{name}",
            $"tokens={d.Tokens} batch={d.Batch} o: max_abs={eo.MaxAbs:E2} rel_to_max={eo.MaxScaled:E2} (tol {otol:E1}, ref max {refMax:F3}); state: max_abs={es.MaxAbs:E2} rel_to_max={es.MaxScaled:E2} (tol {stol:E1}, ref max {stateMax:F3}); other slots untouched={untouched}");
    }

    // A sequence's output and state must not depend on the other sequences in the batch (the BATCH_INVARIANT flag): run it
    // alone and as part of a batch, in the same slot, and compare bit for bit.
    private static void BatchInvariance(Library lib, Report r, nint stream, in OpEntry e, string label)
    {
        const int PoolSlots = 9;
        int[] lens = label == "chunk" ? [333, 200, 129] : [1, 1, 1];
        using var many = Make(lens, true, 0.2, 31, PoolSlots, [4, 7, 2]);
        // The same first sequence, alone: copy its rows, g, beta and state.
        int n0 = lens[0];
        using var one = Make([n0], true, 0.2, 31, PoolSlots, [4]);
        Buffer.MemoryCopy((void*)many.Fused.Host, (void*)one.Fused.Host, one.Fused.Bytes, (long)n0 * Fused * 2);
        Buffer.MemoryCopy((void*)many.G.Host, (void*)one.G.Host, one.G.Bytes, (long)n0 * Heads * 4);
        Buffer.MemoryCopy((void*)many.Beta.Host, (void*)one.Beta.Host, one.Beta.Bytes, (long)n0 * Heads * 2);
        Buffer.MemoryCopy((void*)many.Pool.Host, (void*)one.Pool.Host, one.Pool.Bytes, many.Pool.Bytes);
        foreach (var d in new[] { many, one })
        {
            foreach (var b in new[] { d.Fused, d.G, d.Beta, d.Pool, d.Slots, d.Cu })
            {
                b.ToDevice();
            }

            Cuda.Fill(d.Output.Device, 0xFF, d.Output.Bytes);
            Execute(lib, e, GpuParams(d, true), stream);
        }

        var outMany = many.Output.FromDevice().AsSpan(0, n0 * Row * 2).ToArray();
        var outOne = one.Output.FromDevice();
        var stateMany = many.Pool.FromDevice().AsSpan(4 * (int)StateElems * 4, (int)StateElems * 4).ToArray();
        var stateOne = one.Pool.FromDevice().AsSpan(4 * (int)StateElems * 4, (int)StateElems * 4).ToArray();
        bool same = outMany.AsSpan().SequenceEqual(outOne) && stateMany.AsSpan().SequenceEqual(stateOne);
        r.Check(same, $"gdn/batch-invariance/{label}", $"sequence 0 alone ({n0} tokens) equals sequence 0 in a batch of {lens.Length}, output and state bit-identical={same}");
    }

    // Arguments the library must refuse before it launches anything, with the typed status the contract names.
    private static void Rejections(Library lib, Report r, nint stream, in OpEntry chunk, in OpEntry recurrent)
    {
        using var d = Make([2, 3], true, 0.2, 41, 4, [1, 3]);
        foreach (var b in new[] { d.Fused, d.G, d.Beta, d.Pool, d.Slots, d.Cu })
        {
            b.ToDevice();
        }

        var good = GpuParams(d, true);
        Expect(lib, r, recurrent, "stride below the row width", Mutate(good, (ref GatedDeltaRuleParams p) => { p.QStride = Row - 8; }), 1);
        Expect(lib, r, recurrent, "view span shorter than its reach", Mutate(good, (ref GatedDeltaRuleParams p) => { p.V.Length -= 1; }), 1);
        Expect(lib, r, recurrent, "8 key heads", Mutate(good, (ref GatedDeltaRuleParams p) => { p.KHeads = 8; p.VHeads = 8; }), 2);
        Expect(lib, r, recurrent, "head size 64", Mutate(good, (ref GatedDeltaRuleParams p) => { p.KDim = 64; p.VDim = 64; }), 2);
        Expect(lib, r, recurrent, "l2_norm_qk = 0", Mutate(good, (ref GatedDeltaRuleParams p) => { p.L2NormQk = 0; }), 2);
        Expect(lib, r, recurrent, "f32 beta", Mutate(good, (ref GatedDeltaRuleParams p) => { p.Beta.Dtype = DType.F32; }), 2);
        Expect(lib, r, recurrent, "g not 16-byte aligned", Mutate(good, (ref GatedDeltaRuleParams p) => { p.G.Address += 4; }), 1);
        Expect(lib, r, recurrent, "struct_size too small", Mutate(good, (ref GatedDeltaRuleParams p) => { p.StructSize = 64; }), 5);
        Expect(lib, r, recurrent, "state pool of a partial slot", Mutate(good, (ref GatedDeltaRuleParams p) => { p.State.Length = (ulong)(StateElems + 5); }), 1);
        // Chunk variant: run() without plan() is refused with NOT_PLANNED (status 8).
        var sizes = lib.QueryWorkspace(chunk, good);
        var dev = Cuda.Malloc((long)sizes.DeviceBytes);
        var plan = (nint)NativeMemory.AllocZeroed((nuint)sizes.PlanBytes);
        var ws = new Workspace { StructSize = (uint)sizeof(Workspace), AbiMinor = 1, Device = dev, DeviceBytes = sizes.DeviceBytes, Plan = plan, PlanBytes = sizes.PlanBytes };
        var rc = lib.RunRaw(chunk, good, &ws, stream);
        r.Check(rc == 8, "gdn/reject/run without plan", $"status {rc} (want 8 NOT_PLANNED): {lib.LastError()}");
        var small = ws;
        small.DeviceBytes = sizes.DeviceBytes - 256;
        rc = lib.RunRaw(chunk, good, &small, stream);
        r.Check(rc == 7, "gdn/reject/workspace too small", $"status {rc} (want 7 WORKSPACE_TOO_SMALL): {lib.LastError()}");
        Cuda.Free(dev);
        NativeMemory.Free((void*)plan);
    }

    private delegate void Edit(ref GatedDeltaRuleParams p);

    private static GatedDeltaRuleParams Mutate(GatedDeltaRuleParams p, Edit change)
    {
        change(ref p);
        return p;
    }

    private static void Expect(Library lib, Report r, in OpEntry e, string what, GatedDeltaRuleParams p, int status)
    {
        // The recurrent entry has no plan and a workspace; validation runs first, so a NULL workspace is enough to see the status.
        var rc = lib.RunRaw(e, p, null, 0);
        r.Check(rc == status, $"gdn/reject/{what}", $"status {rc} (want {status}): {lib.LastError()}");
    }
}
