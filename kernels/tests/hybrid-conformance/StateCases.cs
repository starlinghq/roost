using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.HybridConformance;

/// <summary>CausalConv1d, RecurrentStateCopy and RecurrentStateSnapshot against CpuOps.</summary>
internal static unsafe class StateCases
{
    public static void Run(Library lib, Report r, nint stream)
    {
        // Qwen3.5-0.8B conv: 6144 channels, width 4, activations bf16, SiLU, no bias.
        Conv(lib, r, stream, "conv/qwen35 prefill packed", DType.BF16, DType.BF16, [37, 1, 2, 130, 0, 5], 6144, 4, bias: false, silu: true, slots: 9, order: [7, 2, 0, 5, 8, 1], seed: 1);
        Conv(lib, r, stream, "conv/qwen35 decode 8 x 1 token", DType.BF16, DType.BF16, [1, 1, 1, 1, 1, 1, 1, 1], 6144, 4, bias: false, silu: true, slots: 10, order: [9, 3, 1, 8, 0, 4, 2, 6], seed: 2);
        Conv(lib, r, stream, "conv/qwen35 one 2-token sequence", DType.BF16, DType.BF16, [2], 6144, 4, bias: false, silu: true, slots: 2, order: [1], seed: 3);
        Conv(lib, r, stream, "conv/qwen35 one 1-token sequence", DType.BF16, DType.BF16, [1], 6144, 4, bias: false, silu: true, slots: 2, order: [0], seed: 4);
        Conv(lib, r, stream, "conv/bf16 act f32 state, bias, no silu", DType.BF16, DType.F32, [17, 3, 1], 1000, 4, bias: true, silu: false, slots: 5, order: [4, 0, 2], seed: 5);
        Conv(lib, r, stream, "conv/f32 width 7", DType.F32, DType.F32, [40, 2, 9], 333, 7, bias: true, silu: true, slots: 4, order: [3, 1, 0], seed: 6);
        Conv(lib, r, stream, "conv/f32 width 16", DType.F32, DType.F32, [64, 1, 20], 130, 16, bias: false, silu: false, slots: 3, order: [2, 0, 1], seed: 7);
        // More tokens than the 65535 a grid dimension y allows: one long prefill in a single call.
        Conv(lib, r, stream, "conv/one sequence of 70000 tokens", DType.BF16, DType.BF16, [70000], 64, 4, bias: false, silu: true, slots: 2, order: [1], seed: 9);
        Conv(lib, r, stream, "conv/f32 width 1 (no state)", DType.F32, DType.F32, [5, 3], 64, 1, bias: true, silu: true, slots: 2, order: [1, 0], seed: 8);

        Copy(lib, r, stream);
        Snapshot(lib, r, stream, restore: false);
        Snapshot(lib, r, stream, restore: true);
    }

    private static void Conv(Library lib, Report r, nint stream, string name, DType act, DType state, int[] lens, int channels, int width, bool bias, bool silu, int slots, int[] order, int seed)
    {
        var entry = lib.Find("causal-conv1d-cuda")!.Value;
        var rng = new Random(seed);
        int tokens = lens.Sum();
        int batch = lens.Length;
        float[] Rand(long n, double s) => Enumerable.Range(0, (int)n).Select(_ => (float)(((rng.NextDouble() * 2) - 1) * s)).ToArray();
        Buf Make(DType t, float[] v) => t == DType.BF16 ? Buf.FromBf16(v) : Buf.FromF32(v);

        using var input = Make(act, Rand((long)tokens * channels, 1.5));
        using var weight = Make(act, Rand((long)channels * width, 0.6));
        using var biasBuf = bias ? Make(act, Rand(channels, 0.3)) : null;
        using var output = new Buf(act, (long)tokens * channels);
        using var pool = Make(state, Rand((long)slots * channels * Math.Max(width - 1, 1), 1.0));
        using var slotBuf = Buf.FromI32(order);
        var cu = new int[batch + 1];
        for (int i = 0; i < batch; i++)
        {
            cu[i + 1] = cu[i] + lens[i];
        }

        using var cuBuf = Buf.FromI32(cu);
        foreach (var b in new[] { input, weight, pool, slotBuf, cuBuf }.Concat(biasBuf is null ? [] : [biasBuf]))
        {
            b.ToDevice();
        }

        Cuda.Fill(output.Device, 0xFF, output.Bytes);

        var cpu = new CausalConv1dArgs
        {
            Input = input.HostSpan(),
            Weight = weight.HostSpan(),
            Bias = biasBuf?.HostSpan() ?? default,
            Output = output.HostSpan(),
            ConvState = pool.HostSpan(),
            StateSlots = slotBuf.HostSpan(),
            CuSeqlens = cuBuf.HostSpan(),
            Batch = batch,
            Tokens = tokens,
            Channels = channels,
            Width = width,
            Silu = silu,
            DType = act,
        };
        CpuOps.CausalConv1d(cpu);

        var gpu = new CausalConv1dParams
        {
            StructSize = (uint)sizeof(CausalConv1dParams),
            AbiMinor = 1,
            Input = input.DevSpan(),
            Weight = weight.DevSpan(),
            Bias = biasBuf?.DevSpan() ?? default,
            Output = output.DevSpan(),
            ConvState = pool.DevSpan(),
            StateSlots = slotBuf.DevSpan(),
            CuSeqlens = cuBuf.DevSpan(),
            Batch = (uint)batch,
            Tokens = (uint)tokens,
            Channels = (uint)channels,
            Width = (uint)width,
            Silu = silu ? 1u : 0u,
        };
        lib.Run(entry, gpu, null, stream);
        Cuda.Sync(stream);

        var got = output.FromDevice();
        var want = output.HostBytes();
        bool bf = act == DType.BF16;
        // Reference is double rounded once to the activation type; the kernel computes in float and rounds once, so bf16
        // outputs may differ by one bf16 step (2^-8 relative) and f32 outputs by float rounding of a short sum.
        double tol = bf ? Math.Pow(2, -7) : 2e-5;
        var e = Compare.Errors(i => output.At(got, i), i => output.At(want, i), output.Elements, bf ? 1e-3 : 0.1);
        var ok = e.MaxScaled <= tol;
        var stateOk = pool.FromDevice().AsSpan().SequenceEqual(pool.HostBytes());
        r.Check(ok && stateOk, name, $"tokens={tokens} batch={batch} channels={channels} width={width} out max_abs={e.MaxAbs:E2} max_scaled={e.MaxScaled:E2} (tol {tol:E2}) state_bit_exact={stateOk}");
    }

    private static void Copy(Library lib, Report r, nint stream)
    {
        var entry = lib.Find("recurrent-state-copy-cuda")!.Value;
        // Slots of 1 MiB (the Qwen3.5 state: 16 heads x 128 x 128 x f32), 12 bytes and 7 bytes (odd sizes take the narrow paths).
        foreach (var slotBytes in new[] { 16 * 128 * 128 * 4, 12, 7 })
        {
            const int Slots = 8;
            var rng = new Random(slotBytes);
            using var pool = new Buf(DType.F32, ((long)Slots * slotBytes / 4) + 4);
            var span = new Span<byte>((void*)pool.Host, Slots * slotBytes);
            rng.NextBytes(span);
            // Pairs include a chain (0 -> 1 then 1 -> 2: slot 2 must end with slot 0's old contents), a self pair and a fan-out.
            int[] pairs = [0, 1, 1, 2, 3, 3, 4, 5, 4, 6, 7, 0];
            using var pairBuf = Buf.FromI32(pairs);
            pool.ToDevice();
            pairBuf.ToDevice();
            CpuOps.RecurrentStateCopy(new RecurrentStateCopyArgs { Pool = pool.HostSpan(), Pairs = pairBuf.HostSpan(), SlotBytes = slotBytes, Count = pairs.Length / 2 });
            lib.Run(entry, new RecurrentStateCopyParams { StructSize = (uint)sizeof(RecurrentStateCopyParams), AbiMinor = 1, Pool = pool.DevSpan(), Pairs = pairBuf.DevSpan(), SlotBytes = (ulong)slotBytes, Count = (uint)(pairs.Length / 2) }, null, stream);
            Cuda.Sync(stream);
            bool ok = pool.FromDevice().AsSpan().SequenceEqual(pool.HostBytes());
            r.Check(ok, $"state-copy/slot_bytes={slotBytes}", $"pairs={pairs.Length / 2} chain 0>1>2 included; bit_exact={ok}");
        }
    }

    private static void Snapshot(Library lib, Report r, nint stream, bool restore)
    {
        var entry = lib.Find("recurrent-state-snapshot-cuda")!.Value;
        foreach (var slotBytes in new[] { 16 * 128 * 128 * 4, 12 })
        {
            const int Slots = 6;
            const int Snaps = 5;
            var rng = new Random(slotBytes + (restore ? 1 : 0));
            using var pool = new Buf(DType.F32, (long)Slots * slotBytes / 4);
            using var snaps = new Buf(DType.F32, (long)Snaps * slotBytes / 4);
            rng.NextBytes(new Span<byte>((void*)pool.Host, (int)pool.Bytes));
            rng.NextBytes(new Span<byte>((void*)snaps.Host, (int)snaps.Bytes));
            // (slot, snapshot) pairs; a snapshot written twice keeps the later pair.
            int[] pairs = [5, 0, 2, 1, 0, 4, 3, 1];
            using var pairBuf = Buf.FromI32(pairs);
            pool.ToDevice();
            snaps.ToDevice();
            pairBuf.ToDevice();
            CpuOps.RecurrentStateSnapshot(new RecurrentStateSnapshotArgs { Pool = pool.HostSpan(), Snapshots = snaps.HostSpan(), Pairs = pairBuf.HostSpan(), SlotBytes = slotBytes, Count = pairs.Length / 2, Restore = restore });
            lib.Run(entry, new RecurrentStateSnapshotParams { StructSize = (uint)sizeof(RecurrentStateSnapshotParams), AbiMinor = 1, Pool = pool.DevSpan(), Snapshots = snaps.DevSpan(), Pairs = pairBuf.DevSpan(), SlotBytes = (ulong)slotBytes, Count = (uint)(pairs.Length / 2), Restore = restore ? 1u : 0u }, null, stream);
            Cuda.Sync(stream);
            bool ok = pool.FromDevice().AsSpan().SequenceEqual(pool.HostBytes()) && snaps.FromDevice().AsSpan().SequenceEqual(snaps.HostBytes());
            r.Check(ok, $"state-snapshot/{(restore ? "restore" : "save")}/slot_bytes={slotBytes}", $"pairs={pairs.Length / 2}; bit_exact={ok}");
        }
    }
}
