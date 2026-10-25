using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// The cases of the hybrid ops (R-034, contract version 3): GatedDeltaRule (one case set for both the chunked and the recurrent
/// variant), CausalConv1d, RecurrentStateCopy, RecurrentStateSnapshot, GatedDeltaRuleGates, GatedRmsNorm and SigmoidMul. Shapes are the
/// Qwen3.5-0.8B linear-attention layer (config.json at 2fc06364715b967f1860aea9cf38778875588b17: 16 key and value heads of 128, the
/// conv1d over the 6144-wide fused q, k, v row with kernel width 4, 8 attention heads of 256 behind the output gate, rms_norm_eps
/// 1e-6) plus the other dtypes and widths the variants register. The fixed shapes follow the R-034 harness
/// (<c>kernels/tests/hybrid-conformance</c>); values are drawn from the case seed like every other case of the suite.
/// </summary>
public static partial class CaseCatalog
{
    private const int GdnHeads = 16, GdnDim = 128, GdnRow = GdnHeads * GdnDim, GdnFused = 3 * GdnRow;
    private const long GdnStateElems = (long)GdnHeads * GdnDim * GdnDim;

    /// <summary>The variant id of the chunked GatedDeltaRule, whose outputs have their own classes (bf16 intermediates of FLA's chunk path).</summary>
    public const string GdnChunkVariant = "gated-delta-rule-chunk-aot";

    private static IEnumerable<OpCase> HybridCases(OpId op, ulong baseSeed, CaseProfile p) => op switch
    {
        OpId.GatedDeltaRule => GdnCases(baseSeed, p),
        OpId.CausalConv1d => ConvCases(baseSeed, p),
        OpId.RecurrentStateCopy => StateCopyCases(baseSeed, p),
        OpId.RecurrentStateSnapshot => SnapshotCases(baseSeed, p),
        OpId.GatedDeltaRuleGates => GatesCases(baseSeed, p),
        OpId.GatedRmsNorm => GatedNormCases(baseSeed, p),
        OpId.SigmoidMul => SigmoidMulCases(baseSeed, p),
        _ => throw new NotSupportedException($"No cases for {op}."),
    };

    private static HostBuf Uniform(DType dt, long n, Rng rng, double lo, double hi)
    {
        var b = new HostBuf(dt, n);
        for (long i = 0; i < n; i++)
        {
            b.Set(i, lo + ((hi - lo) * rng.Uniform()));
        }

        return b;
    }

    private static int[] CuSeqlens(int[] lens)
    {
        var cu = new int[lens.Length + 1];
        for (int i = 0; i < lens.Length; i++)
        {
            cu[i + 1] = cu[i] + lens[i];
        }

        return cu;
    }

    /// <summary>Distinct pool slots for <paramref name="batch"/> sequences out of <paramref name="poolSlots"/>, in a seeded order.</summary>
    private static int[] SlotOrder(Rng rng, int batch, int poolSlots)
    {
        int[] all = Enumerable.Range(0, poolSlots).ToArray();
        for (int i = poolSlots - 1; i > 0; i--)
        {
            int j = rng.Int(0, i + 1);
            (all[i], all[j]) = (all[j], all[i]);
        }

        return all[..batch];
    }

    private static string Lens(int[] lens) => lens.Length <= 6 ? string.Join('.', lens) : $"{lens.Length}x{lens[0]}";

    // ---- GatedDeltaRule -----------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> GdnCases(ulong baseSeed, CaseProfile p)
    {
        if (p.Large)
        {
            yield return Gdn("fixed", [300, 77, 1500, 64, 1], strided: true, init: true, decay: 0.2, baseSeed);
            yield return Gdn("fixed", [300, 77, 1500], strided: true, init: false, decay: 0.2, baseSeed);
            yield return Gdn("fixed", [1000], strided: false, init: true, decay: 0.2, baseSeed);
            yield return Gdn("fixed", [200], strided: true, init: true, decay: 2.5, baseSeed);
            yield return Gdn("fixed", Enumerable.Repeat(1, 32).ToArray(), strided: true, init: true, decay: 0.2, baseSeed);
        }

        yield return Gdn("fixed", [64, 128, 65, 1, 0, 63], strided: true, init: true, decay: 0.2, baseSeed);
        yield return Gdn("fixed", [70, 40], strided: true, init: true, decay: 2.5, baseSeed);
        yield return Gdn("fixed", [1, 1, 1, 1], strided: true, init: true, decay: 0.2, baseSeed);
        yield return Gdn("fixed", [5, 1, 1], strided: false, init: true, decay: 0.2, baseSeed);
        yield return Gdn("fixed", [3], strided: true, init: false, decay: 0.2, baseSeed);
        yield return Gdn("fixed", [1, 0, 1], strided: true, init: true, decay: 0.2, baseSeed);
        var rng = new Rng(SeedFor(OpId.GatedDeltaRule, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int batch = rng.Int(1, 5);
            int[] lens = [.. Enumerable.Range(0, batch).Select(_ => rng.Int(0, 4) == 0 ? rng.Int(0, 3) : rng.Int(1, p.Large ? 400 : 48))];
            if (lens.Sum() == 0)
            {
                lens[0] = 1;
            }

            yield return Gdn($"random-{i}", lens, strided: rng.Int(0, 3) != 0, init: rng.Int(0, 4) != 0, decay: rng.Pick(0.2, 1.0, 2.5), baseSeed);
        }
    }

    private static OpCase Gdn(string family, int[] lens, bool strided, bool init, double decay, ulong baseSeed)
    {
        int tokens = lens.Sum(), batch = lens.Length, poolSlots = batch + 2;
        string id = $"{family}-BF16-{Lens(lens)}-{(strided ? "views" : "dense")}{(init ? "-init" : string.Empty)}-g{decay}";
        ulong seed = SeedFor(OpId.GatedDeltaRule, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase
        {
            Id = id,
            Op = OpId.GatedDeltaRule,
            Dtype = DType.BF16,
            ToleranceClass = "gdn-recurrent-bf16",
            Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
        };

        // The fused conv1d output row [tokens, 6144] holds the q, k and v columns; a dense case splits it into three tensors.
        var fused = new HostBuf(DType.BF16, (long)tokens * GdnFused).Randn(rng, 1.0);
        if (strided)
        {
            c.Buffers["qkv"] = fused;
        }
        else
        {
            c.Buffers["q"] = fused.Compact(tokens, GdnRow, GdnFused);
            c.Buffers["k"] = Columns(fused, tokens, GdnRow);
            c.Buffers["v"] = Columns(fused, tokens, 2 * GdnRow);
            fused.Dispose();
        }

        c.Buffers["g"] = Uniform(DType.F32, (long)tokens * GdnHeads, rng, -decay, 0);
        c.Buffers["beta"] = Uniform(DType.BF16, (long)tokens * GdnHeads, rng, 0, 1);
        c.Buffers["state"] = new HostBuf(DType.F32, poolSlots * GdnStateElems).Randn(rng, 0.1);
        c.Buffers["slots"] = HostBuf.Ints(SlotOrder(rng, batch, poolSlots));
        c.Buffers["cu"] = HostBuf.Ints(CuSeqlens(lens));
        c.Buffers["out"] = new HostBuf(DType.BF16, (long)tokens * GdnRow);
        c.Num["tokens"] = tokens;
        c.Num["batch"] = batch;
        c.Num["strided"] = strided ? 1 : 0;
        c.Num["init"] = init ? 1 : 0;
        c.Num["head_dim"] = GdnDim;
        c.Num["q_heads"] = GdnHeads;
        c.Num["kv_heads"] = GdnHeads;
        c.Num["seq_row_width"] = GdnRow;
        c.Num["slot_elems"] = GdnStateElems;
        c.Outputs.Add("out");
        c.Outputs.Add("state");
        c.OutputClasses["state"] = "gdn-recurrent-state-f32";
        c.OutputClasses["out@" + GdnChunkVariant] = "gdn-chunk-bf16";
        c.OutputClasses["state@" + GdnChunkVariant] = "gdn-chunk-state-f32";
        return c;
    }

    private static HostBuf Columns(HostBuf fused, int tokens, int col)
    {
        var o = new HostBuf(fused.Dtype, (long)tokens * GdnRow);
        for (long t = 0; t < tokens; t++)
        {
            fused.Raw().Slice((int)(((t * GdnFused) + col) * 2), GdnRow * 2).CopyTo(o.Raw().Slice((int)(t * GdnRow * 2), GdnRow * 2));
        }

        return o;
    }

    // ---- CausalConv1d -------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> ConvCases(ulong baseSeed, CaseProfile p)
    {
        // Qwen3.5-0.8B: 6144 channels, width 4, bf16 activations and state, SiLU, no bias.
        yield return Conv("fixed", DType.BF16, DType.BF16, [37, 1, 2, 130, 0, 5], 6144, 4, bias: false, silu: true, baseSeed);
        yield return Conv("fixed", DType.BF16, DType.BF16, [1, 1, 1, 1, 1, 1, 1, 1], 6144, 4, bias: false, silu: true, baseSeed);
        yield return Conv("fixed", DType.BF16, DType.BF16, [2], 6144, 4, bias: false, silu: true, baseSeed);
        yield return Conv("fixed", DType.BF16, DType.BF16, [1], 6144, 4, bias: false, silu: true, baseSeed);
        yield return Conv("fixed", DType.BF16, DType.F32, [17, 3, 1], 1000, 4, bias: true, silu: false, baseSeed);
        yield return Conv("fixed", DType.F32, DType.F32, [40, 2, 9], 333, 7, bias: true, silu: true, baseSeed);
        yield return Conv("fixed", DType.F32, DType.F32, [64, 1, 20], 130, 16, bias: false, silu: false, baseSeed);
        yield return Conv("fixed", DType.F32, DType.F32, [5, 3], 64, 1, bias: true, silu: true, baseSeed);
        if (p.Large)
        {
            // More tokens than the 65535 a grid dimension y allows, in one call.
            yield return Conv("fixed", DType.BF16, DType.BF16, [70000], 64, 4, bias: false, silu: true, baseSeed);
        }

        var rng = new Rng(SeedFor(OpId.CausalConv1d, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            DType act = rng.Pick(DType.BF16, DType.F32);
            DType state = act == DType.F32 ? DType.F32 : rng.Pick(DType.BF16, DType.F32);
            int batch = rng.Int(1, 6);
            int[] lens = [.. Enumerable.Range(0, batch).Select(_ => rng.Int(0, 3) == 0 ? rng.Int(0, 3) : rng.Int(1, p.Large ? 200 : 40))];
            if (lens.Sum() == 0)
            {
                lens[0] = 1;
            }

            yield return Conv($"random-{i}", act, state, lens, 8 * rng.Int(1, p.Large ? 800 : 100), rng.Int(1, 17), bias: rng.Int(0, 2) == 0, silu: rng.Int(0, 2) == 0, baseSeed);
        }
    }

    private static OpCase Conv(string family, DType act, DType state, int[] lens, int channels, int width, bool bias, bool silu, ulong baseSeed)
    {
        int tokens = lens.Sum(), batch = lens.Length, poolSlots = batch + 2;
        long slotElems = (long)channels * Math.Max(width - 1, 1);
        string id = $"{family}-{act}-state{state}-{Lens(lens)}-c{channels}-w{width}{(bias ? "-bias" : string.Empty)}{(silu ? "-silu" : string.Empty)}";
        ulong seed = SeedFor(OpId.CausalConv1d, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase
        {
            Id = id,
            Op = OpId.CausalConv1d,
            Dtype = act,
            ToleranceClass = act == DType.BF16 ? "elementwise-bf16" : "hybrid-f32",
            Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
        };
        c.Buffers["x"] = Uniform(act, (long)tokens * channels, rng, -1.5, 1.5);
        c.Buffers["w"] = Uniform(act, (long)channels * width, rng, -0.6, 0.6);
        c.Buffers["bias"] = bias ? Uniform(act, channels, rng, -0.3, 0.3) : new HostBuf(act, 1);
        c.Buffers["conv_state"] = Uniform(state, poolSlots * slotElems, rng, -1, 1);
        c.Buffers["slots"] = HostBuf.Ints(SlotOrder(rng, batch, poolSlots));
        c.Buffers["cu"] = HostBuf.Ints(CuSeqlens(lens));
        c.Buffers["out"] = new HostBuf(act, (long)tokens * channels);
        c.Num["tokens"] = tokens;
        c.Num["batch"] = batch;
        c.Num["channels"] = channels;
        c.Num["width"] = width;
        c.Num["use_bias"] = bias ? 1 : 0;
        c.Num["silu"] = silu ? 1 : 0;
        c.Num["seq_row_width"] = channels;
        c.Num["slot_elems"] = slotElems;
        c.Outputs.Add("out");
        c.Outputs.Add("conv_state");
        c.OutputClasses["conv_state"] = "exact";
        return c;
    }

    // ---- RecurrentStateCopy and RecurrentStateSnapshot ----------------------------------------------------------

    private static IEnumerable<OpCase> StateCopyCases(ulong baseSeed, CaseProfile p)
    {
        // Pairs include a chain (0 -> 1 then 1 -> 2: slot 2 ends with slot 0's old contents), a self pair and a fan-out. Slots of
        // 1 MiB (the Qwen3.5 recurrent state: 16 x 128 x 128 f32), 12 bytes and 7 bytes (odd sizes take the narrow paths).
        foreach (int slotBytes in new[] { 16 * 128 * 128 * 4, 12, 7 })
        {
            yield return StateCopy("fixed", slotBytes, 8, [0, 1, 1, 2, 3, 3, 4, 5, 4, 6, 7, 0], baseSeed);
        }

        var rng = new Rng(SeedFor(OpId.RecurrentStateCopy, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int slots = rng.Int(2, 12);
            int count = rng.Int(1, 10);
            int[] pairs = [.. Enumerable.Range(0, 2 * count).Select(_ => rng.Int(0, slots))];
            yield return StateCopy($"random-{i}", rng.Pick(4, 9, 64, 4096, 65536 + 3), slots, pairs, baseSeed);
        }
    }

    private static OpCase StateCopy(string family, int slotBytes, int slots, int[] pairs, ulong baseSeed)
    {
        string id = $"{family}-slot{slotBytes}-n{slots}-p{pairs.Length / 2}";
        ulong seed = SeedFor(OpId.RecurrentStateCopy, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.RecurrentStateCopy, Dtype = DType.U8, ToleranceClass = "exact", Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed", Seed = seed };
        c.Buffers["pool"] = RandomBytes(rng, (long)slots * slotBytes);
        c.Buffers["pairs"] = HostBuf.Ints(pairs);
        c.Num["slot_bytes"] = slotBytes;
        c.Num["count"] = pairs.Length / 2;
        c.Outputs.Add("pool");
        return c;
    }

    private static IEnumerable<OpCase> SnapshotCases(ulong baseSeed, CaseProfile p)
    {
        // (slot, snapshot) pairs; a snapshot written twice keeps the later pair.
        foreach (bool restore in new[] { false, true })
        {
            foreach (int slotBytes in new[] { 16 * 128 * 128 * 4, 12 })
            {
                yield return Snapshot("fixed", slotBytes, 6, 5, [5, 0, 2, 1, 0, 4, 3, 1], restore, baseSeed);
            }
        }

        var rng = new Rng(SeedFor(OpId.RecurrentStateSnapshot, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            int slots = rng.Int(1, 10), snaps = rng.Int(1, 10), count = rng.Int(1, 8);
            bool restore = rng.Int(0, 2) == 0;
            // A restore writes slots: one pair per slot, so the result does not depend on the order of the writes.
            count = restore ? Math.Min(count, slots) : count;
            int[] slotPick = restore ? SlotOrder(rng, count, slots) : [.. Enumerable.Range(0, count).Select(_ => rng.Int(0, slots))];
            int[] snapPick = restore ? [.. Enumerable.Range(0, count).Select(_ => rng.Int(0, snaps))] : SlotOrder(rng, Math.Min(count, snaps), snaps);
            count = Math.Min(slotPick.Length, snapPick.Length);
            int[] pairs = [.. Enumerable.Range(0, count).SelectMany(k => new[] { slotPick[k], snapPick[k] })];
            yield return Snapshot($"random-{i}", rng.Pick(4, 9, 64, 4096, 65536 + 3), slots, snaps, pairs, restore, baseSeed);
        }
    }

    private static OpCase Snapshot(string family, int slotBytes, int slots, int snaps, int[] pairs, bool restore, ulong baseSeed)
    {
        string id = $"{family}-{(restore ? "restore" : "save")}-slot{slotBytes}-n{slots}-s{snaps}-p{pairs.Length / 2}";
        ulong seed = SeedFor(OpId.RecurrentStateSnapshot, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.RecurrentStateSnapshot, Dtype = DType.U8, ToleranceClass = "exact", Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed", Seed = seed };
        c.Buffers["pool"] = RandomBytes(rng, (long)slots * slotBytes);
        c.Buffers["snapshots"] = RandomBytes(rng, (long)snaps * slotBytes);
        c.Buffers["pairs"] = HostBuf.Ints(pairs);
        c.Num["slot_bytes"] = slotBytes;
        c.Num["count"] = pairs.Length / 2;
        c.Num["restore"] = restore ? 1 : 0;
        c.Outputs.Add("pool");
        c.Outputs.Add("snapshots");
        return c;
    }

    private static HostBuf RandomBytes(Rng rng, long n)
    {
        var b = new HostBuf(DType.U8, n);
        Span<byte> raw = b.Raw();
        for (int i = 0; i < raw.Length; i++)
        {
            raw[i] = (byte)rng.NextU64();
        }

        return b;
    }

    // ---- GatedDeltaRuleGates ------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> GatesCases(ulong baseSeed, CaseProfile p)
    {
        // Qwen3.5: a and b bf16 (in_proj_a, in_proj_b outputs), A_log f32, dt_bias bf16, g f32, beta bf16.
        DType[] qwen = [DType.BF16, DType.F32, DType.BF16, DType.F32, DType.BF16];
        yield return Gates("fixed", 37, 0, 0, 0, qwen, 6.0, baseSeed);
        yield return Gates("fixed", 300, 32, 0, 16, qwen, 6.0, baseSeed);
        yield return Gates("fixed", 1, 0, 0, 0, qwen, 6.0, baseSeed);
        yield return Gates("fixed-softplus-threshold", 64, 0, 0, 0, qwen, 40.0, baseSeed);
        yield return Gates("fixed", 41, 40, 3, 21, [DType.F32, DType.BF16, DType.F32, DType.BF16, DType.F32], 8.0, baseSeed);
        var rng = new Rng(SeedFor(OpId.GatedDeltaRuleGates, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            DType act = rng.Pick(DType.BF16, DType.F32);
            DType[] types = [act, rng.Pick(DType.BF16, DType.F32), rng.Pick(DType.BF16, DType.F32), rng.Pick(DType.BF16, DType.F32), rng.Pick(DType.BF16, DType.F32)];
            bool fused = rng.Int(0, 2) == 0;
            int stride = fused ? 32 + (2 * rng.Int(0, 8)) : 0;
            int aCol = fused ? 2 * rng.Int(0, ((stride - 32) / 2) + 1) : 0; // even, so bf16 rows stay 4-byte aligned
            yield return Gates($"random-{i}", rng.Int(1, p.Large ? 400 : 60), stride, aCol, fused ? aCol + 16 : 0, types, rng.Pick(3.0, 6.0, 30.0), baseSeed);
        }
    }

    /// <summary>types: a and b, A_log, dt_bias, g, beta. stride 0: a and b are separate [tokens, 16] tensors; otherwise column views of one [tokens, stride] buffer.</summary>
    private static OpCase Gates(string family, int tokens, int stride, int aCol, int bCol, DType[] types, double aScale, ulong baseSeed)
    {
        DType act = types[0];
        string id = $"{family}-{act}-alog{types[1]}-dt{types[2]}-g{types[3]}-beta{types[4]}-t{tokens}{(stride > 0 ? $"-s{stride}-{aCol}-{bCol}" : string.Empty)}";
        ulong seed = SeedFor(OpId.GatedDeltaRuleGates, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.GatedDeltaRuleGates, Dtype = act, ToleranceClass = "hybrid-f32", Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed", Seed = seed };
        if (stride == 0)
        {
            c.Buffers["a"] = Uniform(act, (long)tokens * GdnHeads, rng, -aScale, aScale);
            c.Buffers["b"] = Uniform(act, (long)tokens * GdnHeads, rng, -6, 6);
        }
        else
        {
            var ab = Uniform(act, (long)tokens * stride, rng, -6, 6);
            for (long t = 0; t < tokens; t++)
            {
                for (int h = 0; h < GdnHeads; h++)
                {
                    long i = (t * stride) + aCol + h;
                    ab.Set(i, ab.Get(i) * aScale / 6.0);
                }
            }

            c.Buffers["ab"] = ab;
        }

        c.Buffers["a_log"] = Uniform(types[1], GdnHeads, rng, -3, 3);
        c.Buffers["dt_bias"] = Uniform(types[2], GdnHeads, rng, -3, 3);
        c.Buffers["g"] = new HostBuf(types[3], (long)tokens * GdnHeads);
        c.Buffers["beta"] = new HostBuf(types[4], (long)tokens * GdnHeads);
        c.Num["tokens"] = tokens;
        c.Num["stride"] = stride;
        c.Num["a_col"] = aCol;
        c.Num["b_col"] = bCol;
        c.Outputs.Add("g");
        c.Outputs.Add("beta");
        c.OutputClasses["g"] = types[3] == DType.BF16 ? "elementwise-bf16" : "hybrid-f32";
        c.OutputClasses["beta"] = types[4] == DType.BF16 ? "elementwise-bf16" : "hybrid-f32";
        return c;
    }

    // ---- GatedRmsNorm and SigmoidMul ----------------------------------------------------------------------------

    private static IEnumerable<OpCase> GatedNormCases(ulong baseSeed, CaseProfile p)
    {
        // Qwen3.5 linear-attention output norm: rows = tokens * 16, dim 128, f32 weight.
        yield return GatedNorm("fixed", 37 * GdnHeads, GdnDim, DType.BF16, DType.F32, false, baseSeed);
        yield return GatedNorm("fixed", GdnHeads, GdnDim, DType.BF16, DType.F32, false, baseSeed);
        yield return GatedNorm("fixed-in-place", (p.Large ? 300 : 30) * GdnHeads, GdnDim, DType.BF16, DType.F32, true, baseSeed);
        yield return GatedNorm("fixed", 33, 100, DType.F32, DType.BF16, false, baseSeed);
        yield return GatedNorm("fixed-in-place", 9, 3000, DType.F32, DType.F32, true, baseSeed);
        var rng = new Rng(SeedFor(OpId.GatedRmsNorm, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            yield return GatedNorm($"random-{i}", rng.Int(1, 300), 4 * rng.Int(1, 600), rng.Pick(DType.BF16, DType.F32), rng.Pick(DType.BF16, DType.F32), rng.Int(0, 3) == 0, baseSeed);
        }
    }

    private static OpCase GatedNorm(string family, int rows, int dim, DType act, DType wt, bool inPlace, ulong baseSeed)
    {
        string id = $"{family}-{act}-w{wt}-{Dim(rows, dim)}";
        ulong seed = SeedFor(OpId.GatedRmsNorm, id, baseSeed);
        var rng = new Rng(seed);
        var x = Uniform(act, (long)rows * dim, rng, -2, 2);
        var gate = Uniform(act, (long)rows * dim, rng, -4, 4);
        // linear_attn.norm.weight of layer 0 holds 0.52 to 1.06.
        var w = Uniform(wt, dim, rng, 0.5, 1.1);
        return GatedNormFrom(id, act, rows, dim, inPlace, seed, x, gate, w, true);
    }

    private static OpCase GatedNormFrom(string id, DType act, int rows, int dim, bool inPlace, ulong seed, HostBuf x, HostBuf gate, HostBuf w, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.GatedRmsNorm,
            Dtype = act,
            ToleranceClass = act == DType.BF16 ? "elementwise-bf16" : "hybrid-f32",
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = rows,
            SingleRow = withRows
                ? r => GatedNormFrom(id + $"-row{r}", act, 1, dim, inPlace, seed, CopyRows(x, r, 1, dim), CopyRows(gate, r, 1, dim), w.Clone(), false)
                : null,
        };
        c.Buffers["x"] = x;
        c.Buffers["gate"] = gate;
        c.Buffers["w"] = w;
        c.Buffers["out"] = new HostBuf(act, (long)rows * dim);
        c.Num["rows"] = rows;
        c.Num["dim"] = dim;
        c.Num["eps"] = Eps;
        c.Num["in_place"] = inPlace ? 1 : 0;
        c.Outputs.Add("out");
        c.RowWidth["out"] = dim;
        return c;
    }

    private static IEnumerable<OpCase> SigmoidMulCases(ulong baseSeed, CaseProfile p)
    {
        // Qwen3.5 attention output gate: rows = tokens, dim = 8 heads * 256.
        yield return SigmoidMul("fixed", 37, QHeads * HeadDim, DType.BF16, false, baseSeed);
        yield return SigmoidMul("fixed", 1, QHeads * HeadDim, DType.BF16, false, baseSeed);
        yield return SigmoidMul("fixed-in-place", p.Large ? 300 : 30, QHeads * HeadDim, DType.BF16, true, baseSeed);
        yield return SigmoidMul("fixed", 7, 1001, DType.F32, false, baseSeed);
        var rng = new Rng(SeedFor(OpId.SigmoidMul, "random-shapes", baseSeed));
        for (int i = 0; i < p.RandomPerOp; i++)
        {
            yield return SigmoidMul($"random-{i}", rng.Int(1, 200), rng.Int(1, 3000), rng.Pick(DType.BF16, DType.F32), rng.Int(0, 3) == 0, baseSeed);
        }
    }

    private static OpCase SigmoidMul(string family, int rows, int dim, DType act, bool inPlace, ulong baseSeed)
    {
        string id = $"{family}-{act}-{Dim(rows, dim)}";
        ulong seed = SeedFor(OpId.SigmoidMul, id, baseSeed);
        var rng = new Rng(seed);
        return SigmoidMulFrom(id, act, rows, dim, inPlace, seed, Uniform(act, (long)rows * dim, rng, -3, 3), Uniform(act, (long)rows * dim, rng, -6, 6), true);
    }

    private static OpCase SigmoidMulFrom(string id, DType act, int rows, int dim, bool inPlace, ulong seed, HostBuf x, HostBuf gate, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.SigmoidMul,
            Dtype = act,
            ToleranceClass = act == DType.BF16 ? "elementwise-bf16" : "hybrid-f32",
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = rows,
            SingleRow = withRows
                ? r => SigmoidMulFrom(id + $"-row{r}", act, 1, dim, inPlace, seed, CopyRows(x, r, 1, dim), CopyRows(gate, r, 1, dim), false)
                : null,
        };
        c.Buffers["x"] = x;
        c.Buffers["gate"] = gate;
        c.Buffers["out"] = new HostBuf(act, (long)rows * dim);
        c.Num["rows"] = rows;
        c.Num["dim"] = dim;
        c.Num["in_place"] = inPlace ? 1 : 0;
        c.Outputs.Add("out");
        c.RowWidth["out"] = dim;
        return c;
    }

    /// <summary>
    /// For the sequence ops (GatedDeltaRule, CausalConv1d): the same call on one sequence of a batch, over the same state pool, so
    /// the runner can check that its token rows and its state slot do not depend on the other sequences (the BATCH_INVARIANT flag).
    /// </summary>
    /// <param name="c">A case with <c>cu</c> and <c>slots</c> buffers.</param>
    /// <param name="s">The sequence.</param>
    /// <returns>The one-sequence case.</returns>
    public static OpCase SequenceAlone(OpCase c, int s)
    {
        int[] cu = c.Buffers["cu"].As<int>().ToArray();
        int slot = c.Buffers["slots"].As<int>()[s];
        int tokens = cu[s + 1] - cu[s];
        var one = new OpCase { Id = c.Id + $"-seq{s}", Op = c.Op, Dtype = c.Dtype, ToleranceClass = c.ToleranceClass, Source = c.Source, Seed = c.Seed };
        foreach (KeyValuePair<string, double> kv in c.Num)
        {
            one.Num[kv.Key] = kv.Value;
        }

        one.Num["batch"] = 1;
        one.Num["tokens"] = tokens;
        foreach ((string name, HostBuf b) in c.Buffers)
        {
            switch (name)
            {
                case "cu": one.Buffers[name] = HostBuf.Ints(0, tokens); break;
                case "slots": one.Buffers[name] = HostBuf.Ints(slot); break;
                case "qkv": one.Buffers[name] = CopyRows(b, cu[s], tokens, GdnFused); break;
                case "q" or "k" or "v" or "out": one.Buffers[name] = CopyRows(b, cu[s], tokens, c.I("seq_row_width")); break;
                case "x": one.Buffers[name] = CopyRows(b, cu[s], tokens, c.I("channels")); break;
                case "g" or "beta": one.Buffers[name] = CopyRows(b, cu[s], tokens, GdnHeads); break;
                default: one.Buffers[name] = b.Clone(); break;
            }
        }

        one.Outputs.AddRange(c.Outputs);
        foreach (KeyValuePair<string, string> kv in c.OutputClasses)
        {
            one.OutputClasses[kv.Key] = kv.Value;
        }

        return one;
    }
}
