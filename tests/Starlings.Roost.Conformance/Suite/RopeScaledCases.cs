using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Conformance;

/// <summary>
/// The cases of RopeScaled (R-098): YaRN, longrope (short and long factors) and dynamic NTK caches built by
/// <see cref="RopeScaling"/> in float32, applied to bf16 and f16 Q and K of the Qwen3.5-0.8B full-attention shape (8 query heads
/// over 2 KV heads, head_dim 256, 64 rotary dimensions, theta 1e7) and of a 128-wide full-rotary head. The cache builders are
/// checked against transformers in <c>RopeScalingTests</c>; these cases check how a backend applies a cache.
/// </summary>
/// <remarks>
/// Only the CPU half exists. No kernel library registers a RopeScaled variant, so on CUDA the op is unsupported and untested on
/// every SM; <see cref="GpuHalfOpen"/> makes the runner say so for a backend without a variant instead of omitting the op.
/// </remarks>
public static partial class CaseCatalog
{
    /// <summary>Ops that have cases but whose GPU half is not built: a backend with no variant for one gets a not_run row naming it.</summary>
    public static readonly IReadOnlyDictionary<OpId, string> GpuHalfOpen = new Dictionary<OpId, string>
    {
        [OpId.RopeScaled] = "unsupported on this backend: no RopeScaled variant is registered (the CUDA kernel of R-098 is not built), so the op is untested on every SM",
    };

    // Rows of the cache: positions are drawn below this, which covers the original lengths of the longrope and dynamic families.
    private const int ScaledRopeRows = 8192;

    private sealed record ScaledRopeFamily(string Name, int QHeads, int KvHeads, int HeadDim, int Rotary, int Kind, float Factor, Func<RopeFrequencies> Build);

    private static readonly ScaledRopeFamily[] ScaledRopeFamilies =
    [
        new("qwen35-yarn-f4", 8, 2, 256, 64, 1, 4f, () => RopeScaling.Yarn(new YarnParameters { RotaryDim = 64, Theta = 1e7, Factor = 4, OriginalMaxPosition = 2048 })),
        new("qwen35-yarn-mscale", 8, 2, 256, 64, 1, 40f, () => RopeScaling.Yarn(new YarnParameters { RotaryDim = 64, Theta = 1e7, Factor = 40, OriginalMaxPosition = 1024, Mscale = 0.707, MscaleAllDim = 1.0 })),
        new("hd128-longrope-short", 8, 2, 128, 128, 2, 4f, () => RopeScaling.LongRope(LongRope128(), 2048)),
        new("hd128-longrope-long", 8, 2, 128, 128, 2, 4f, () => RopeScaling.LongRope(LongRope128(), ScaledRopeRows)),
        new("qwen35-dynamic-ntk", 8, 2, 256, 64, 3, 2f, () => RopeScaling.DynamicNtk(64, 1e7, 2, 2048, ScaledRopeRows)),
        new("hd128-dynamic-ntk", 16, 8, 128, 0, 3, 4f, () => RopeScaling.DynamicNtk(128, 1e4, 4, 2048, ScaledRopeRows)),
    ];

    // Synthetic factors that rise across the pairs, as published longrope factors do; not taken from any checkpoint.
    private static LongRopeParameters LongRope128() => new()
    {
        RotaryDim = 128,
        Theta = 1e4,
        ShortFactor = [.. Enumerable.Range(0, 64).Select(i => 1.0 + (0.25 * i / 63))],
        LongFactor = [.. Enumerable.Range(0, 64).Select(i => 1.0 + (31.0 * i / 63))],
        OriginalMaxPosition = 2048,
        MaxPosition = ScaledRopeRows,
    };

    private static IEnumerable<OpCase> RopeScaledCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            foreach (ScaledRopeFamily f in ScaledRopeFamilies)
            {
                yield return RopeScaled("fixed", dt, f, 17, baseSeed);
            }

            yield return RopeScaled("fixed", dt, ScaledRopeFamilies[0], 1, baseSeed);
            var rng = new Rng(SeedFor(OpId.RopeScaled, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                yield return RopeScaled($"random-{i}", dt, rng.Pick(ScaledRopeFamilies), rng.Int(1, 140), baseSeed);
            }
        }
    }

    private static OpCase RopeScaled(string family, DType dt, ScaledRopeFamily f, int tokens, ulong baseSeed)
    {
        string id = $"{family}-{f.Name}-{dt}-t{tokens}";
        ulong seed = SeedFor(OpId.RopeScaled, id, baseSeed);
        var rng = new Rng(seed);
        var q = new HostBuf(dt, (long)tokens * f.QHeads * f.HeadDim).Randn(rng);
        var k = new HostBuf(dt, (long)tokens * f.KvHeads * f.HeadDim).Randn(rng);
        var pos = new HostBuf(DType.I32, tokens);
        for (int i = 0; i < tokens; i++)
        {
            // The last token sits on the cache's last row.
            pos.As<int>()[i] = i == tokens - 1 ? ScaledRopeRows - 1 : rng.Int(0, ScaledRopeRows);
        }

        int rot = f.Rotary == 0 ? f.HeadDim : f.Rotary;
        var cache = new HostBuf(DType.F32, (long)ScaledRopeRows * rot);
        RopeScaling.WriteCosSinCache(f.Build(), cache.Span, ScaledRopeRows);
        return RopeScaledFrom(id, dt, f, tokens, seed, q, k, pos, cache, true);
    }

    private static OpCase RopeScaledFrom(string id, DType dt, ScaledRopeFamily f, int tokens, ulong seed, HostBuf q, HostBuf k, HostBuf pos, HostBuf cache, bool withRows)
    {
        var c = new OpCase
        {
            Id = id,
            Op = OpId.RopeScaled,
            Dtype = dt,
            ToleranceClass = Elementwise(dt),
            Source = id.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed",
            Seed = seed,
            Rows = tokens,
            SingleRow = withRows
                ? t => RopeScaledFrom(id + $"-row{t}", dt, f, 1, seed, CopyRows(q, t, 1, f.QHeads * f.HeadDim), CopyRows(k, t, 1, f.KvHeads * f.HeadDim), CopyRows(pos, t, 1, 1), cache.Clone(), false)
                : null,
        };
        c.Num["tokens"] = tokens;
        c.Num["q_heads"] = f.QHeads;
        c.Num["kv_heads"] = f.KvHeads;
        c.Num["head_dim"] = f.HeadDim;
        c.Num["rotary_dim"] = f.Rotary;
        c.Num["scaling_kind"] = f.Kind;
        c.Num["factor"] = f.Factor;
        c.Buffers["q"] = q;
        c.Buffers["k"] = k;
        c.Buffers["pos"] = pos;
        c.Buffers["cos_sin_cache"] = cache;
        c.Outputs.AddRange(["q", "k"]);
        c.RowWidth["q"] = f.QHeads * f.HeadDim;
        c.RowWidth["k"] = f.KvHeads * f.HeadDim;
        return c;
    }
}
