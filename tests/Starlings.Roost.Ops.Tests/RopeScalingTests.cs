using System.Text.Json;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops.Tests;

/// <summary>
/// RopeScaling's YaRN, longrope, dynamic NTK and llama3 frequencies and cos/sin rows, and CpuOps.RopeScaled, against
/// transformers 5.20.1 (R-098).
/// </summary>
/// <remarks>
/// <c>Reference/rope-scaling-cases.json</c> is written by <c>Reference/make_rope_scaling_cases.py</c>: for each parameter set it
/// records the inverse frequencies and attention factor from transformers' own <c>modeling_rope_utils</c> functions (float32,
/// sha256 6a072707…) and from the same formulas in numpy float64, cos/sin cache rows in float64, and rotations made by
/// transformers' <c>apply_rotary_pos_emb</c> in float64. CpuOps computes in double, so it is compared with the float64 values
/// to rounding error and with the transformers values to float32 accuracy.
/// </remarks>
public sealed class RopeScalingTests
{
    private static readonly Lazy<JsonElement> s_doc = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Reference", "rope-scaling-cases.json"))).RootElement);

    private static IEnumerable<JsonElement> Cases => s_doc.Value.GetProperty("cases").EnumerateArray();

    private static JsonElement Case(string name) => Cases.Single(c => c.GetProperty("name").GetString() == name);

    public static TheoryData<string> CaseNames() => [.. s_doc.Value.GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!)];

    [Fact]
    public void TheCasesComeFromThePinnedReference()
    {
        Assert.Equal("5.20.1", s_doc.Value.GetProperty("transformers").GetString());
        Assert.Equal("6a0727073e080e163d0287e15fc1acb90035e9a9df9cb19a13b632a40abfac24", s_doc.Value.GetProperty("modeling_rope_utils_sha256").GetString());
        string[] kinds = [.. Cases.Select(c => c.GetProperty("kind").GetString()!).Distinct().Order(StringComparer.Ordinal)];
        Assert.Equal(["dynamic", "llama3", "longrope", "yarn"], kinds);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void FrequenciesMatchTheFloat64FormulasAndTransformers(string name)
    {
        JsonElement c = Case(name);
        RopeFrequencies f = Build(c);
        double[] got = [.. f.InverseFrequencies];
        Rel(Arr(c, "inv_freq"), got, 1e-13);
        Rel(Arr(c, "inv_freq_transformers"), got, 2e-6);
        Assert.Equal(c.GetProperty("attention_factor").GetDouble(), f.AttentionFactor, 1e-14);
        Assert.Equal(c.GetProperty("attention_factor_transformers").GetDouble(), f.AttentionFactor, 1e-12);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void CacheRowsMatchTheReference(string name)
    {
        JsonElement c = Case(name);
        RopeFrequencies f = Build(c);
        int rot = Int(c, "rotary_dim");
        double[] expected = Arr(c, "cos_sin_rows");
        int[] positions = [.. c.GetProperty("positions").EnumerateArray().Select(e => e.GetInt32())];
        var got = new double[positions.Length * rot];
        for (int i = 0; i < positions.Length; i++)
        {
            RopeScaling.CosSinRow(f, positions[i], got.AsSpan(i * rot, rot));
        }

        // Angles reach about 1e6 radians; a relative error of 1e-16 in a frequency moves cos and sin by about 1e-10.
        Near.Eq(expected, got, 1e-9);
    }

    [Fact]
    public void ScalingIsNotTheUnscaledRope()
    {
        // Each kind changes the frequencies it should change (the short longrope factors here are within 25% of 1, and dynamic
        // NTK is checked below); the YaRN and longrope attention factors are not 1.
        foreach (JsonElement c in Cases.Where(c => c.GetProperty("kind").GetString() != "dynamic" && c.GetProperty("name").GetString() != "longrope-short-hd96"))
        {
            double[] plain = [.. RopeScaling.Default(Int(c, "rotary_dim"), c.GetProperty("theta").GetDouble()).InverseFrequencies];
            double[] scaled = Arr(c, "inv_freq");
            Assert.True(plain.Zip(scaled).Max(p => Math.Abs((p.Second / p.First) - 1)) > 0.5, c.GetProperty("name").GetString());
        }

        Assert.All(Cases.Where(c => c.GetProperty("kind").GetString() is "yarn" or "longrope"), c => Assert.True(Math.Abs(c.GetProperty("attention_factor").GetDouble() - 1) > 0.05));
    }

    [Fact]
    public void DynamicNtkKeepsTheBaseUpToMaxPositionAndGrowsItBeyond()
    {
        double[] plain = [.. RopeScaling.Default(128, 1e4).InverseFrequencies];
        Rel(plain, Arr(Case("dynamic-init-hd128"), "inv_freq"), 1e-15);
        Rel(plain, Arr(Case("dynamic-short-hd128"), "inv_freq"), 1e-15);
        double[] longer = Arr(Case("dynamic-10000-hd128"), "inv_freq");
        double[] longest = Arr(Case("dynamic-65536-hd128"), "inv_freq");
        Assert.True(longer[^1] < plain[^1] && longest[^1] < longer[^1]);
    }

    [Fact]
    public void LongRopeSwitchesToTheLongFactorsAboveTheOriginalLength()
    {
        Rel(Arr(Case("longrope-short-hd96"), "inv_freq"), [.. Build(Case("longrope-short-hd96")).InverseFrequencies], 1e-13);
        double[] shortF = Arr(Case("longrope-short-hd96"), "inv_freq");
        double[] longF = Arr(Case("longrope-long-hd96"), "inv_freq");
        Assert.True(shortF.Zip(longF).Max(p => Math.Abs((p.Second / p.First) - 1)) > 0.5);
    }

    [Fact]
    public void RopeLlama3MatchesTransformers()
    {
        // A head whose first half is ones and second half zeros comes out as [cos..., sin...] of its position.
        foreach (string name in new[] { "llama3-f8", "llama3-f32" })
        {
            JsonElement c = Case(name);
            JsonElement p = c.GetProperty("params");
            int hd = Int(c, "head_dim");
            int[] positions = [.. c.GetProperty("positions").EnumerateArray().Select(e => e.GetInt32())];
            using var bag = new Bag();
            Mem q = bag.Add(Mem.F64([.. Enumerable.Range(0, positions.Length * hd).Select(i => i % hd < hd / 2 ? 1.0 : 0.0)]));
            Mem k = bag.Add(Mem.Zeros(DType.F64, positions.Length * hd));
            Mem pos = bag.Add(Mem.I32(positions));
            CpuOps.Rope(new RopeArgs
            {
                Q = q.Span, K = k.Span, Positions = pos.Span, Tokens = positions.Length, QHeads = 1, KvHeads = 1, HeadDim = hd, Kind = RopeKind.Llama3,
                Theta = (float)c.GetProperty("theta").GetDouble(), Llama3Factor = (float)D(p, "factor"), Llama3LowFreqFactor = (float)D(p, "low_freq_factor"),
                Llama3HighFreqFactor = (float)D(p, "high_freq_factor"), Llama3OriginalMaxPosition = (int)D(p, "original_max_position_embeddings"), DType = DType.F64,
            });
            Near.Eq(Arr(c, "cos_sin_rows"), q.ReadF64(), 1e-9);
        }
    }

    [Fact]
    public void RopeScaledMatchesTransformersRotationsInDouble()
    {
        foreach (JsonElement r in s_doc.Value.GetProperty("rotations").EnumerateArray())
        {
            JsonElement c = Case(r.GetProperty("case").GetString()!);
            (double[] q, double[] k) = Rotate(c, r, DType.F64);
            Near.Eq(Arr(r, "q_out"), q, 1e-9);
            Near.Eq(Arr(r, "k_out"), k, 1e-9);
        }
    }

    [Fact]
    public void RopeScaledMatchesTransformersRotationsInFloat32()
    {
        // Q, K and the cache in float32: each value is rounded on input, in the cache and on output, a few float32 ulps of
        // values below about 4 in magnitude.
        foreach (JsonElement r in s_doc.Value.GetProperty("rotations").EnumerateArray())
        {
            JsonElement c = Case(r.GetProperty("case").GetString()!);
            (double[] q, double[] k) = Rotate(c, r, DType.F32);
            Near.Eq(Arr(r, "q_out"), q, 4e-6);
            Near.Eq(Arr(r, "k_out"), k, 4e-6);
        }
    }

    [Fact]
    public void RopeScaledLeavesDimensionsBeyondRotaryDimAlone()
    {
        JsonElement r = s_doc.Value.GetProperty("rotations").EnumerateArray().First(x => x.GetProperty("case").GetString() == "dynamic-hd256-rot64");
        JsonElement c = Case("dynamic-hd256-rot64");
        (double[] q, _) = Rotate(c, r, DType.F64);
        double[] input = Arr(r, "q");
        int hd = Int(c, "head_dim"), rot = Int(c, "rotary_dim");
        for (int i = 0; i < q.Length; i++)
        {
            if (i % hd >= rot)
            {
                Assert.Equal(input[i], q[i]);
            }
        }
    }

    [Fact]
    public void RopeScaledRejectsBadArgumentsBeforeWriting()
    {
        using var bag = new Bag();
        Mem q = bag.Add(Mem.F32(1, 2, 3, 4, 5, 6, 7, 8));
        Mem k = bag.Add(Mem.F32(1, 2, 3, 4, 5, 6, 7, 8));
        Mem cache = bag.Add(Mem.Zeros(DType.F32, 4 * 4));
        RopeScaling.WriteCosSinCache(RopeScaling.Default(4, 1e4), cache.Span, 4);
        Mem inRange = bag.Add(Mem.I32(0, 3));
        Mem outOfRange = bag.Add(Mem.I32(1, 4));
        Mem negative = bag.Add(Mem.I32(-1, 0));
        RopeScaledArgs Args(Mem pos) => new()
        {
            Q = q.Span, K = k.Span, Positions = pos.Span, CosSinCache = cache.Span, Tokens = 2, QHeads = 1, KvHeads = 1, HeadDim = 4, RotaryDim = 4,
            ScalingKind = 1, Factor = 4f, DType = DType.F32,
        };

        float[] before = q.ReadF32();
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(outOfRange)));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(negative)));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(inRange) with { ScalingKind = 0 }));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(inRange) with { ScalingKind = 4 }));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(inRange) with { Factor = 0f }));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(inRange) with { Factor = float.NaN }));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(inRange) with { RotaryDim = 3 }));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(inRange) with { RotaryDim = 6 }));
        Assert.Throws<ArgumentException>(() => CpuOps.RopeScaled(Args(inRange) with { CosSinCache = default }));
        Assert.Equal(before, q.ReadF32());

        // Position 0 is the identity row (cos 1, sin 0): the first token is unchanged and the second rotated.
        CpuOps.RopeScaled(Args(inRange));
        float[] after = q.ReadF32();
        Assert.Equal(before[..4], after[..4]);
        Assert.NotEqual(before[4..], after[4..]);
    }

    [Fact]
    public void RotaryDimZeroMeansHeadDim()
    {
        JsonElement r = s_doc.Value.GetProperty("rotations").EnumerateArray().First(x => x.GetProperty("case").GetString() == "yarn-f40-mscale-hd64");
        JsonElement c = Case("yarn-f40-mscale-hd64");
        Assert.Equal(Int(c, "head_dim"), Int(c, "rotary_dim"));
        (double[] q, double[] k) = Rotate(c, r, DType.F64, rotaryDimZero: true);
        Near.Eq(Arr(r, "q_out"), q, 1e-9);
        Near.Eq(Arr(r, "k_out"), k, 1e-9);
    }

    [Fact]
    public void WriteCosSinCacheWritesRowsInTheSpanType()
    {
        RopeFrequencies f = RopeScaling.Yarn(new YarnParameters { RotaryDim = 8, Theta = 1e4, Factor = 4, OriginalMaxPosition = 64 });
        using var bag = new Bag();
        Mem f64 = bag.Add(Mem.Zeros(DType.F64, 10 * 8));
        Mem bf16 = bag.Add(Mem.Zeros(DType.BF16, 10 * 8));
        RopeScaling.WriteCosSinCache(f, f64.Span, 10);
        RopeScaling.WriteCosSinCache(f, bf16.Span, 10);
        var row = new double[8];
        RopeScaling.CosSinRow(f, 7, row);
        Assert.Equal(row, f64.ReadF64()[56..64]);
        Near.Eq(row, [.. bf16.ReadBf16()[56..64].Select(x => (double)x)], 1.2 * Math.Pow(2, -8));
        Assert.Throws<ArgumentException>(() => RopeScaling.WriteCosSinCache(f, f64.Span, 11));
    }

    private static (double[] Q, double[] K) Rotate(JsonElement c, JsonElement r, DType dt, bool rotaryDimZero = false)
    {
        RopeFrequencies f = Build(c);
        int hd = Int(c, "head_dim"), rot = Int(c, "rotary_dim");
        int tokens = Int(r, "tokens"), qh = Int(r, "q_heads"), kvh = Int(r, "kv_heads");
        int[] positions = [.. r.GetProperty("positions").EnumerateArray().Select(e => e.GetInt32())];
        int rows = positions.Max() + 1;
        using var bag = new Bag();
        Mem cache = bag.Add(Mem.Zeros(dt, rows * rot));
        RopeScaling.WriteCosSinCache(f, cache.Span, rows);
        Mem q = bag.Add(dt == DType.F64 ? Mem.F64(Arr(r, "q")) : Mem.F32(Arr(r, "q")));
        Mem k = bag.Add(dt == DType.F64 ? Mem.F64(Arr(r, "k")) : Mem.F32(Arr(r, "k")));
        Mem pos = bag.Add(Mem.I32(positions));
        CpuOps.RopeScaled(new RopeScaledArgs
        {
            Q = q.Span, K = k.Span, Positions = pos.Span, CosSinCache = cache.Span, Tokens = tokens, QHeads = qh, KvHeads = kvh, HeadDim = hd,
            RotaryDim = rotaryDimZero ? 0 : rot, ScalingKind = KindId(c), Factor = (float)FactorOf(c), DType = dt,
        });
        return dt == DType.F64
            ? (q.ReadF64(), k.ReadF64())
            : ([.. q.ReadF32().Select(x => (double)x)], [.. k.ReadF32().Select(x => (double)x)]);
    }

    private static int KindId(JsonElement c) => c.GetProperty("kind").GetString() switch
    {
        "yarn" => 1,
        "longrope" => 2,
        "dynamic" => 3,
        var k => throw new InvalidOperationException(k),
    };

    private static double FactorOf(JsonElement c) =>
        Opt(c.GetProperty("params"), "factor") ?? ((double)Int(c, "max_position_embeddings") / D(c.GetProperty("params"), "original_max_position_embeddings"));

    private static RopeFrequencies Build(JsonElement c)
    {
        JsonElement p = c.GetProperty("params");
        int rot = Int(c, "rotary_dim");
        double theta = c.GetProperty("theta").GetDouble();
        int maxPos = Int(c, "max_position_embeddings");
        int seqLen = c.GetProperty("seq_len").ValueKind == JsonValueKind.Null ? 0 : Int(c, "seq_len");
        return c.GetProperty("kind").GetString() switch
        {
            "yarn" => RopeScaling.Yarn(new YarnParameters
            {
                RotaryDim = rot,
                Theta = theta,
                Factor = Opt(p, "factor"),
                OriginalMaxPosition = (int)D(p, "original_max_position_embeddings"),
                MaxPosition = maxPos,
                AttentionFactor = Opt(p, "attention_factor"),
                Mscale = Opt(p, "mscale"),
                MscaleAllDim = Opt(p, "mscale_all_dim"),
                BetaFast = Opt(p, "beta_fast") ?? 0,
                BetaSlow = Opt(p, "beta_slow") ?? 0,
                Truncate = !p.TryGetProperty("truncate", out JsonElement t) || t.GetBoolean(),
            }),
            "longrope" => RopeScaling.LongRope(
                new LongRopeParameters
                {
                    RotaryDim = rot,
                    Theta = theta,
                    ShortFactor = Arr(p, "short_factor"),
                    LongFactor = Arr(p, "long_factor"),
                    OriginalMaxPosition = (int)D(p, "original_max_position_embeddings"),
                    MaxPosition = maxPos,
                    Factor = Opt(p, "factor"),
                    AttentionFactor = Opt(p, "attention_factor"),
                },
                seqLen),
            "dynamic" => RopeScaling.DynamicNtk(rot, theta, D(p, "factor"), maxPos, seqLen),
            "llama3" => RopeScaling.Llama3(rot, theta, D(p, "factor"), D(p, "low_freq_factor"), D(p, "high_freq_factor"), (int)D(p, "original_max_position_embeddings")),
            var k => throw new InvalidOperationException(k),
        };
    }

    private static double? Opt(JsonElement p, string name) =>
        p.TryGetProperty(name, out JsonElement v) && v.ValueKind != JsonValueKind.Null ? v.GetDouble() : null;

    private static double D(JsonElement p, string name) => p.GetProperty(name).GetDouble();

    private static double[] Arr(JsonElement c, string name) => [.. c.GetProperty(name).EnumerateArray().Select(e => e.GetDouble())];

    private static int Int(JsonElement c, string name) => c.GetProperty(name).GetInt32();

    // |expected - actual| <= rel * |expected| per element.
    private static void Rel(double[] expected, double[] actual, double rel)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= rel * Math.Abs(expected[i]), $"[{i}] expected {expected[i]:R} actual {actual[i]:R}");
        }
    }
}
