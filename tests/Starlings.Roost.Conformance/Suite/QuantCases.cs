using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// The cases of the M2 quantisation ops (R-077, R-043): Quantize, Dequantize, GemmFp8 and GemmW4A16, checked against
/// <c>CpuOps</c> only. No kernel library registers a variant of them yet, so only the CPU half runs them; the GPU half is
/// <c>untested</c>. Shapes are the Qwen3.5-0.8B projections (hidden 1024, MLP 3584) with AWQ's and GPTQ's usual group of 128,
/// plus random shapes. Quantised inputs are random codes: FP8 codes other than NaN (and, for E5M2, infinity), and packed 4-bit
/// bytes in C3's plain layout ([N, K], low nibble first). A GemmW4A16 kernel in Marlin layout has to repack B from that.
/// </summary>
public static partial class CaseCatalog
{
    private const int QuantGroup = 128;
    private static readonly DType[] Fp8Types = [DType.F8E4M3, DType.F8E5M2];
    private static readonly DType[] Int4Types = [DType.U4, DType.I4];
    private static readonly DType[] DequantInputs = [DType.F8E4M3, DType.F8E5M2, DType.U4, DType.I4, DType.I8];

    private static IEnumerable<OpCase> QuantCases(OpId op, ulong baseSeed, CaseProfile p) => op switch
    {
        OpId.Quantize => QuantizeCases(baseSeed, p),
        OpId.Dequantize => DequantizeCases(baseSeed, p),
        OpId.GemmFp8 => GemmFp8Cases(baseSeed, p),
        OpId.GemmW4A16 => GemmW4A16Cases(baseSeed, p),
        _ => throw new NotSupportedException($"No cases for {op}."),
    };

    private static string GemmClass(DType t) => t == DType.F16 ? "gemm-f16" : "gemm-bf16";

    private static HostBuf Fp8Codes(DType t, long n, Rng rng)
    {
        var b = new HostBuf(t, n);
        Span<byte> raw = b.Raw();
        for (long i = 0; i < n; i++)
        {
            byte c;
            do
            {
                c = (byte)rng.Int(0, 256);
            }
            while (t == DType.F8E4M3 ? (c & 0x7F) == 0x7F : (c & 0x7C) == 0x7C);
            raw[(int)i] = c;
        }

        return b;
    }

    // Random bytes: every 8-bit integer, or every pair of 4-bit values (packed, so only the first (n + 1) / 2 bytes are elements).
    private static HostBuf RandomBytes(DType t, long n, Rng rng)
    {
        var b = new HostBuf(t, n);
        Span<byte> raw = b.Raw();
        long bytes = t is DType.U4 or DType.I4 ? (n + 1) / 2 : b.Bytes;
        for (long i = 0; i < bytes; i++)
        {
            raw[(int)i] = (byte)rng.Int(0, 256);
        }

        return b;
    }

    // ---- Quantize -----------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> QuantizeCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            foreach (DType q in new[] { DType.F8E4M3, DType.F8E5M2, DType.I8, DType.I4 })
            {
                yield return Quantize("fixed", dt, q, 7, 1024, 0, baseSeed);
                yield return Quantize("fixed", dt, q, 5, 3584, QuantGroup, baseSeed);
                var rng = new Rng(SeedFor(OpId.Quantize, $"random-shapes-{dt}-{q}", baseSeed));
                for (int i = 0; i < p.RandomPerOp; i++)
                {
                    int g = 8 * rng.Int(1, 33);
                    yield return Quantize($"random-{i}", dt, q, rng.Int(1, 40), g * rng.Int(1, 12), rng.Int(0, 2) == 0 ? 0 : g, baseSeed);
                }
            }
        }
    }

    private static OpCase Quantize(string family, DType dt, DType q, int rows, int dim, int group, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{q}-{Dim(rows, dim)}-g{group}";
        ulong seed = SeedFor(OpId.Quantize, id, baseSeed);
        var rng = new Rng(seed);
        int groups = group == 0 ? 1 : dim / group;
        var c = new OpCase { Id = id, Op = OpId.Quantize, Dtype = dt, ToleranceClass = "exact", Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed", Seed = seed };
        c.Num["rows"] = rows;
        c.Num["dim"] = dim;
        c.Num["group"] = group;
        c.Num["out_dtype"] = (double)q;
        c.Buffers["x"] = new HostBuf(dt, (long)rows * dim).Randn(rng, 3.0);
        c.Buffers["out"] = new HostBuf(q, (long)rows * dim);
        c.Buffers["scales"] = new HostBuf(DType.F32, (long)rows * groups);
        c.Outputs.Add("out");
        c.Outputs.Add("scales");
        return c;
    }

    // ---- Dequantize ---------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> DequantizeCases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            foreach (DType q in Fp8Types)
            {
                yield return Dequantize("fixed", dt, q, 7, 1024, 0, false, baseSeed);
                yield return Dequantize("fixed", dt, q, 5, 3584, QuantGroup, false, baseSeed);
            }

            yield return Dequantize("fixed", dt, DType.U4, 64, 1024, QuantGroup, true, baseSeed);
            yield return Dequantize("fixed", dt, DType.U4, 64, 1024, QuantGroup, false, baseSeed);
            yield return Dequantize("fixed", dt, DType.I4, 32, 3584, QuantGroup, false, baseSeed);
            var rng = new Rng(SeedFor(OpId.Dequantize, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                DType q = rng.Pick(DequantInputs);
                int g = 8 * rng.Int(1, 33);
                yield return Dequantize($"random-{i}", dt, q, rng.Int(1, 40), g * rng.Int(1, 12), rng.Int(0, 2) == 0 ? 0 : g, q == DType.U4 && rng.Int(0, 2) == 0, baseSeed);
            }
        }
    }

    private static OpCase Dequantize(string family, DType dt, DType q, int rows, int dim, int group, bool zeros, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{q}-{Dim(rows, dim)}-g{group}{(zeros ? "-z" : string.Empty)}";
        ulong seed = SeedFor(OpId.Dequantize, id, baseSeed);
        var rng = new Rng(seed);
        long n = (long)rows * dim;
        long groups = (long)rows * (group == 0 ? 1 : dim / group);
        bool fp8 = q is DType.F8E4M3 or DType.F8E5M2;
        var c = new OpCase { Id = id, Op = OpId.Dequantize, Dtype = dt, ToleranceClass = Elementwise(dt), Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed", Seed = seed };
        c.Num["rows"] = rows;
        c.Num["dim"] = dim;
        c.Num["group"] = group;
        c.Num["in_dtype"] = (double)q;
        c.Buffers["q"] = fp8 ? Fp8Codes(q, n, rng) : RandomBytes(q, n, rng);

        // FP8 scales as float32 (per-tensor or per-token, as vLLM keeps them); 4-bit group scales as float16 (as checkpoints keep them).
        c.Buffers["scales"] = fp8 ? Uniform(DType.F32, groups, rng, 0.001, 0.05) : Uniform(DType.F16, groups, rng, 0.002, 0.03);
        if (zeros)
        {
            c.Buffers["zeros"] = RandomBytes(DType.U4, groups, rng);
        }

        c.Buffers["out"] = new HostBuf(dt, n);
        c.Outputs.Add("out");
        return c;
    }

    // ---- GemmFp8 ------------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> GemmFp8Cases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            yield return GemmFp8("fixed", dt, DType.F8E4M3, DType.F8E4M3, 7, 256, 1024, false, false, baseSeed);
            yield return GemmFp8("fixed", dt, DType.F8E4M3, DType.F8E4M3, 5, 128, 3584, true, true, baseSeed);
            yield return GemmFp8("fixed", dt, DType.F8E5M2, DType.F8E4M3, 3, 64, 1024, true, false, baseSeed);
            if (p.Large)
            {
                yield return GemmFp8("fixed-large", dt, DType.F8E4M3, DType.F8E4M3, 64, 1024, 1024, true, true, baseSeed);
            }

            var rng = new Rng(SeedFor(OpId.GemmFp8, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                yield return GemmFp8($"random-{i}", dt, rng.Pick(Fp8Types), rng.Pick(Fp8Types), rng.Int(1, 33), 16 * rng.Int(1, 17), 16 * rng.Int(1, 65), rng.Int(0, 2) == 0, rng.Int(0, 2) == 0, baseSeed);
            }
        }
    }

    private static OpCase GemmFp8(string family, DType dt, DType at, DType bt, int m, int n, int k, bool perToken, bool perChannel, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{at}x{bt}-{Dim(m, n, k)}{(perToken ? "-tok" : string.Empty)}{(perChannel ? "-ch" : string.Empty)}";
        ulong seed = SeedFor(OpId.GemmFp8, id, baseSeed);
        var rng = new Rng(seed);
        var c = new OpCase { Id = id, Op = OpId.GemmFp8, Dtype = dt, ToleranceClass = GemmClass(dt), Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed", Seed = seed };
        c.Num["m"] = m;
        c.Num["n"] = n;
        c.Num["k"] = k;
        c.Buffers["a"] = Fp8Codes(at, (long)m * k, rng);
        c.Buffers["b"] = Fp8Codes(bt, (long)n * k, rng);

        // Scales that keep the random codes' products near unit size (E5M2 codes reach 57344, E4M3 codes 448).
        double sa = at == DType.F8E4M3 ? 1.0 / 64 : 1.0 / 4096, sb = bt == DType.F8E4M3 ? 1.0 / 64 : 1.0 / 4096;
        c.Buffers["a_scale"] = Uniform(DType.F32, perToken ? m : 1, rng, 0.5 * sa, sa);
        c.Buffers["b_scale"] = Uniform(DType.F32, perChannel ? n : 1, rng, 0.5 * sb, sb);
        c.Buffers["out"] = new HostBuf(dt, (long)m * n);
        c.Outputs.Add("out");
        return c;
    }

    // ---- GemmW4A16 ----------------------------------------------------------------------------------------------

    private static IEnumerable<OpCase> GemmW4A16Cases(ulong baseSeed, CaseProfile p)
    {
        foreach (DType dt in Half)
        {
            yield return GemmW4A16("fixed", dt, DType.U4, 1, 256, 1024, QuantGroup, true, baseSeed);
            yield return GemmW4A16("fixed", dt, DType.U4, 7, 256, 1024, QuantGroup, false, baseSeed);
            yield return GemmW4A16("fixed", dt, DType.U4, 5, 128, 3584, QuantGroup, true, baseSeed);
            yield return GemmW4A16("fixed", dt, DType.I4, 3, 64, 1024, 0, false, baseSeed);
            if (p.Large)
            {
                yield return GemmW4A16("fixed-large", dt, DType.U4, 16, 3584, 1024, QuantGroup, true, baseSeed);
            }

            var rng = new Rng(SeedFor(OpId.GemmW4A16, "random-shapes-" + dt, baseSeed));
            for (int i = 0; i < p.RandomPerOp; i++)
            {
                int g = 8 * rng.Int(2, 17);
                yield return GemmW4A16($"random-{i}", dt, rng.Pick(Int4Types), rng.Int(1, 33), 8 * rng.Int(1, 33), g * rng.Int(1, 9), g, rng.Int(0, 2) == 0, baseSeed);
            }
        }
    }

    private static OpCase GemmW4A16(string family, DType dt, DType wt, int m, int n, int k, int group, bool zeros, ulong baseSeed)
    {
        string id = $"{family}-{dt}-{wt}-{Dim(m, n, k)}-g{group}{(zeros ? "-z" : string.Empty)}";
        ulong seed = SeedFor(OpId.GemmW4A16, id, baseSeed);
        var rng = new Rng(seed);
        long groups = (long)n * (group == 0 ? 1 : k / group);
        var c = new OpCase { Id = id, Op = OpId.GemmW4A16, Dtype = dt, ToleranceClass = GemmClass(dt), Source = family.StartsWith("random", StringComparison.Ordinal) ? "random" : "fixed", Seed = seed };
        c.Num["m"] = m;
        c.Num["n"] = n;
        c.Num["k"] = k;
        c.Num["group"] = group;
        c.Buffers["a"] = new HostBuf(dt, (long)m * k).Randn(rng);
        c.Buffers["b"] = RandomBytes(wt, (long)n * k, rng);
        c.Buffers["scales"] = Uniform(DType.F16, groups, rng, 0.002, 0.03);
        if (zeros)
        {
            c.Buffers["zeros"] = RandomBytes(DType.U4, groups, rng);
        }

        c.Buffers["out"] = new HostBuf(dt, (long)m * n);
        c.Outputs.Add("out");
        return c;
    }
}
