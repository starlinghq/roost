using System.Text.Json;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>Reads tensors of the locked Qwen3.5-0.8B checkpoint (read in place, never copied into a repository).</summary>
internal sealed class SafeTensors
{
    private readonly string _file;
    private readonly long _dataStart;
    private readonly Dictionary<string, (string Dtype, long[] Shape, long Begin, long End)> _index = [];

    public SafeTensors(string file)
    {
        _file = file;
        using FileStream fs = File.OpenRead(file);
        Span<byte> len = stackalloc byte[8];
        fs.ReadExactly(len);
        long n = (long)BitConverter.ToUInt64(len);
        var header = new byte[n];
        fs.ReadExactly(header);
        _dataStart = 8 + n;
        using JsonDocument doc = JsonDocument.Parse(header);
        foreach (JsonProperty p in doc.RootElement.EnumerateObject())
        {
            if (p.Name == "__metadata__")
            {
                continue;
            }

            var shape = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            var off = p.Value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            _index[p.Name] = (p.Value.GetProperty("dtype").GetString()!, shape, off[0], off[1]);
        }
    }

    public Buf Read(string name, DType expect)
    {
        var (dtype, shape, begin, end) = _index[name];
        if (dtype != "BF16" || expect != DType.BF16)
        {
            throw new InvalidOperationException($"{name} is {dtype}; only BF16 is read here");
        }

        var buf = new Buf(DType.BF16, shape.Aggregate(1L, (a, b) => a * b));
        using FileStream fs = File.OpenRead(_file);
        fs.Seek(_dataStart + begin, SeekOrigin.Begin);
        long left = end - begin;
        long done = 0;
        while (left > 0)
        {
            int chunk = (int)Math.Min(left, 1 << 28);
            unsafe
            {
                fs.ReadExactly(new Span<byte>((void*)(buf.Host + (nint)done), chunk));
            }

            done += chunk;
            left -= chunk;
        }

        return buf;
    }
}

/// <summary>The ops on real Qwen3.5-0.8B weights (layer 3 is the first full-attention layer).</summary>
internal static class RealCases
{
    private const string Layer = "model.language_model.layers.3.";

    public static void Run(Ctx c, string checkpointFile)
    {
        var st = new SafeTensors(checkpointFile);
        c.Line($"real weights from {Path.GetFileName(checkpointFile)} (read in place)");
        using (Buf w = st.Read(Layer + "input_layernorm.weight", DType.BF16))
        {
            RmsNorm(c, w);
        }

        using (Buf qw = st.Read(Layer + "self_attn.q_norm.weight", DType.BF16))
        using (Buf kw = st.Read(Layer + "self_attn.k_norm.weight", DType.BF16))
        {
            QkNorm(c, qw, kw);
        }

        foreach ((string name, int n, int k) in new[] { ("self_attn.q_proj", 4096, 1024), ("self_attn.k_proj", 512, 1024), ("self_attn.v_proj", 512, 1024), ("self_attn.o_proj", 1024, 2048), ("mlp.gate_proj", 3584, 1024), ("mlp.up_proj", 3584, 1024), ("mlp.down_proj", 1024, 3584) })
        {
            using Buf w = st.Read(Layer + name + ".weight", DType.BF16);
            foreach (int m in new[] { 1, 17 })
            {
                GemmCases.Gemm(c, DType.BF16, "real " + name, m, n, k, true, false, 1f, 0f, 0, w);
            }
        }

        using Buf table = st.Read("model.language_model.embed_tokens.weight", DType.BF16);
        EmbeddingAndLmHead(c, table);
    }

    private static unsafe void RmsNorm(Ctx c, Buf w)
    {
        const int rows = 5, dim = 1024;
        using var x = new Buf(DType.BF16, rows * dim).Randn(c.Rng, 3.0);
        using var y = new Buf(DType.BF16, rows * dim);
        x.Up();
        w.Up();
        var p = new RmsNormParams { StructSize = (uint)sizeof(RmsNormParams), Input = x.DevSpan, Weight = w.DevSpan, Output = y.DevSpan, Rows = rows, Dim = dim, Eps = 1e-6f };
        c.Lib.Exec("rms-norm-flashinfer", ref p, c.Stream);
        using var got = Ctx.Fetch(y);
        using var want = new Buf(DType.BF16, rows * dim);
        CpuOps.RmsNorm(new RmsNormArgs { Input = x.HostSpan, Weight = w.HostSpan, Output = want.HostSpan, Rows = rows, Dim = dim, Eps = 1e-6f, DType = DType.BF16 });
        c.Report(Compare.Close("RmsNorm bf16 real layer-3 input_layernorm rows=5", got, want, Tol.Elementwise(DType.BF16)));
    }

    private static unsafe void QkNorm(Ctx c, Buf qw, Buf kw)
    {
        const int tokens = 9, qh = 8, kh = 2, hd = 256;
        using var q = new Buf(DType.BF16, tokens * qh * hd).Randn(c.Rng, 2.0);
        using var k = new Buf(DType.BF16, tokens * kh * hd).Randn(c.Rng, 2.0);
        using var qc = q.Clone();
        using var kc = k.Clone();
        q.Up();
        k.Up();
        qw.Up();
        kw.Up();
        var p = new QkNormParams { StructSize = (uint)sizeof(QkNormParams), Q = q.DevSpan, K = k.DevSpan, QWeight = qw.DevSpan, KWeight = kw.DevSpan, Tokens = tokens, QHeads = qh, KvHeads = kh, HeadDim = hd, Eps = 1e-6f };
        c.Lib.Exec("qk-norm-flashinfer", ref p, c.Stream);
        using var gq = Ctx.Fetch(q);
        using var gk = Ctx.Fetch(k);
        CpuOps.QkNorm(new QkNormArgs { Q = qc.HostSpan, K = kc.HostSpan, QWeight = qw.HostSpan, KWeight = kw.HostSpan, Tokens = tokens, QHeads = qh, KvHeads = kh, HeadDim = hd, Eps = 1e-6f, DType = DType.BF16 });
        c.Report(Compare.Close("QkNorm bf16 real layer-3 q_norm tokens=9", gq, qc, Tol.Elementwise(DType.BF16)));
        c.Report(Compare.Close("QkNorm bf16 real layer-3 k_norm tokens=9", gk, kc, Tol.Elementwise(DType.BF16)));
    }

    private static unsafe void EmbeddingAndLmHead(Ctx c, Buf table)
    {
        const int vocab = 248320, dim = 1024, tokens = 200;
        using var ids = new Buf(DType.I32, tokens);
        for (int i = 0; i < tokens; i++)
        {
            ids.As<int>()[i] = i == 0 ? 0 : i == 1 ? vocab - 1 : c.Rng.Int(0, vocab);
        }

        using var outBuf = new Buf(DType.BF16, (long)tokens * dim);
        table.Up();
        ids.Up();
        var p = new EmbeddingParams { StructSize = (uint)sizeof(EmbeddingParams), Ids = ids.DevSpan, Table = table.DevSpan, Output = outBuf.DevSpan, Tokens = tokens, Dim = dim, Vocab = vocab };
        c.Lib.Exec("embedding-roost", ref p, c.Stream);
        using var got = Ctx.Fetch(outBuf);
        using var want = new Buf(DType.BF16, (long)tokens * dim);
        CpuOps.Embedding(new EmbeddingArgs { Ids = ids.HostSpan, Table = table.HostSpan, Output = want.HostSpan, Tokens = tokens, Dim = dim, Vocab = vocab, DType = DType.BF16 });
        c.Report(Compare.Bits("Embedding bf16 real embed_tokens vocab=248320 tokens=200", got, want));

        // The tied LM head: hidden states are normalised, so use the embeddings of random tokens (already O(0.01..0.1)) times a gain.
        foreach (int selected in new[] { 1, 3 })
        {
            using var hidden = new Buf(DType.BF16, (long)(selected + 2) * dim);
            for (long i = 0; i < hidden.Count; i++)
            {
                hidden.Set(i, (c.Rng.Normal() * 1.0) + 0.0);
            }

            GemmCases.LmHead(c, DType.BF16, vocab, dim, vocab, selected, 0f, weight: table, hidden: hidden, tokens: selected + 2, label: "real tied ");
        }
    }
}
