using System.Runtime.InteropServices;
using Starlings.Roost;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Cuda;

/// <summary>A device buffer of exactly the requested size, so compute-sanitizer memcheck sees any access past it.</summary>
internal sealed class DevBuf : IDisposable
{
    private nint _ptr;

    private DevBuf(nint ptr, DType dtype, long count)
    {
        _ptr = ptr;
        Dtype = dtype;
        Count = count;
    }

    public nint Ptr => _ptr;

    public DType Dtype { get; }

    public long Count { get; }

    public long Bytes => Count * Elem.Size(Dtype);

    public DeviceSpan Span => new() { Address = _ptr, Length = (ulong)Count, Dtype = Dtype };

    public static DevBuf Upload(HostBuf h)
    {
        var d = new DevBuf(CudaRuntime.Alloc(h.Bytes), h.Dtype, h.Count);
        CudaRuntime.ToDevice(d._ptr, h.Host, h.Bytes);
        return d;
    }

    public static DevBuf Empty(DType dtype, long count, byte fill)
    {
        var d = new DevBuf(CudaRuntime.Alloc(count * Elem.Size(dtype)), dtype, count);
        CudaRuntime.Fill(d._ptr, fill, d.Bytes);
        return d;
    }

    public HostBuf Download()
    {
        var h = new HostBuf(Dtype, Count);
        CudaRuntime.ToHost(h.Host, _ptr, Bytes);
        return h;
    }

    public void Dispose()
    {
        if (_ptr != 0)
        {
            CudaRuntime.Release(_ptr);
            _ptr = 0;
        }
    }
}

/// <summary>A kernel library loaded through its C1 exports, as a conformance backend: every registered variant, every op of the suite.</summary>
internal sealed unsafe class CudaBackend : IKernelBackend
{
    private sealed class ProbeDone(int status) : Exception
    {
        public int Status { get; } = status;
    }

    private readonly Library _lib;
    private readonly nint _stream;
    private readonly List<VariantInfo> _variants;
    private bool _probing;

    public CudaBackend(string libraryPath, int device)
    {
        CudaRuntime.Device(device);
        Sm = CudaRuntime.ComputeCapability(device);
        _stream = CudaRuntime.NewStream();
        _lib = new Library(libraryPath, device);
        _variants = [.. _lib.Entries.Select(kv => new VariantInfo(kv.Value.Op, kv.Key, kv.Value.Flags, kv.Value.Constraints))];
    }

    public string Name => _lib.Name;

    public int Sm { get; }

    public IReadOnlyList<VariantInfo> Variants => _variants;

    public uint AbiVersion => _lib.AbiVersion;

    public int? ProbeShortStruct(string variant, OpCase c)
    {
        _probing = true;
        try
        {
            using ExecResult r = Execute(variant, c);
            return null;
        }
        catch (ProbeDone d)
        {
            return d.Status;
        }
        finally
        {
            _probing = false;
            CudaRuntime.Sync(_stream);
            CudaRuntime.ClearError();
        }
    }

    public ExecResult Execute(string variant, OpCase c)
    {
        var res = new ExecResult { Outputs = new Dictionary<string, HostBuf>(StringComparer.Ordinal) };
        // Status and message are init-only on ExecResult, so a refusal is returned as a fresh result.
        try
        {
            Run(variant, c, res);
            return res;
        }
        catch (KernelException e)
        {
            res.Dispose();
            CudaRuntime.Sync(_stream);
            CudaRuntime.ClearError();
            return new ExecResult { Status = e.Status, Message = e.Message };
        }
    }

    public void Dispose()
    {
        _lib.Dispose();
        CudaRuntime.DestroyStream(_stream);
    }

    private void Run(string v, OpCase c, ExecResult res)
    {
        switch (c.Op)
        {
            case OpId.RmsNorm: RmsNorm(v, c, res); break;
            case OpId.QkNorm: QkNorm(v, c, res); break;
            case OpId.Rope: Rope(v, c, res); break;
            case OpId.PagedAttention: Attention(v, c, res); break;
            case OpId.SiluAndMul: Silu(v, c, res); break;
            case OpId.Gemm: Gemm(v, c, res); break;
            case OpId.Embedding: Embedding(v, c, res); break;
            case OpId.LmHead: LmHead(v, c, res); break;
            case OpId.Residual: Residual(v, c, res); break;
            case OpId.CacheWrite: CacheWrite(v, c, res); break;
            case OpId.CacheCopy: CacheCopy(v, c, res); break;
            default: throw new NotSupportedException($"No GPU case runner for {c.Op}.");
        }
    }

    // ---- launching ----------------------------------------------------------------------------------------------

    private WorkspaceSet? MakeWorkspace<T>(string v, ref T p) where T : unmanaged
        => _lib.Entries[v].Workspace != 0 ? _lib.MakeWorkspace(v, ref p) : null;

    /// <summary>
    /// Runs the variant on the stream. A variant flagged NEEDS_PLAN is first run without a plan (C1: NOT_PLANNED, before any
    /// launch), then planned and run. In a probe the struct is made 8 bytes short and the call's status is returned through
    /// <see cref="ProbeDone"/>.
    /// </summary>
    private void Launch<T>(string v, ref T p, ExecResult res) where T : unmanaged
    {
        VariantTraits flags = _lib.Entries[v].Flags;
        using WorkspaceSet? ws = MakeWorkspace(v, ref p);
        if (_probing)
        {
            T shortP = p;
            *(uint*)&shortP -= 8;
            int status = ws is null ? _lib.Try(v, ref shortP, _stream) : _lib.TryRunWith(v, ref shortP, ws, _stream);
            throw new ProbeDone(status);
        }

        if ((flags & VariantTraits.NeedsPlan) != 0)
        {
            if (ws is null)
            {
                throw new KernelException((int)KernelStatus.Internal, $"{v}: flagged NEEDS_PLAN but has no workspace query");
            }

            int unplanned = _lib.TryRunWith(v, ref p, ws, _stream);
            res.Checks.Add(new BackendCheck("run before plan -> NOT_PLANNED", unplanned == (int)KernelStatus.NotPlanned, $"status {unplanned}"));
            _lib.PlanWith(v, ref p, ws, _stream);
            _lib.RunWith(v, ref p, ws, _stream);
        }
        else if (ws is null)
        {
            _lib.Exec(v, ref p, _stream);
        }
        else
        {
            _lib.RunWith(v, ref p, ws, _stream);
        }
    }

    private static void PaddingCheck(ExecResult res, string name, HostBuf got, byte fill, long rows, long width, long stride)
    {
        if (stride <= width)
        {
            return;
        }

        int es = Elem.Size(got.Dtype);
        Span<byte> raw = got.Raw();
        bool ok = true;
        for (long r = 0; r < rows && ok; r++)
        {
            ok = raw.Slice((int)(((r * stride) + width) * es), (int)((stride - width) * es)).IndexOfAnyExcept(fill) < 0;
        }

        res.Checks.Add(new BackendCheck($"{name}: elements between rows untouched", ok));
    }

    // ---- ops ----------------------------------------------------------------------------------------------------

    private void RmsNorm(string v, OpCase c, ExecResult res)
    {
        int rows = c.I("rows"), dim = c.I("dim"), inS = c.I("in_stride"), outS = c.I("out_stride");
        int istr = inS == 0 ? dim : inS, ostr = outS == 0 ? dim : outS;
        using HostBuf xs = c.Buffers["x"].Strided(rows, dim, istr, 0xCD);
        using DevBuf x = DevBuf.Upload(xs);
        using DevBuf w = DevBuf.Upload(c.Buffers["w"]);
        using DevBuf y = DevBuf.Empty(c.Dtype, (long)rows * ostr, 0xAB);
        var p = new RmsNormParams { StructSize = (uint)sizeof(RmsNormParams), Input = x.Span, Weight = w.Span, Output = y.Span, Rows = (uint)rows, Dim = (uint)dim, InputStride = (uint)inS, OutputStride = (uint)outS, Eps = c.F("eps") };
        Launch(v, ref p, res);
        using HostBuf got = y.Download();
        PaddingCheck(res, "output", got, 0xAB, rows, dim, ostr);
        res.Outputs["out"] = got.Compact(rows, dim, ostr);
    }

    private void QkNorm(string v, OpCase c, ExecResult res)
    {
        int tokens = c.I("tokens"), qh = c.I("q_heads"), kh = c.I("kv_heads"), hd = c.I("head_dim");
        int qS = c.I("q_stride"), kS = c.I("k_stride");
        int qstr = qS == 0 ? qh * hd : qS, kstr = kS == 0 ? kh * hd : kS;
        using HostBuf qs = c.Buffers["q"].Strided(tokens, qh * hd, qstr, 0xCD);
        using HostBuf ks = c.Buffers["k"].Strided(tokens, kh * hd, kstr, 0xCD);
        using DevBuf q = DevBuf.Upload(qs);
        using DevBuf k = DevBuf.Upload(ks);
        using DevBuf qw = DevBuf.Upload(c.Buffers["qw"]);
        using DevBuf kw = DevBuf.Upload(c.Buffers["kw"]);
        var p = new QkNormParams { StructSize = (uint)sizeof(QkNormParams), Q = q.Span, K = k.Span, QWeight = qw.Span, KWeight = kw.Span, Tokens = (uint)tokens, QHeads = (uint)qh, KvHeads = (uint)kh, HeadDim = (uint)hd, QStride = (uint)qS, KStride = (uint)kS, Eps = c.F("eps") };
        Launch(v, ref p, res);
        using HostBuf gq = q.Download();
        using HostBuf gk = k.Download();
        PaddingCheck(res, "q", gq, 0xCD, tokens, qh * hd, qstr);
        PaddingCheck(res, "k", gk, 0xCD, tokens, kh * hd, kstr);
        res.Outputs["q"] = gq.Compact(tokens, qh * hd, qstr);
        res.Outputs["k"] = gk.Compact(tokens, kh * hd, kstr);
    }

    private void Rope(string v, OpCase c, ExecResult res)
    {
        using DevBuf q = DevBuf.Upload(c.Buffers["q"]);
        using DevBuf k = DevBuf.Upload(c.Buffers["k"]);
        using DevBuf pos = DevBuf.Upload(c.Buffers["pos"]);
        var p = new RopeParams
        {
            StructSize = (uint)sizeof(RopeParams),
            Q = q.Span,
            K = k.Span,
            Positions = pos.Span,
            Tokens = (uint)c.I("tokens"),
            QHeads = (uint)c.I("q_heads"),
            KvHeads = (uint)c.I("kv_heads"),
            HeadDim = (uint)c.I("head_dim"),
            RotaryDim = (uint)c.I("rotary_dim"),
            Kind = (RopeKind)c.I("kind"),
            Interleaved = (uint)c.I("interleaved"),
            Theta = c.F("theta"),
            Llama3Factor = c.F("llama3_factor"),
            Llama3LowFreqFactor = c.F("llama3_low_freq_factor"),
            Llama3HighFreqFactor = c.F("llama3_high_freq_factor"),
            Llama3OriginalMaxPosition = (uint)c.I("llama3_original_max_position"),
        };
        Launch(v, ref p, res);
        res.Outputs["q"] = q.Download();
        res.Outputs["k"] = k.Download();
    }

    private void Attention(string v, OpCase c, ExecResult res)
    {
        HostBuf qiH = c.Buffers["q_indptr"], kiH = c.Buffers["kv_indptr"], lastH = c.Buffers["last_page_len"];
        using DevBuf q = DevBuf.Upload(c.Buffers["q"]);
        using DevBuf kc = DevBuf.Upload(c.Buffers["k_cache"]);
        using DevBuf vc = DevBuf.Upload(c.Buffers["v_cache"]);
        using DevBuf qi = DevBuf.Upload(qiH);
        using DevBuf ki = DevBuf.Upload(kiH);
        using DevBuf idx = DevBuf.Upload(c.Buffers["kv_indices"]);
        using DevBuf last = DevBuf.Upload(lastH);
        using DevBuf o = DevBuf.Empty(c.Dtype, c.Buffers["out"].Count, 0xFF);
        var p = new PagedAttentionParams
        {
            StructSize = (uint)sizeof(PagedAttentionParams),
            Q = q.Span,
            Output = o.Span,
            KCache = kc.Span,
            VCache = vc.Span,
            QIndptr = qi.Span,
            KvIndptr = ki.Span,
            KvIndices = idx.Span,
            KvLastPageLen = last.Span,
            QIndptrHost = qiH.Host,
            KvIndptrHost = kiH.Host,
            KvLastPageLenHost = lastH.Host,
            Mode = (AttentionMode)c.I("mode"),
            Batch = (uint)c.I("batch"),
            Tokens = (uint)c.I("tokens"),
            QHeads = (uint)c.I("q_heads"),
            KvHeads = (uint)c.I("kv_heads"),
            HeadDim = (uint)c.I("head_dim"),
            PageSize = (uint)c.I("page_size"),
            KvLayout = (TensorLayout)c.I("kv_layout"),
            SmScale = 0f,
            LogitsSoftCap = 0f,
            WindowLeft = -1,
            Causal = (uint)c.I("causal"),
        };
        Launch(v, ref p, res);
        res.Outputs["out"] = o.Download();
    }

    private void Silu(string v, OpCase c, ExecResult res)
    {
        int rows = c.I("rows"), dim = c.I("dim");
        using DevBuf x = DevBuf.Upload(c.Buffers["x"]);
        using DevBuf y = DevBuf.Empty(c.Dtype, (long)rows * dim, 0xAB);
        var p = new ActAndMulParams { StructSize = (uint)sizeof(ActAndMulParams), Input = x.Span, Output = y.Span, Rows = (uint)rows, Dim = (uint)dim };
        Launch(v, ref p, res);
        res.Outputs["out"] = y.Download();
    }

    private void Gemm(string v, OpCase c, ExecResult res)
    {
        using DevBuf a = DevBuf.Upload(c.Buffers["a"]);
        using DevBuf b = DevBuf.Upload(c.Buffers["b"]);
        using DevBuf bias = DevBuf.Upload(c.Buffers["bias"]);
        using DevBuf cc = DevBuf.Upload(c.Buffers["c"]);
        using DevBuf d = DevBuf.Empty(c.Dtype, c.Buffers["d"].Count, 0xAB);
        var p = new GemmParams
        {
            StructSize = (uint)sizeof(GemmParams),
            A = a.Span,
            B = b.Span,
            C = c.I("use_c") != 0 ? cc.Span : default,
            D = d.Span,
            Bias = c.I("use_bias") != 0 ? bias.Span : default,
            M = (ulong)c.I("m"),
            N = (ulong)c.I("n"),
            K = (ulong)c.I("k"),
            TransposeB = (uint)c.I("trans_b"),
            Alpha = c.F("alpha"),
            Beta = c.F("beta"),
            ShapeBucket = (uint)c.I("bucket"),
        };
        Launch(v, ref p, res);
        res.Outputs["d"] = d.Download();
    }

    private void Embedding(string v, OpCase c, ExecResult res)
    {
        int tokens = c.I("tokens"), dim = c.I("dim");
        using DevBuf ids = DevBuf.Upload(c.Buffers["ids"]);
        using DevBuf table = DevBuf.Upload(c.Buffers["table"]);
        using DevBuf y = DevBuf.Empty(c.Dtype, (long)tokens * dim, 0xAB);
        var p = new EmbeddingParams { StructSize = (uint)sizeof(EmbeddingParams), Ids = ids.Span, Table = table.Span, Output = y.Span, Tokens = (uint)tokens, Dim = (uint)dim, Vocab = (uint)c.I("vocab") };
        Launch(v, ref p, res);
        res.Outputs["out"] = y.Download();
    }

    private void LmHead(string v, OpCase c, ExecResult res)
    {
        int sel = c.I("selected"), vocab = c.I("vocab"), padded = c.I("vocab_padded");
        using DevBuf h = DevBuf.Upload(c.Buffers["hidden"]);
        using DevBuf rows = DevBuf.Upload(c.Buffers["rows"]);
        using DevBuf w = DevBuf.Upload(c.Buffers["weight"]);
        using DevBuf logits = DevBuf.Empty(DType.F32, (long)sel * padded, 0xAB);
        var p = new LmHeadParams { StructSize = (uint)sizeof(LmHeadParams), Hidden = h.Span, Rows = rows.Span, Weight = w.Span, Logits = logits.Span, Selected = (uint)sel, Dim = (uint)c.I("dim"), Vocab = (uint)vocab, VocabPadded = (uint)padded, SoftCap = c.F("soft_cap"), ShapeBucket = (uint)c.I("bucket") };
        Launch(v, ref p, res);
        using HostBuf got = logits.Download();
        PaddingCheck(res, "logits", got, 0xAB, sel, vocab, padded);
        res.Outputs["logits"] = got.Compact(sel, vocab, padded);
    }

    private void Residual(string v, OpCase c, ExecResult res)
    {
        int n = c.I("elements");
        using DevBuf a = DevBuf.Upload(c.Buffers["a"]);
        using DevBuf b = DevBuf.Upload(c.Buffers["b"]);
        using DevBuf y = DevBuf.Empty(c.Dtype, n, 0xAB);
        var p = new ResidualParams { StructSize = (uint)sizeof(ResidualParams), A = a.Span, B = b.Span, Output = y.Span, Elements = (ulong)n };
        Launch(v, ref p, res);
        res.Outputs["out"] = y.Download();
    }

    private void CacheWrite(string v, OpCase c, ExecResult res)
    {
        using DevBuf k = DevBuf.Upload(c.Buffers["k"]);
        using DevBuf vv = DevBuf.Upload(c.Buffers["v"]);
        using DevBuf kc = DevBuf.Upload(c.Buffers["k_cache"]);
        using DevBuf vc = DevBuf.Upload(c.Buffers["v_cache"]);
        using DevBuf slots = DevBuf.Upload(c.Buffers["slots"]);
        var p = new CacheWriteParams
        {
            StructSize = (uint)sizeof(CacheWriteParams),
            K = k.Span,
            V = vv.Span,
            KCache = kc.Span,
            VCache = vc.Span,
            SlotMapping = slots.Span,
            Tokens = (uint)c.I("tokens"),
            KvHeads = (uint)c.I("kv_heads"),
            HeadDim = (uint)c.I("head_dim"),
            PageSize = (uint)c.I("page_size"),
            KvLayout = (TensorLayout)c.I("kv_layout"),
            KScale = 1f,
            VScale = 1f,
        };
        Launch(v, ref p, res);
        res.Outputs["k_cache"] = kc.Download();
        res.Outputs["v_cache"] = vc.Download();
    }

    private void CacheCopy(string v, OpCase c, ExecResult res)
    {
        bool same = c.I("same_pool") != 0;
        using DevBuf src = DevBuf.Upload(c.Buffers["src"]);
        using DevBuf? dst = same ? null : DevBuf.Upload(c.Buffers["dst"]);
        using DevBuf pairs = DevBuf.Upload(c.Buffers["pairs"]);
        var p = new CacheCopyParams { StructSize = (uint)sizeof(CacheCopyParams), SrcPool = src.Span, DstPool = (dst ?? src).Span, Pairs = pairs.Span, BlockBytes = (ulong)c.I("block_bytes"), Count = (uint)c.I("count") };
        Launch(v, ref p, res);
        res.Outputs["dst"] = (dst ?? src).Download();
    }
}
