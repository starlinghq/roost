using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>Embedding, CacheWrite and CacheCopy: data-movement ops, compared bit for bit with CpuOps.</summary>
internal static unsafe class DataCases
{
    public static void Run(Ctx c)
    {
        foreach (DType dt in new[] { DType.BF16, DType.F16, DType.F32 })
        {
            Embedding(c, dt, 4096, 1024, 1);
            Embedding(c, dt, 4096, 1024, 300);
        }

        Embedding(c, DType.BF16, 4096, 1001, 17);
        Embedding(c, DType.BF16, 4096, 1002, 17);
        EmbeddingOutOfRange(c);
        foreach (TensorLayout layout in new[] { TensorLayout.PagedNhd, TensorLayout.PagedHnd })
        {
            foreach (int ps in new[] { 16, 32, 64 })
            {
                CacheWrite(c, DType.BF16, layout, ps, 37);
            }
        }

        CacheWrite(c, DType.F32, TensorLayout.PagedNhd, 16, 9);
        CacheWriteBounds(c);
        CacheCopy(c, 16384, sameBuffer: true);
        CacheCopy(c, 16384, sameBuffer: false);
        CacheCopy(c, 1001, sameBuffer: true);
        CacheCopyBounds(c);
    }

    private static void Embedding(Ctx c, DType dt, int vocab, int dim, int tokens)
    {
        using var table = new Buf(dt, (long)vocab * dim).Randn(c.Rng);
        using var ids = new Buf(DType.I32, tokens);
        for (int i = 0; i < tokens; i++)
        {
            ids.As<int>()[i] = i % 7 == 0 ? vocab - 1 : c.Rng.Int(0, vocab);
        }

        using var outBuf = new Buf(dt, (long)tokens * dim);
        table.Up();
        ids.Up();
        outBuf.FillDevice(0xAB);
        var p = new EmbeddingParams { StructSize = (uint)sizeof(EmbeddingParams), Ids = ids.DevSpan, Table = table.DevSpan, Output = outBuf.DevSpan, Tokens = (uint)tokens, Dim = (uint)dim, Vocab = (uint)vocab };
        c.Lib.Exec("embedding-roost", ref p, c.Stream);
        using var got = Ctx.Fetch(outBuf);
        using var want = new Buf(dt, (long)tokens * dim);
        CpuOps.Embedding(new EmbeddingArgs { Ids = ids.HostSpan, Table = table.HostSpan, Output = want.HostSpan, Tokens = tokens, Dim = dim, Vocab = vocab, DType = dt });
        c.Report(Compare.Bits($"Embedding {dt} vocab={vocab} dim={dim} tokens={tokens}", got, want));
    }

    private static void EmbeddingOutOfRange(Ctx c)
    {
        const int vocab = 64, dim = 128;
        using var table = new Buf(DType.BF16, vocab * dim).Randn(c.Rng);
        using var ids = Buf.Ints(3, -1, vocab, int.MaxValue, 5);
        using var outBuf = new Buf(DType.BF16, 5 * dim);
        table.Up();
        ids.Up();
        outBuf.FillDevice(0xAB);
        var p = new EmbeddingParams { StructSize = (uint)sizeof(EmbeddingParams), Ids = ids.DevSpan, Table = table.DevSpan, Output = outBuf.DevSpan, Tokens = 5, Dim = dim, Vocab = vocab };
        c.Lib.Exec("embedding-roost", ref p, c.Stream);
        using var got = Ctx.Fetch(outBuf);
        bool ok = true;
        var o = got.As<ushort>();
        var t = table.As<ushort>();
        for (int d = 0; d < dim; d++)
        {
            ok &= o[d] == t[(3 * dim) + d] && o[(4 * dim) + d] == t[(5 * dim) + d];
            ok &= o[dim + d] == 0 && o[(2 * dim) + d] == 0 && o[(3 * dim) + d] == 0;
        }

        c.Check("Embedding out-of-range ids give zero rows, no fault", ok);
    }

    private static void CacheWrite(Ctx c, DType dt, TensorLayout layout, int ps, int tokens)
    {
        const int kh = 2, hd = 256;
        int pages = 40;
        long poolElems = (long)pages * ps * kh * hd;
        using var k = new Buf(dt, (long)tokens * kh * hd).Randn(c.Rng);
        using var v = new Buf(dt, (long)tokens * kh * hd).Randn(c.Rng);
        using var kc = new Buf(dt, poolElems);
        using var vc = new Buf(dt, poolElems);
        var slots = new long[tokens];
        var used = new HashSet<long>();
        for (int t = 0; t < tokens; t++)
        {
            long s;
            do
            {
                s = c.Rng.Int(0, pages * ps);
            }
            while (!used.Add(s));
            slots[t] = t % 11 == 5 ? -1 : s;
        }

        using var slot = Buf.Longs(slots);
        new Span<byte>((void*)kc.Host, (int)kc.Bytes).Fill(0xAB);
        new Span<byte>((void*)vc.Host, (int)vc.Bytes).Fill(0xAB);
        using var wantK = kc.Clone();
        using var wantV = vc.Clone();
        k.Up();
        v.Up();
        kc.Up();
        vc.Up();
        slot.Up();
        var p = new CacheWriteParams { StructSize = (uint)sizeof(CacheWriteParams), K = k.DevSpan, V = v.DevSpan, KCache = kc.DevSpan, VCache = vc.DevSpan, SlotMapping = slot.DevSpan, Tokens = (uint)tokens, KvHeads = kh, HeadDim = hd, PageSize = (uint)ps, KvLayout = layout, KScale = 1f, VScale = 1f };
        c.Lib.Exec("cache-write-roost", ref p, c.Stream);
        using var gotK = Ctx.Fetch(kc);
        using var gotV = Ctx.Fetch(vc);
        CpuOps.CacheWrite(new CacheWriteArgs { K = k.HostSpan, V = v.HostSpan, KCache = wantK.HostSpan, VCache = wantV.HostSpan, SlotMapping = slot.HostSpan, Tokens = tokens, KvHeads = kh, HeadDim = hd, BlockSize = ps, KvLayout = layout, KScale = 1f, VScale = 1f, KvDType = dt });
        c.Report(Compare.Bits($"CacheWrite {dt} {layout} page={ps} tokens={tokens} K pool", gotK, wantK));
        c.Report(Compare.Bits($"CacheWrite {dt} {layout} page={ps} tokens={tokens} V pool", gotV, wantV));
    }

    private static void CacheWriteBounds(Ctx c)
    {
        const int kh = 2, hd = 256, ps = 16, pages = 4;
        using var k = new Buf(DType.BF16, 3L * kh * hd).Randn(c.Rng);
        using var v = new Buf(DType.BF16, 3L * kh * hd).Randn(c.Rng);
        using var kc = new Buf(DType.BF16, (long)pages * ps * kh * hd);
        using var vc = new Buf(DType.BF16, (long)pages * ps * kh * hd);
        using var slot = Buf.Longs(pages * ps, 1L << 40, 5);
        new Span<byte>((void*)kc.Host, (int)kc.Bytes).Fill(0xAB);
        new Span<byte>((void*)vc.Host, (int)vc.Bytes).Fill(0xAB);
        k.Up();
        v.Up();
        kc.Up();
        vc.Up();
        slot.Up();
        var p = new CacheWriteParams { StructSize = (uint)sizeof(CacheWriteParams), K = k.DevSpan, V = v.DevSpan, KCache = kc.DevSpan, VCache = vc.DevSpan, SlotMapping = slot.DevSpan, Tokens = 3, KvHeads = kh, HeadDim = hd, PageSize = ps, KvLayout = TensorLayout.PagedNhd, KScale = 1f, VScale = 1f };
        c.Lib.Exec("cache-write-roost", ref p, c.Stream);
        using var got = Ctx.Fetch(kc);
        var raw = got.As<byte>();
        long rowBytes = (long)kh * hd * 2;
        bool onlySlot5 = true;
        for (long i = 0; i < raw.Length; i++)
        {
            bool inSlot5 = i >= 5 * rowBytes && i < 6 * rowBytes;
            onlySlot5 &= inSlot5 || raw[(int)i] == 0xAB;
        }

        c.Check("CacheWrite out-of-pool slots are skipped, in-pool slot written", onlySlot5);
        var q = p;
        q.KScale = 0.5f;
        c.Check("CacheWrite k_scale != 1 -> UNSUPPORTED", c.Lib.Try("cache-write-roost", ref q, c.Stream) == (int)KernelStatus.Unsupported);
        var m = p;
        m.KCache.Dtype = DType.F16;
        c.Check("CacheWrite cache dtype != activation dtype -> UNSUPPORTED", c.Lib.Try("cache-write-roost", ref m, c.Stream) == (int)KernelStatus.Unsupported);
        Cuda.Sync(c.Stream);
        Cuda.ClearError();
    }

    private static void CacheCopy(Ctx c, int blockBytes, bool sameBuffer)
    {
        const int blocks = 24;
        long elems = (long)blocks * blockBytes;
        using var src = new Buf(DType.U8, elems);
        for (long i = 0; i < elems; i++)
        {
            src.As<byte>()[(int)i] = (byte)c.Rng.Int(0, 256);
        }

        Buf dst = sameBuffer ? src : new Buf(DType.U8, elems);
        if (!sameBuffer)
        {
            new Span<byte>((void*)dst.Host, (int)dst.Bytes).Fill(0xAB);
        }

        // Independent pairs: sources 0..4 go to 10..14 (and 20 receives block 7); a self pair is a no-op.
        using var pairs = Buf.Ints(0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 7, 20, 8, 8);
        using var want = sameBuffer ? src.Clone() : dst.Clone();
        try
        {
        src.Up();
        if (!sameBuffer)
        {
            dst.Up();
        }

        pairs.Up();
        var p = new CacheCopyParams { StructSize = (uint)sizeof(CacheCopyParams), SrcPool = src.DevSpan, DstPool = dst.DevSpan, Pairs = pairs.DevSpan, BlockBytes = (ulong)blockBytes, Count = 7 };
        c.Lib.Exec("cache-copy-roost", ref p, c.Stream);
        using var got = Ctx.Fetch(dst);
        // The reference reads the original source (the clone itself for the same-pool case) and writes the clone.
        CpuOps.CacheCopy(new CacheCopyArgs { Source = sameBuffer ? want.HostSpan : src.HostSpan, Destination = want.HostSpan, Pairs = pairs.HostSpan, BlockBytes = blockBytes, Count = 7 });
        c.Report(Compare.Bits($"CacheCopy block_bytes={blockBytes} {(sameBuffer ? "same pool" : "two pools")}", got, want));
        }
        finally
        {
            if (!sameBuffer)
            {
                dst.Dispose();
            }
        }
    }

    private static void CacheCopyBounds(Ctx c)
    {
        const int blockBytes = 256, blocks = 4;
        using var pool = new Buf(DType.U8, (long)blocks * blockBytes);
        for (int i = 0; i < pool.Bytes; i++)
        {
            pool.As<byte>()[i] = (byte)(i / blockBytes + 1);
        }

        using var pairs = Buf.Ints(0, 9, -1, 1, 1, 2, 5, 3);
        using var want = pool.Clone();
        pool.Up();
        pairs.Up();
        var p = new CacheCopyParams { StructSize = (uint)sizeof(CacheCopyParams), SrcPool = pool.DevSpan, DstPool = pool.DevSpan, Pairs = pairs.DevSpan, BlockBytes = blockBytes, Count = 4 };
        c.Lib.Exec("cache-copy-roost", ref p, c.Stream);
        using var got = Ctx.Fetch(pool);
        var o = got.As<byte>();
        bool ok = true;
        for (int b = 0; b < blocks; b++)
        {
            byte expect = (byte)(b == 2 ? 2 : b + 1); // only the pair (1 -> 2) is inside the pool
            for (int i = 0; i < blockBytes; i++)
            {
                ok &= o[(b * blockBytes) + i] == expect;
            }
        }

        c.Check("CacheCopy pairs outside the pool are skipped, valid pair copied", ok);
    }
}
