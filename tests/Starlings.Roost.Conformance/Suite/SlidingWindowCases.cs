using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// Sliding-window PagedAttention cases (R-113): the M0 attention shapes with <c>window_left</c> set, so a query at position p reads
/// keys p - window_left to p. The windows are synthetic (no model's window is claimed): window boundaries just below, at and above
/// the context and on page edges, a window of 1, a window longer than every context, decode and chunked prefill, non-causal, and
/// cases whose leading pages are released (-1 in kv_indices) as the block manager's sliding-window release leaves them.
/// <c>paged-attention-flashinfer-fa2</c> refuses every one of them (<see cref="RefusalRules.PagedAttentionSlidingWindow"/>); on the
/// CPU half the independent oracle checks <c>CpuOps</c> on them.
/// </summary>
public static partial class CaseCatalog
{
    private static IEnumerable<OpCase> SlidingWindowAttentionCases(ulong baseSeed, CaseProfile p)
    {
        const TensorLayout Nhd = TensorLayout.PagedNhd;
        const TensorLayout Hnd = TensorLayout.PagedHnd;
        const AttentionMode Decode1 = AttentionMode.Decode;
        const AttentionMode Prefill = AttentionMode.Prefill;

        // Decode, window 32 against contexts of 31 to 64: the window covers all of the first three, then starts to slide; 33 keys
        // are exactly a window, and 48 and 64 put its start mid-page and on a page edge.
        yield return Attention("window-decode-edges", Decode1, Decode(31, 32, 33, 34, 48, 64), 16, Nhd, true, baseSeed, 32);
        yield return Attention("window-decode-edges", Decode1, Decode(31, 32, 33, 34, 48, 64), 16, Nhd, true, baseSeed, 32, release: true);
        yield return Attention("window-decode-mixed4", Decode1, Decode(5, 37, 130, 16), 16, Hnd, true, baseSeed, 31, release: true);
        yield return Attention("window-1-decode", Decode1, Decode(1, 2, 17, 40), 16, Hnd, true, baseSeed, 1, release: true);
        yield return Attention("window-longer-than-context-decode", Decode1, Decode(5, 37, 130, 16), 32, Nhd, true, baseSeed, 4096);
        yield return Attention("window-prefill-fresh-100", Prefill, [new(100, 100)], 16, Nhd, true, baseSeed, 15);
        yield return Attention("window-1-prefill-fresh-40", Prefill, [new(40, 40)], 16, Hnd, true, baseSeed, 1);
        yield return Attention("window-prefill-chunk-64-of-300", Prefill, [new(64, 300)], 32, Hnd, true, baseSeed, 63, release: true);
        yield return Attention("window-prefill-multi-seq", Prefill, [new(17, 17), new(64, 300), new(1, 50), new(60, 60)], 16, Nhd, true, baseSeed, 24, release: true);
        yield return Attention("window-longer-than-context-prefill", Prefill, [new(33, 33), new(5, 21)], 16, Nhd, true, baseSeed, 4096);
        yield return Attention("window-prefill-non-causal-32-of-200", Prefill, [new(32, 200)], 16, Nhd, false, baseSeed, 40, release: true);
        if (p.Large)
        {
            yield return Attention("window-decode-single-6000", Decode1, Decode(6000), 16, Nhd, true, baseSeed, 1023, release: true);
            yield return Attention("window-prefill-chunk-128-of-1000", Prefill, [new(128, 1000)], 64, Nhd, true, baseSeed, 511, release: true);
        }

        // A stream of its own, so the earlier random cases keep their shapes and seeds.
        var rng = new Rng(SeedFor(OpId.PagedAttention, "random-window-shapes", baseSeed));
        for (int i = 0; i < Math.Max(1, p.RandomPerOp / 2); i++)
        {
            int ps = rng.Pick(16, 32, 64);
            TensorLayout layout = rng.Pick(Nhd, Hnd);
            int batch = rng.Int(1, 6);
            int cap = p.Large ? 700 : 150;
            int window = rng.Pick(1, rng.Int(1, cap), rng.Int(1, 64));
            bool release = rng.Int(0, 2) == 0;
            if (rng.Int(0, 2) == 0)
            {
                yield return Attention($"random-window-{i}-decode", Decode1, Enumerable.Range(0, batch).Select(_ => new Seq(1, rng.Int(1, cap))).ToArray(), ps, layout, true, baseSeed, window, release);
            }
            else
            {
                var seqs = Enumerable.Range(0, batch).Select(_ =>
                {
                    int kv = rng.Int(1, cap);
                    return new Seq(rng.Int(1, kv + 1), kv);
                }).ToArray();
                yield return Attention($"random-window-{i}-prefill", Prefill, seqs, ps, layout, rng.Int(0, 4) != 0, baseSeed, window, release);
            }
        }
    }
}
