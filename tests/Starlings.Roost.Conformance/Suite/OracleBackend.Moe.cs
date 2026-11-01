using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// The MoE ops of the oracle, written from the C3 definitions (R-080) without CpuOps: routing sorts the experts of each token by
/// (logit descending, index ascending); the permutation collects pairs per expert in flat order; FusedMoe runs one token at a time
/// through its chosen experts, rounding each stage's output to bf16 once (<see cref="RoundBf16"/>, by scaling and
/// <see cref="Math.Round(double, MidpointRounding)"/>), and never builds the permuted matrix. Only where the roundings fall is shared
/// with CpuOps, because it is part of the definition.
/// </summary>
/// <remarks>
/// The intermediates are not stored through <see cref="HostBuf.Set"/>, which rounds double to float to bf16: that double rounding moves
/// about one value in 2^16 by one bf16 ulp, and one such flip in a gate or up value, carried through SiLU, the down projection and the
/// combine, put one element of the full profile's 16-token Qwen3-width FusedMoe case at utilisation 1.14 of <c>gemm-bf16</c>.
/// </remarks>
public sealed partial class OracleBackend
{
    private static int[] TopExperts(HostBuf logits, long row, int experts, int topK)
    {
        int[] order = Enumerable.Range(0, experts).ToArray();
        Array.Sort(order, (x, y) =>
        {
            int byValue = logits.Get(row + y).CompareTo(logits.Get(row + x));
            return byValue != 0 ? byValue : x.CompareTo(y);
        });
        return order[..topK];
    }

    /// <summary>The routing weights of one token in double, before they are stored.</summary>
    private static double[] RouteWeights(HostBuf logits, long row, int experts, int[] chosen, bool renormalize)
    {
        double max = double.NegativeInfinity;
        for (int j = 0; j < experts; j++)
        {
            max = Math.Max(max, logits.Get(row + j));
        }

        double denom = 0;
        for (int j = 0; j < experts; j++)
        {
            denom += Math.Exp(logits.Get(row + j) - max);
        }

        double[] p = [.. chosen.Select(j => Math.Exp(logits.Get(row + j) - max) / denom)];
        double sum = p.Sum();
        return renormalize ? [.. p.Select(v => v / sum)] : p;
    }

    private static void Route(OpCase c, Dictionary<string, HostBuf> b)
    {
        int experts = c.I("experts"), topK = c.I("top_k");
        for (int t = 0; t < c.I("tokens"); t++)
        {
            long row = (long)t * experts;
            int[] chosen = TopExperts(b["logits"], row, experts, topK);
            double[] w = RouteWeights(b["logits"], row, experts, chosen, c.I("renormalize") != 0);
            for (int k = 0; k < topK; k++)
            {
                b["ids"].Set(((long)t * topK) + k, chosen[k]);
                b["weights"].Set(((long)t * topK) + k, w[k]);
            }
        }
    }

    private static void Permute(OpCase c, Dictionary<string, HostBuf> b)
    {
        int experts = c.I("experts"), topK = c.I("top_k"), dim = c.I("dim");
        int pairs = c.I("tokens") * topK;
        var buckets = Enumerable.Range(0, experts).Select(_ => new List<int>()).ToArray();
        for (int i = 0; i < pairs; i++)
        {
            buckets[(int)b["ids"].Get(i)].Add(i);
        }

        int row = 0;
        int es = Elem.Size(b["x"].Dtype);
        for (int e = 0; e < experts; e++)
        {
            b["counts"].Set(e, buckets[e].Count);
            b["offsets"].Set(e, row);
            foreach (int i in buckets[e])
            {
                b["perm"].Set(i, row);
                b["x"].Raw().Slice(i / topK * dim * es, dim * es).CopyTo(b["out"].Raw()[(row * dim * es)..]);
                row++;
            }
        }

        b["offsets"].Set(experts, row);
    }

    private static void Unpermute(OpCase c, Dictionary<string, HostBuf> b)
    {
        int topK = c.I("top_k"), dim = c.I("dim");
        for (int t = 0; t < c.I("tokens"); t++)
        {
            for (int d = 0; d < dim; d++)
            {
                double acc = 0;
                for (int k = 0; k < topK; k++)
                {
                    long pair = ((long)t * topK) + k;
                    acc += b["weights"].Get(pair) * b["y"].Get((long)b["perm"].Get(pair) * dim + d);
                }

                b["out"].Set((long)t * dim + d, acc);
            }
        }
    }

    private static void GroupedGemm(OpCase c, Dictionary<string, HostBuf> b)
    {
        int experts = c.I("experts"), n = c.I("n"), k = c.I("k");
        for (int e = 0; e < experts; e++)
        {
            for (long r = (long)b["offsets"].Get(e); r < (long)b["offsets"].Get(e + 1); r++)
            {
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int p = 0; p < k; p++)
                    {
                        acc += b["a"].Get(r * k + p) * b["b"].Get(((long)e * n + j) * k + p);
                    }

                    b["d"].Set(r * n + j, acc);
                }
            }
        }
    }

    /// <summary>Rounds a finite, normal-range double to the nearest bfloat16 (ties to even) in one step.</summary>
    private static double RoundBf16(double v)
    {
        if (v == 0 || !double.IsFinite(v))
        {
            return v;
        }

        double scale = Math.ScaleB(1.0, 7 - Math.ILogB(v));
        return Math.Round(v * scale, MidpointRounding.ToEven) / scale;
    }

    private static void FusedMoe(OpCase c, Dictionary<string, HostBuf> b)
    {
        int experts = c.I("experts"), topK = c.I("top_k"), dim = c.I("dim"), inter = c.I("inter");
        HostBuf x = b["x"], w13 = b["w13"], w2 = b["w2"];
        if (x.Dtype != DType.BF16)
        {
            throw new NotSupportedException("The oracle's FusedMoe rounds its intermediates to bf16 only.");
        }

        var gateUp = new double[2 * inter];
        var act = new double[inter];
        var y = new double[dim];
        var acc = new double[dim];
        for (int t = 0; t < c.I("tokens"); t++)
        {
            long row = (long)t * experts;
            int[] chosen = TopExperts(b["logits"], row, experts, topK);
            double[] w = RouteWeights(b["logits"], row, experts, chosen, c.I("renormalize") != 0);
            Array.Clear(acc);
            for (int s = 0; s < topK; s++)
            {
                double weight = (float)w[s];
                long e = chosen[s];
                for (int i = 0; i < 2 * inter; i++)
                {
                    double v = 0;
                    for (int d = 0; d < dim; d++)
                    {
                        v += x.Get((long)t * dim + d) * w13.Get((e * 2 * inter + i) * dim + d);
                    }

                    gateUp[i] = RoundBf16(v);
                }

                for (int i = 0; i < inter; i++)
                {
                    double g = gateUp[i];
                    act[i] = RoundBf16(g / (1.0 + Math.Exp(-g)) * gateUp[inter + i]);
                }

                for (int d = 0; d < dim; d++)
                {
                    double v = 0;
                    for (int i = 0; i < inter; i++)
                    {
                        v += act[i] * w2.Get((e * dim + d) * inter + i);
                    }

                    y[d] = RoundBf16(v);
                }

                for (int d = 0; d < dim; d++)
                {
                    acc[d] += weight * y[d];
                }
            }

            for (int d = 0; d < dim; d++)
            {
                b["out"].Set((long)t * dim + d, RoundBf16(acc[d]));
            }
        }
    }
}
