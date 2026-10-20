using Starlings.Roost.Abi;
using Starlings.Roost.Models.Tiny;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// An independent double-precision transcription of the roost-tiny-hybrid forward: whole sequences from position 0, no paging,
/// no cache, no state pools, activations never rounded. It is written from the Qwen3.5 layer definition (transformers 5.20.1
/// <c>Qwen3_5GatedDeltaNet</c> and <c>Qwen3_5Attention</c>: the causal convolution with SiLU over the fused q/k/v channels,
/// the L2-normalised q and k, the gated delta rule, the gated RMS norm; the sigmoid output gate of attention), not from the
/// generated code or from CpuOps. The generated forward on CpuOps (bf16 weights and activations, float32 recurrent state) is
/// compared against it.
/// </summary>
/// <remarks>
/// With <c>roundActivations</c> false nothing is rounded, so the difference from the generated forward is the bfloat16 noise of
/// its activations. With it true, every value a backend stores (each op's output) is rounded to bfloat16 the way the
/// definition's dtype says, the decay to float32, and the logits to float32; the remaining difference is then only the order
/// of double accumulation and the rare tie that rounds the other way, which is a much tighter check of the structure.
/// </remarks>
public sealed class TinyHybridReference
{
    private readonly TinyConfig _c;
    private readonly Dictionary<string, double[]> _w;
    private readonly bool _round;

    public TinyHybridReference(TinyConfig config, IReadOnlyDictionary<string, double[]> weights, bool roundActivations = false)
    {
        _round = roundActivations;
        _c = config;
        _w = new Dictionary<string, double[]>(weights);
    }

    /// <summary>Logits at every position of the token sequence, [positions, vocab] (the head is the embedding table).</summary>
    public double[][] Logits(int[] tokens)
    {
        int t = tokens.Length;
        int h = _c.Hidden;
        double[][] x = tokens.Select(id => _w["embed"].AsSpan(id * h, h).ToArray()).ToArray();
        for (int l = 0; l < _c.Layers; l++)
        {
            double[][] n = x.Select(r => B(Norm(r, _w[$"l{l}.input_norm"], 1e-6))).ToArray();
            double[][] mix = _c.LayerType(l) == TinyLayerType.FullAttention ? Attention(l, n) : LinearAttention(l, n);
            for (int p = 0; p < t; p++)
            {
                for (int i = 0; i < h; i++)
                {
                    x[p][i] = B(x[p][i] + mix[p][i]);
                }

                double[] y = B(Norm(x[p], _w[$"l{l}.post_norm"], 1e-6));
                double[] up = B(MatVec(_w[$"l{l}.gate_up"], y, 2 * _c.Intermediate, h));
                var act = new double[_c.Intermediate];
                for (int i = 0; i < act.Length; i++)
                {
                    act[i] = B(Silu(up[i]) * up[_c.Intermediate + i]);
                }

                double[] mlp = B(MatVec(_w[$"l{l}.down"], act, h, _c.Intermediate));
                for (int i = 0; i < h; i++)
                {
                    x[p][i] = B(x[p][i] + mlp[i]);
                }
            }
        }

        return x.Select(r => MatVec(_w["embed"], B(Norm(r, _w["final_norm"], 1e-6)), _c.Vocab, h).Select(v => _round ? (double)(float)v : v).ToArray()).ToArray();
    }

    /// <summary>Greedy continuation by recomputing the whole sequence at every step; first maximum wins.</summary>
    public int[] Greedy(int[] prompt, int count)
    {
        var seq = prompt.ToList();
        for (int i = 0; i < count; i++)
        {
            double[] last = Logits(seq.ToArray())[^1];
            seq.Add(TinyReference.ArgMax(last));
        }

        return seq.Skip(prompt.Length).ToArray();
    }

    // Gated DeltaNet token mixer for one layer; n is the normed input per position.
    private double[][] LinearAttention(int l, double[][] n)
    {
        int t = n.Length;
        int h = _c.Hidden;
        int kh = _c.LinearKeyHeads;
        int vh = _c.LinearValueHeads;
        int kd = _c.LinearKeyDim;
        int vd = _c.LinearValueDim;
        int convDim = _c.LinearConvDim;
        int width = _c.ConvWidth;
        int keyDim = kh * kd;

        // Fused projection, then the depthwise causal convolution over every fused channel (taps oldest first, zero history) and SiLU.
        double[][] qkv = n.Select(r => B(MatVec(_w[$"l{l}.in_proj_qkv"], r, convDim, h))).ToArray();
        double[] cw = _w[$"l{l}.conv"];
        var conv = new double[t][];
        for (int p = 0; p < t; p++)
        {
            conv[p] = new double[convDim];
            for (int c = 0; c < convDim; c++)
            {
                double y = 0;
                for (int j = 0; j < width; j++)
                {
                    int src = p - (width - 1) + j;
                    if (src >= 0)
                    {
                        y += cw[(c * width) + j] * qkv[src][c];
                    }
                }

                conv[p][c] = B(Silu(y));
            }
        }

        // The decay and beta from the raw a and b projections.
        double[] aLog = _w[$"l{l}.A_log"];
        double[] dtBias = _w[$"l{l}.dt_bias"];
        double[][] g = new double[t][];
        double[][] beta = new double[t][];
        for (int p = 0; p < t; p++)
        {
            double[] a = B(MatVec(_w[$"l{l}.in_proj_a"], n[p], vh, h));
            double[] b = B(MatVec(_w[$"l{l}.in_proj_b"], n[p], vh, h));
            g[p] = new double[vh];
            beta[p] = new double[vh];
            for (int head = 0; head < vh; head++)
            {
                double s = a[head] + dtBias[head];
                g[p][head] = _round ? (float)(-Math.Exp(aLog[head]) * (s > 20 ? s : Math.Log(1 + Math.Exp(s)))) : -Math.Exp(aLog[head]) * (s > 20 ? s : Math.Log(1 + Math.Exp(s)));
                beta[p][head] = B(1 / (1 + Math.Exp(-b[head])));
            }
        }

        // The recurrence per value head over S[kd, vd]; q and k are L2-normalised per key head, q is scaled by 1/sqrt(kd).
        var core = new double[t][];
        for (int p = 0; p < t; p++)
        {
            core[p] = new double[vh * vd];
        }

        int group = vh / kh;
        for (int head = 0; head < vh; head++)
        {
            int key = head / group;
            var s = new double[kd, vd];
            for (int p = 0; p < t; p++)
            {
                double[] q = L2(conv[p].AsSpan(key * kd, kd).ToArray());
                double[] k = L2(conv[p].AsSpan(keyDim + (key * kd), kd).ToArray());
                for (int i = 0; i < kd; i++)
                {
                    q[i] /= Math.Sqrt(kd);
                }

                double decay = Math.Exp(g[p][head]);
                for (int j = 0; j < vd; j++)
                {
                    double kv = 0;
                    for (int i = 0; i < kd; i++)
                    {
                        s[i, j] *= decay;
                        kv += s[i, j] * k[i];
                    }

                    double delta = (conv[p][(2 * keyDim) + (head * vd) + j] - kv) * beta[p][head];
                    for (int i = 0; i < kd; i++)
                    {
                        s[i, j] += k[i] * delta;
                    }

                    double o = 0;
                    for (int i = 0; i < kd; i++)
                    {
                        o += s[i, j] * q[i];
                    }

                    core[p][(head * vd) + j] = B(o);
                }
            }
        }

        // The gated RMS norm per value head (norm, then the weight, then silu(z)), then out_proj.
        double[] nw = _w[$"l{l}.linear_norm"];
        var mix = new double[t][];
        for (int p = 0; p < t; p++)
        {
            double[] z = B(MatVec(_w[$"l{l}.in_proj_z"], n[p], vh * vd, h));
            var normed = new double[vh * vd];
            for (int head = 0; head < vh; head++)
            {
                double[] row = Norm(core[p].AsSpan(head * vd, vd).ToArray(), nw, 1e-6);
                for (int j = 0; j < vd; j++)
                {
                    normed[(head * vd) + j] = B(row[j] * Silu(z[(head * vd) + j]));
                }
            }

            mix[p] = B(MatVec(_w[$"l{l}.out_proj"], normed, h, vh * vd));
        }

        return mix;
    }

    // Full attention with the output gate: the query and the gate are separate projections here (the loader splits q_proj).
    private double[][] Attention(int l, double[][] n)
    {
        int t = n.Length;
        int h = _c.Hidden;
        int hd = _c.HeadDim;
        double[][] q = n.Select(r => B(MatVec(_w[$"l{l}.q_proj"], r, _c.Heads * hd, h))).ToArray();
        double[][] gate = n.Select(r => B(MatVec(_w[$"l{l}.q_gate_proj"], r, _c.Heads * hd, h))).ToArray();
        double[][] k = n.Select(r => B(MatVec(_w[$"l{l}.k_proj"], r, _c.KvHeads * hd, h))).ToArray();
        double[][] v = n.Select(r => B(MatVec(_w[$"l{l}.v_proj"], r, _c.KvHeads * hd, h))).ToArray();
        for (int p = 0; p < t; p++)
        {
            PerHeadNorm(q[p], _w[$"l{l}.q_norm"]);
            PerHeadNorm(k[p], _w[$"l{l}.k_norm"]);
            Rotate(q[p], p);
            Rotate(k[p], p);
            B(q[p]);
            B(k[p]);
        }

        int group = _c.Heads / _c.KvHeads;
        var mix = new double[t][];
        for (int p = 0; p < t; p++)
        {
            var ao = new double[_c.Heads * hd];
            for (int head = 0; head < _c.Heads; head++)
            {
                int kvh = head / group;
                var s = new double[p + 1];
                for (int j = 0; j <= p; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < hd; d++)
                    {
                        dot += q[p][(head * hd) + d] * k[j][(kvh * hd) + d];
                    }

                    s[j] = dot / Math.Sqrt(hd);
                }

                double mx = s.Max();
                double z = s.Sum(e => Math.Exp(e - mx));
                for (int d = 0; d < hd; d++)
                {
                    double acc = 0;
                    for (int j = 0; j <= p; j++)
                    {
                        acc += Math.Exp(s[j] - mx) / z * v[j][(kvh * hd) + d];
                    }

                    ao[(head * hd) + d] = B(B(acc) / (1 + Math.Exp(-gate[p][(head * hd) + d])));
                }
            }

            mix[p] = B(MatVec(_w[$"l{l}.o_proj"], ao, h, _c.Heads * hd));
        }

        return mix;
    }

    private double B(double v) => _round ? TinyWeights.Round(v, DType.BF16) : v;

    private double[] B(double[] v)
    {
        if (_round)
        {
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = TinyWeights.Round(v[i], DType.BF16);
            }
        }

        return v;
    }

    private static double Silu(double x) => x / (1 + Math.Exp(-x));

    private static double[] L2(double[] x)
    {
        double inv = 1 / Math.Sqrt(x.Sum(v => v * v) + 1e-6);
        return x.Select(v => v * inv).ToArray();
    }

    private static double[] Norm(double[] x, double[] w, double eps)
    {
        double ss = 0;
        foreach (double v in x)
        {
            ss += v * v;
        }

        double inv = 1 / Math.Sqrt((ss / x.Length) + eps);
        return x.Select((v, i) => v * inv * w[i]).ToArray();
    }

    private void PerHeadNorm(double[] x, double[] w)
    {
        for (int head = 0; head < x.Length / _c.HeadDim; head++)
        {
            B(Norm(x.AsSpan(head * _c.HeadDim, _c.HeadDim).ToArray(), w, 1e-6)).CopyTo(x, head * _c.HeadDim);
        }
    }

    // Partial NeoX rotary embedding: the first RotaryDim elements of each head, element i paired with i + RotaryDim / 2,
    // frequency theta^(-2i / RotaryDim), theta 1e7 (partial_rotary_factor 0.25 of the head).
    private void Rotate(double[] x, int position)
    {
        const double theta = 10_000_000;
        int rot = _c.HeadDim / 4;
        int half = rot / 2;
        for (int head = 0; head < x.Length / _c.HeadDim; head++)
        {
            for (int i = 0; i < half; i++)
            {
                double angle = position * Math.Pow(theta, -2.0 * i / rot);
                double cos = Math.Cos(angle);
                double sin = Math.Sin(angle);
                int a = (head * _c.HeadDim) + i;
                int b = a + half;
                double x1 = x[a];
                double x2 = x[b];
                x[a] = (x1 * cos) - (x2 * sin);
                x[b] = (x2 * cos) + (x1 * sin);
            }
        }
    }

    private static double[] MatVec(double[] w, double[] x, int rows, int cols)
    {
        var y = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double s = 0;
            for (int c = 0; c < cols; c++)
            {
                s += w[(r * cols) + c] * x[c];
            }

            y[r] = s;
        }

        return y;
    }
}
