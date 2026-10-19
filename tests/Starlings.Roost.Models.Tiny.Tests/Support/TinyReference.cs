using Starlings.Roost.Models.Tiny;

namespace Starlings.Roost.Models.Tiny.Tests;

/// <summary>
/// An independent double-precision transcription of the roost-tiny forward: whole sequences, no paging, no cache, written
/// from the Qwen3 layer definition rather than from the generated code. The generated forward on CpuOps (paged cache,
/// activations stored as float32) is compared against it.
/// </summary>
public sealed class TinyReference
{
    private readonly TinyConfig _c;
    private readonly Dictionary<string, double[]> _w;

    public TinyReference(TinyConfig config, IReadOnlyDictionary<string, double[]> weights)
    {
        _c = config;
        _w = new Dictionary<string, double[]>(weights);
    }

    /// <summary>Logits at every position of the token sequence, [positions, vocab].</summary>
    public double[][] Logits(int[] tokens)
    {
        int t = tokens.Length;
        int h = _c.Hidden;
        double[][] x = tokens.Select(id => _w["embed"].AsSpan(id * h, h).ToArray()).ToArray();
        for (int l = 0; l < _c.Layers; l++)
        {
            double[][] n = x.Select(r => Norm(r, _w[$"l{l}.input_norm"])).ToArray();
            double[][] q = n.Select(r => MatVec(_w[$"l{l}.q_proj"], r, _c.Heads * _c.HeadDim, h)).ToArray();
            double[][] k = n.Select(r => MatVec(_w[$"l{l}.k_proj"], r, _c.KvHeads * _c.HeadDim, h)).ToArray();
            double[][] v = n.Select(r => MatVec(_w[$"l{l}.v_proj"], r, _c.KvHeads * _c.HeadDim, h)).ToArray();
            for (int p = 0; p < t; p++)
            {
                PerHeadNorm(q[p], _w[$"l{l}.q_norm"]);
                PerHeadNorm(k[p], _w[$"l{l}.k_norm"]);
                Rotate(q[p], p);
                Rotate(k[p], p);
            }

            double[][] ao = new double[t][];
            int group = _c.Heads / _c.KvHeads;
            for (int p = 0; p < t; p++)
            {
                ao[p] = new double[_c.Heads * _c.HeadDim];
                for (int head = 0; head < _c.Heads; head++)
                {
                    int kvh = head / group;
                    var s = new double[p + 1];
                    for (int j = 0; j <= p; j++)
                    {
                        double dot = 0;
                        for (int d = 0; d < _c.HeadDim; d++)
                        {
                            dot += q[p][(head * _c.HeadDim) + d] * k[j][(kvh * _c.HeadDim) + d];
                        }

                        s[j] = dot / Math.Sqrt(_c.HeadDim);
                    }

                    double mx = s.Max();
                    double z = s.Sum(e => Math.Exp(e - mx));
                    for (int d = 0; d < _c.HeadDim; d++)
                    {
                        double acc = 0;
                        for (int j = 0; j <= p; j++)
                        {
                            acc += Math.Exp(s[j] - mx) / z * v[j][(kvh * _c.HeadDim) + d];
                        }

                        ao[p][(head * _c.HeadDim) + d] = acc;
                    }
                }
            }

            for (int p = 0; p < t; p++)
            {
                double[] o = MatVec(_w[$"l{l}.o_proj"], ao[p], h, _c.Heads * _c.HeadDim);
                for (int i = 0; i < h; i++)
                {
                    x[p][i] += o[i];
                }

                double[] y = Norm(x[p], _w[$"l{l}.post_norm"]);
                double[] up = MatVec(_w[$"l{l}.gate_up"], y, 2 * _c.Intermediate, h);
                var act = new double[_c.Intermediate];
                for (int i = 0; i < act.Length; i++)
                {
                    double g = up[i];
                    act[i] = g / (1 + Math.Exp(-g)) * up[_c.Intermediate + i];
                }

                double[] mlp = MatVec(_w[$"l{l}.down"], act, h, _c.Intermediate);
                for (int i = 0; i < h; i++)
                {
                    x[p][i] += mlp[i];
                }
            }
        }

        return x.Select(r => MatVec(_w["lm_head"], Norm(r, _w["final_norm"]), _c.Vocab, h)).ToArray();
    }

    /// <summary>Greedy continuation by recomputing the whole sequence at every step; first maximum wins.</summary>
    public int[] Greedy(int[] prompt, int count)
    {
        var seq = prompt.ToList();
        for (int i = 0; i < count; i++)
        {
            double[] last = Logits(seq.ToArray())[^1];
            seq.Add(ArgMax(last));
        }

        return seq.Skip(prompt.Length).ToArray();
    }

    public static int ArgMax(IReadOnlyList<double> values)
    {
        int best = 0;
        for (int i = 1; i < values.Count; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static double[] Norm(double[] x, double[] w)
    {
        double ss = 0;
        foreach (double v in x)
        {
            ss += v * v;
        }

        double inv = 1 / Math.Sqrt((ss / x.Length) + 1e-6);
        return x.Select((v, i) => v * inv * w[i]).ToArray();
    }

    private void PerHeadNorm(double[] x, double[] w)
    {
        for (int head = 0; head < x.Length / _c.HeadDim; head++)
        {
            double[] part = Norm(x.AsSpan(head * _c.HeadDim, _c.HeadDim).ToArray(), w);
            part.CopyTo(x, head * _c.HeadDim);
        }
    }

    // NeoX rotary embedding over the whole head: element i pairs with i + HeadDim / 2, frequency theta^(-2i / HeadDim).
    private void Rotate(double[] x, int position)
    {
        const double theta = 1_000_000;
        int half = _c.HeadDim / 2;
        for (int head = 0; head < x.Length / _c.HeadDim; head++)
        {
            for (int i = 0; i < half; i++)
            {
                double angle = position * Math.Pow(theta, -2.0 * i / _c.HeadDim);
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

    // y = W x with W stored [rows, cols] row-major (the checkpoint layout, TransposeB in the definition).
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
