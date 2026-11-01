using Starlings.Roost.Abi;
using static Starlings.Roost.Ops.CpuElements;

namespace Starlings.Roost.Ops;

/// <summary>The inverse frequencies of the rotary pairs and the factor cos and sin are multiplied by.</summary>
/// <param name="InverseFrequencies">One frequency per rotary pair, RotaryDim / 2 of them, in double.</param>
/// <param name="AttentionFactor">The factor applied to cos and sin (YaRN's mscale, longrope's scaling); 1 for the other kinds.</param>
public sealed record RopeFrequencies(IReadOnlyList<double> InverseFrequencies, double AttentionFactor);

/// <summary>Parameters of YaRN, named as in transformers' <c>rope_parameters</c>.</summary>
public sealed record YarnParameters
{
    /// <summary>Rotated dimensions (head_dim * partial_rotary_factor); even.</summary>
    public required int RotaryDim { get; init; }

    /// <summary><c>rope_theta</c>.</summary>
    public required double Theta { get; init; }

    /// <summary><c>factor</c>; null means <c>MaxPosition / OriginalMaxPosition</c>, as transformers does.</summary>
    public double? Factor { get; init; }

    /// <summary><c>original_max_position_embeddings</c>.</summary>
    public required int OriginalMaxPosition { get; init; }

    /// <summary><c>max_position_embeddings</c>; read only when <see cref="Factor"/> is null.</summary>
    public int MaxPosition { get; init; }

    /// <summary><c>attention_factor</c>; null derives it from the factor and the two mscales.</summary>
    public double? AttentionFactor { get; init; }

    /// <summary><c>mscale</c>; used only with <see cref="MscaleAllDim"/>, and null or 0 means absent.</summary>
    public double? Mscale { get; init; }

    /// <summary><c>mscale_all_dim</c>; null or 0 means absent.</summary>
    public double? MscaleAllDim { get; init; }

    /// <summary><c>beta_fast</c>; 0 means 32.</summary>
    public double BetaFast { get; init; } = 32;

    /// <summary><c>beta_slow</c>; 0 means 1.</summary>
    public double BetaSlow { get; init; } = 1;

    /// <summary><c>truncate</c>: round the correction range outward to whole dimensions.</summary>
    public bool Truncate { get; init; } = true;
}

/// <summary>Parameters of longrope, named as in transformers' <c>rope_parameters</c>.</summary>
public sealed record LongRopeParameters
{
    /// <summary>Rotated dimensions; even.</summary>
    public required int RotaryDim { get; init; }

    /// <summary><c>rope_theta</c>.</summary>
    public required double Theta { get; init; }

    /// <summary><c>short_factor</c>, RotaryDim / 2 values.</summary>
    public required IReadOnlyList<double> ShortFactor { get; init; }

    /// <summary><c>long_factor</c>, RotaryDim / 2 values.</summary>
    public required IReadOnlyList<double> LongFactor { get; init; }

    /// <summary><c>original_max_position_embeddings</c>: a sequence longer than this uses the long factors.</summary>
    public required int OriginalMaxPosition { get; init; }

    /// <summary><c>max_position_embeddings</c>; read only when <see cref="Factor"/> is null.</summary>
    public int MaxPosition { get; init; }

    /// <summary><c>factor</c>, which only sets the default attention factor; null means <c>MaxPosition / OriginalMaxPosition</c>.</summary>
    public double? Factor { get; init; }

    /// <summary><c>attention_factor</c>; null means <c>sqrt(1 + ln(factor) / ln(OriginalMaxPosition))</c>, or 1 when factor &lt;= 1.</summary>
    public double? AttentionFactor { get; init; }
}

/// <summary>
/// Builds the cos/sin cache <see cref="CpuOps.RopeScaled"/> reads, in double, for YaRN, longrope, dynamic NTK and llama3 (R-098).
/// </summary>
/// <remarks>
/// <para>Each builder is the formula of the matching function of transformers 5.20.1 <c>modeling_rope_utils.py</c> (sha256
/// 6a072707…, <c>_compute_yarn_parameters</c>, <c>_compute_longrope_parameters</c>, <c>_compute_dynamic_ntk_parameters</c>,
/// <c>_compute_llama3_parameters</c>), computed in double where transformers computes in float32.
/// <c>RopeScalingTests</c> checks them against values recorded from those functions and from the same formulas in float64
/// (<c>tests/Starlings.Roost.Ops.Tests/Reference/make_rope_scaling_cases.py</c>).</para>
/// <para>A cache row for position p holds RotaryDim values: <c>cos(p * f_i) * AttentionFactor</c> for the RotaryDim / 2 pairs,
/// then <c>sin(p * f_i) * AttentionFactor</c>, the layout of <see cref="RopeScaledArgs.CosSinCache"/>.</para>
/// <para>The length-dependent kinds are the caller's choice of cache, as in transformers' <c>dynamic_rope_update</c>: for
/// dynamic NTK the cache is built for the sequence length it serves (transformers uses the largest position + 1, and keeps
/// the unscaled frequencies at or below <c>MaxPosition</c>); for longrope the long factors apply when that length exceeds
/// <c>OriginalMaxPosition</c>.</para>
/// <para>These builders allocate; they run once per model load or per length change, not per step.</para>
/// </remarks>
public static class RopeScaling
{
    /// <summary>Unscaled frequencies <c>f_i = theta^(-2i / RotaryDim)</c>.</summary>
    /// <param name="rotaryDim">Rotated dimensions; even and positive.</param>
    /// <param name="theta">Base.</param>
    /// <returns>The frequencies, attention factor 1.</returns>
    public static RopeFrequencies Default(int rotaryDim, double theta) => new(Unscaled(rotaryDim, theta), 1.0);

    /// <summary>Llama 3.1 scaling (<c>_compute_llama3_parameters</c>), the same frequencies <see cref="CpuOps.Rope"/> uses for <see cref="RopeKind.Llama3"/>.</summary>
    /// <param name="rotaryDim">Rotated dimensions.</param>
    /// <param name="theta">Base.</param>
    /// <param name="factor"><c>factor</c>.</param>
    /// <param name="lowFreqFactor"><c>low_freq_factor</c>.</param>
    /// <param name="highFreqFactor"><c>high_freq_factor</c>.</param>
    /// <param name="originalMaxPosition"><c>original_max_position_embeddings</c>.</param>
    /// <returns>The frequencies, attention factor 1.</returns>
    public static RopeFrequencies Llama3(int rotaryDim, double theta, double factor, double lowFreqFactor, double highFreqFactor, int originalMaxPosition)
    {
        double[] f = Unscaled(rotaryDim, theta);
        double lowWavelen = originalMaxPosition / lowFreqFactor;
        double highWavelen = originalMaxPosition / highFreqFactor;
        for (int i = 0; i < f.Length; i++)
        {
            double wavelen = 2.0 * Math.PI / f[i];
            double scaled = wavelen > lowWavelen ? f[i] / factor : f[i];
            if (!(wavelen < highWavelen) && !(wavelen > lowWavelen))
            {
                double smooth = ((originalMaxPosition / wavelen) - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
                scaled = ((1.0 - smooth) * scaled / factor) + (smooth * scaled);
            }

            f[i] = scaled;
        }

        return new RopeFrequencies(f, 1.0);
    }

    /// <summary>YaRN (<c>_compute_yarn_parameters</c>): NTK-by-parts interpolation and the mscale attention factor.</summary>
    /// <param name="p">The parameters.</param>
    /// <returns>The frequencies and attention factor.</returns>
    /// <remarks>
    /// <para>factor s (or MaxPosition / OriginalMaxPosition), attention factor <c>mscale(s, Mscale) / mscale(s, MscaleAllDim)</c> when both are
    /// set, else <c>mscale(s, 1)</c>, with <c>mscale(s, m) = s &lt;= 1 ? 1 : 0.1 m ln(s) + 1</c>.</para>
    /// <para>Correction dimension of r rotations: <c>d(r) = R ln(OriginalMax / (2 pi r)) / (2 ln theta)</c>; <c>low = d(BetaFast)</c>,
    /// <c>high = d(BetaSlow)</c>, floored and ceiled when Truncate, then clamped to [0, R - 1] (R, not R / 2, as transformers does).
    /// Ramp <c>r_i = clamp((i - low) / (high - low), 0, 1)</c> (high += 0.001 when equal); <c>f_i = f_i^extrap (1 - r_i) + (f_i^extrap / s) r_i</c>.</para>
    /// </remarks>
    public static RopeFrequencies Yarn(YarnParameters p)
    {
        ArgumentNullException.ThrowIfNull(p);
        int dim = p.RotaryDim;
        double theta = p.Theta;
        double factor = p.Factor ?? ((double)p.MaxPosition / p.OriginalMaxPosition);
        double attention = p.AttentionFactor
            ?? (IsSet(p.Mscale) && IsSet(p.MscaleAllDim) ? YarnMscale(factor, p.Mscale!.Value) / YarnMscale(factor, p.MscaleAllDim!.Value) : YarnMscale(factor, 1.0));
        double betaFast = p.BetaFast == 0 ? 32 : p.BetaFast;
        double betaSlow = p.BetaSlow == 0 ? 1 : p.BetaSlow;
        double low = CorrectionDim(betaFast, dim, theta, p.OriginalMaxPosition);
        double high = CorrectionDim(betaSlow, dim, theta, p.OriginalMaxPosition);
        if (p.Truncate)
        {
            low = Math.Floor(low);
            high = Math.Ceiling(high);
        }

        low = Math.Max(low, 0);
        high = Math.Min(high, dim - 1);
        if (low == high)
        {
            high += 0.001;
        }

        double[] f = Unscaled(dim, theta);
        for (int i = 0; i < f.Length; i++)
        {
            double ramp = Math.Clamp((i - low) / (high - low), 0.0, 1.0);
            double extrapolation = 1.0 - ramp;
            double posFreq = Math.Pow(theta, 2.0 * i / dim);
            f[i] = (1.0 / (factor * posFreq) * (1.0 - extrapolation)) + (1.0 / posFreq * extrapolation);
        }

        return new RopeFrequencies(f, attention);
    }

    /// <summary>Longrope (<c>_compute_longrope_parameters</c>): per-pair rescale factors, short or long by sequence length.</summary>
    /// <param name="p">The parameters.</param>
    /// <param name="sequenceLength">The sequence length the cache serves; above OriginalMaxPosition the long factors apply.</param>
    /// <returns>The frequencies <c>f_i = 1 / (factor_i * theta^(2i / R))</c> and the attention factor.</returns>
    public static RopeFrequencies LongRope(LongRopeParameters p, int sequenceLength)
    {
        ArgumentNullException.ThrowIfNull(p);
        int dim = p.RotaryDim;
        CheckDim(dim);
        IReadOnlyList<double> ext = sequenceLength > p.OriginalMaxPosition ? p.LongFactor : p.ShortFactor;
        if (p.ShortFactor.Count != dim / 2 || p.LongFactor.Count != dim / 2)
        {
            throw new ArgumentException($"ShortFactor and LongFactor need RotaryDim / 2 = {dim / 2} values.", nameof(p));
        }

        double factor = p.Factor ?? ((double)p.MaxPosition / p.OriginalMaxPosition);
        double attention = p.AttentionFactor ?? (factor <= 1.0 ? 1.0 : Math.Sqrt(1 + (Math.Log(factor) / Math.Log(p.OriginalMaxPosition))));
        var f = new double[dim / 2];
        for (int i = 0; i < f.Length; i++)
        {
            f[i] = 1.0 / (ext[i] * Math.Pow(p.Theta, 2.0 * i / dim));
        }

        return new RopeFrequencies(f, attention);
    }

    /// <summary>Dynamic NTK (<c>_compute_dynamic_ntk_parameters</c>): the base grows with the sequence length.</summary>
    /// <param name="rotaryDim">Rotated dimensions; even and greater than 2.</param>
    /// <param name="theta"><c>rope_theta</c>.</param>
    /// <param name="factor"><c>factor</c>.</param>
    /// <param name="maxPosition"><c>max_position_embeddings</c>.</param>
    /// <param name="sequenceLength">The sequence length the cache serves; values below maxPosition count as maxPosition.</param>
    /// <returns>Unscaled frequencies of base <c>theta * (factor * L / maxPosition - (factor - 1))^(R / (R - 2))</c>, L = max(sequenceLength, maxPosition); attention factor 1.</returns>
    public static RopeFrequencies DynamicNtk(int rotaryDim, double theta, double factor, int maxPosition, int sequenceLength)
    {
        CheckDim(rotaryDim);
        if (rotaryDim <= 2 || maxPosition <= 0)
        {
            throw new ArgumentException("Dynamic NTK needs RotaryDim > 2 and a positive maxPosition.", nameof(rotaryDim));
        }

        double len = Math.Max(sequenceLength, maxPosition);
        double b = theta * Math.Pow((factor * len / maxPosition) - (factor - 1), (double)rotaryDim / (rotaryDim - 2));
        return new RopeFrequencies(Unscaled(rotaryDim, b), 1.0);
    }

    /// <summary>Writes the cache row of one position: RotaryDim / 2 cosines, then RotaryDim / 2 sines, each times the attention factor.</summary>
    /// <param name="f">The frequencies.</param>
    /// <param name="position">The position.</param>
    /// <param name="row">Destination, RotaryDim values.</param>
    public static void CosSinRow(RopeFrequencies f, long position, Span<double> row)
    {
        ArgumentNullException.ThrowIfNull(f);
        int half = f.InverseFrequencies.Count;
        if (row.Length < 2 * half)
        {
            throw new ArgumentException($"The row needs {2 * half} values.", nameof(row));
        }

        for (int i = 0; i < half; i++)
        {
            double angle = position * f.InverseFrequencies[i];
            row[i] = Math.Cos(angle) * f.AttentionFactor;
            row[half + i] = Math.Sin(angle) * f.AttentionFactor;
        }
    }

    /// <summary>Fills rows [0, positions) of a host cache in its span's element type, each value rounded once from double.</summary>
    /// <param name="f">The frequencies.</param>
    /// <param name="cache">Host buffer of at least positions * RotaryDim elements, F64, F32, BF16 or F16.</param>
    /// <param name="positions">Rows to write (the MaxPosition of <see cref="RopeScaledArgs.CosSinCache"/>).</param>
    public static void WriteCosSinCache(RopeFrequencies f, DeviceSpan cache, int positions)
    {
        ArgumentNullException.ThrowIfNull(f);
        int rot = 2 * f.InverseFrequencies.Count;
        long n = (long)positions * rot;
        if (positions < 0 || cache.Address == 0 || cache.Length < (ulong)n)
        {
            throw new ArgumentException($"The cache needs {n} elements.", nameof(cache));
        }

        Span<double> row = rot <= 1024 ? stackalloc double[rot] : new double[rot];
        for (int p = 0; p < positions; p++)
        {
            CosSinRow(f, p, row);
            for (int j = 0; j < rot; j++)
            {
                Store(cache.Address, cache.Dtype, ((long)p * rot) + j, row[j]);
            }
        }
    }

    private static bool IsSet(double? v) => v is { } x && x != 0;

    private static double YarnMscale(double scale, double m) => scale <= 1 ? 1.0 : (0.1 * m * Math.Log(scale)) + 1.0;

    private static double CorrectionDim(double rotations, int dim, double theta, int originalMaxPosition) =>
        dim * Math.Log(originalMaxPosition / (rotations * 2 * Math.PI)) / (2 * Math.Log(theta));

    private static void CheckDim(int rotaryDim)
    {
        if (rotaryDim <= 0 || (rotaryDim & 1) != 0)
        {
            throw new ArgumentException("RotaryDim must be even and positive.", nameof(rotaryDim));
        }
    }

    private static double[] Unscaled(int rotaryDim, double theta)
    {
        CheckDim(rotaryDim);
        var f = new double[rotaryDim / 2];
        for (int i = 0; i < f.Length; i++)
        {
            f[i] = 1.0 / Math.Pow(theta, 2.0 * i / rotaryDim);
        }

        return f;
    }
}
