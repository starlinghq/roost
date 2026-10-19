#pragma warning disable CA1000 // the static abstract members of IRoostOps are static by contract

using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Worker.Tests.Support;

/// <summary>Receives a copy of an op's output after the op ran.</summary>
public sealed class Recorder
{
    /// <summary>Reads <c>elements</c> values of a span as floats (a host read for CpuOps, a stream sync and download for CudaOps).</summary>
    public required Func<DeviceSpan, long, float[]> Read { get; init; }

    /// <summary>The recorded outputs in call order.</summary>
    public List<Snapshot> Snapshots { get; } = [];

    /// <summary>Layer counter: advanced by every second Residual.</summary>
    public int Layer { get; set; }

    /// <summary>Residuals seen.</summary>
    public int Residuals { get; set; }

    /// <summary>Records <paramref name="elements"/> values of <paramref name="span"/>.</summary>
    /// <param name="op">The op's name.</param>
    /// <param name="what">Which output.</param>
    /// <param name="span">Where it is.</param>
    /// <param name="elements">How many elements.</param>
    public void Add(string op, string what, DeviceSpan span, long elements) =>
        Snapshots.Add(new Snapshot(Snapshots.Count, Layer, op, what, Read(span, elements)));
}

/// <summary>One recorded op output.</summary>
/// <param name="Ordinal">Position in the call sequence.</param>
/// <param name="Layer">Decoder layer the op belongs to.</param>
/// <param name="Op">The op.</param>
/// <param name="What">Which output of the op.</param>
/// <param name="Values">The values.</param>
public sealed record Snapshot(int Ordinal, int Layer, string Op, string What, float[] Values);

/// <summary>The per-thread switches of <see cref="Probe{TInner}"/>, shared by every instantiation.</summary>
public static class ProbeState
{
    [ThreadStatic]
    private static Recorder? s_recorder;

    [ThreadStatic]
    private static bool s_keepWindowZero;

    /// <summary>The recorder of the calling thread, or null.</summary>
    public static Recorder? Recorder
    {
        get => s_recorder;
        set => s_recorder = value;
    }

    /// <summary>Whether to leave the generated <c>WindowLeft = 0</c> alone (reproduces CpuOps' self-only attention).</summary>
    public static bool KeepWindowZero
    {
        get => s_keepWindowZero;
        set => s_keepWindowZero = value;
    }
}

/// <summary>
/// A pass-through backend for the GPU/CPU comparison. It forwards every op to <typeparamref name="TInner"/> and, when a
/// <see cref="Recorder"/> is set on the thread, records each op's output after it ran. It also works around one defect of
/// the generated forward: the generated PagedAttention arguments leave <c>WindowLeft</c> at 0, which means "a window of
/// zero" to CpuOps and to the GPU kernel (the kernel refuses it) and "no window" to nobody; the model has none, so the
/// wrapper passes -1. See the R-025 report.
/// </summary>
/// <typeparam name="TInner">The backend.</typeparam>
public readonly struct Probe<TInner> : IRoostOps
    where TInner : struct, IRoostOps
{
    /// <inheritdoc/>
    public static bool Supports(OpId op) => TInner.Supports(op);

    /// <inheritdoc/>
    public static void RmsNorm(in RmsNormArgs a)
    {
        TInner.RmsNorm(in a);
        ProbeState.Recorder?.Add("RmsNorm", "out", a.Output, (long)a.Rows * a.Dim);
    }

    /// <inheritdoc/>
    public static void QkNorm(in QkNormArgs a)
    {
        TInner.QkNorm(in a);
        ProbeState.Recorder?.Add("QkNorm", "q", a.Q, (long)a.Tokens * a.QHeads * a.HeadDim);
        ProbeState.Recorder?.Add("QkNorm", "k", a.K, (long)a.Tokens * a.KvHeads * a.HeadDim);
    }

    /// <inheritdoc/>
    public static void Rope(in RopeArgs a)
    {
        TInner.Rope(in a);
        ProbeState.Recorder?.Add("Rope", "q", a.Q, (long)a.Tokens * a.QHeads * a.HeadDim);
        ProbeState.Recorder?.Add("Rope", "k", a.K, (long)a.Tokens * a.KvHeads * a.HeadDim);
    }

    /// <inheritdoc/>
    public static void PagedAttention(in PagedAttentionArgs a)
    {
        var b = a;
        if (!ProbeState.KeepWindowZero && b.WindowLeft == 0)
        {
            b.WindowLeft = -1;
        }

        TInner.PagedAttention(in b);
        ProbeState.Recorder?.Add("PagedAttention", "out", a.Output, (long)a.Tokens * a.QHeads * a.HeadDim);
    }

    /// <inheritdoc/>
    public static void SiluAndMul(in SiluAndMulArgs a)
    {
        TInner.SiluAndMul(in a);
        ProbeState.Recorder?.Add("SiluAndMul", "out", a.Output, (long)a.Rows * a.Dim);
    }

    /// <inheritdoc/>
    public static void GeluAndMul(in GeluAndMulArgs a) => TInner.GeluAndMul(in a);

    /// <inheritdoc/>
    public static void Gemm(in GemmArgs a)
    {
        TInner.Gemm(in a);
        ProbeState.Recorder?.Add("Gemm", "d", a.D, a.M * a.N);
    }

    /// <inheritdoc/>
    public static void Embedding(in EmbeddingArgs a)
    {
        TInner.Embedding(in a);
        ProbeState.Recorder?.Add("Embedding", "out", a.Output, (long)a.Tokens * a.Dim);
    }

    /// <inheritdoc/>
    public static void LmHead(in LmHeadArgs a)
    {
        TInner.LmHead(in a);
        ProbeState.Recorder?.Add("LmHead", "logits", new DeviceSpan { Address = a.Logits.Address, Length = (ulong)a.Selected * (ulong)a.VocabPadded, Dtype = DType.F32 }, (long)a.Selected * a.Vocab);
    }

    /// <inheritdoc/>
    public static void Residual(in ResidualArgs a)
    {
        TInner.Residual(in a);
        if (ProbeState.Recorder is { } r)
        {
            // The pair of residuals per layer: first the token mixer's (hm), then the MLP's (the layer output h{i+1}).
            r.Add("Residual", r.Residuals % 2 == 0 ? "mix" : "layer", a.Output, a.Elements);
            r.Residuals++;
            if (r.Residuals % 2 == 0)
            {
                r.Layer++;
            }
        }
    }

    /// <inheritdoc/>
    public static void CacheWrite(in CacheWriteArgs a) => TInner.CacheWrite(in a);

    /// <inheritdoc/>
    public static void CacheCopy(in CacheCopyArgs a) => TInner.CacheCopy(in a);

    /// <inheritdoc/>
    public static void Argmax(in ArgmaxArgs a) => TInner.Argmax(in a);

    /// <inheritdoc/>
    public static void Sample(in SampleArgs a) => TInner.Sample(in a);

    /// <inheritdoc/>
    public static void GatedDeltaRule(in GatedDeltaRuleArgs a)
    {
        TInner.GatedDeltaRule(in a);
        if (ProbeState.Recorder is { } r)
        {
            r.Add("GatedDeltaRule", "out", a.Output, (long)a.Tokens * a.VHeads * a.VDim);
            // The state of the first slot of the sequence (single-sequence runs use slot 0).
            r.Add("GatedDeltaRule", "state", a.State, (long)a.VHeads * a.KDim * a.VDim);
        }
    }

    /// <inheritdoc/>
    public static void CausalConv1d(in CausalConv1dArgs a)
    {
        TInner.CausalConv1d(in a);
        if (ProbeState.Recorder is { } r)
        {
            r.Add("CausalConv1d", "out", a.Output, (long)a.Tokens * a.Channels);
            r.Add("CausalConv1d", "state", a.ConvState, (long)a.Channels * (a.Width - 1));
        }
    }

    /// <inheritdoc/>
    public static void RecurrentStateCopy(in RecurrentStateCopyArgs a) => TInner.RecurrentStateCopy(in a);

    /// <inheritdoc/>
    public static void RecurrentStateSnapshot(in RecurrentStateSnapshotArgs a) => TInner.RecurrentStateSnapshot(in a);

    /// <inheritdoc/>
    public static void GatedDeltaRuleGates(in GatedDeltaRuleGatesArgs a)
    {
        TInner.GatedDeltaRuleGates(in a);
        ProbeState.Recorder?.Add("GatedDeltaRuleGates", "g", a.G, (long)a.Tokens * a.VHeads);
        ProbeState.Recorder?.Add("GatedDeltaRuleGates", "beta", a.Beta, (long)a.Tokens * a.VHeads);
    }

    /// <inheritdoc/>
    public static void GatedRmsNorm(in GatedRmsNormArgs a)
    {
        TInner.GatedRmsNorm(in a);
        ProbeState.Recorder?.Add("GatedRmsNorm", "out", a.Output, (long)a.Rows * a.Dim);
    }

    /// <inheritdoc/>
    public static void SigmoidMul(in SigmoidMulArgs a)
    {
        TInner.SigmoidMul(in a);
        ProbeState.Recorder?.Add("SigmoidMul", "out", a.Output, (long)a.Rows * a.Dim);
    }
}

/// <summary>Reads host spans as floats (the CpuOps side of a <see cref="Recorder"/>).</summary>
public static unsafe class HostReader
{
    /// <summary>Reads <paramref name="elements"/> values.</summary>
    /// <param name="span">A host span.</param>
    /// <param name="elements">How many.</param>
    /// <returns>The values.</returns>
    public static float[] Read(DeviceSpan span, long elements) => Convert((void*)span.Address, span.Dtype, elements);

    /// <summary>Converts a host buffer of any float dtype to floats.</summary>
    /// <param name="p">The buffer.</param>
    /// <param name="dtype">Its element type.</param>
    /// <param name="elements">How many.</param>
    /// <returns>The values.</returns>
    public static float[] Convert(void* p, DType dtype, long elements)
    {
        var o = new float[elements];
        switch (dtype)
        {
            case DType.F32:
                new ReadOnlySpan<float>(p, (int)elements).CopyTo(o);
                break;
            case DType.BF16:
                for (long i = 0; i < elements; i++)
                {
                    o[i] = BitConverter.Int32BitsToSingle(((ushort*)p)[i] << 16);
                }

                break;
            case DType.F16:
                for (long i = 0; i < elements; i++)
                {
                    o[i] = (float)BitConverter.UInt16BitsToHalf(((ushort*)p)[i]);
                }

                break;
            default:
                throw new NotSupportedException($"cannot read {dtype} as float");
        }

        return o;
    }
}
