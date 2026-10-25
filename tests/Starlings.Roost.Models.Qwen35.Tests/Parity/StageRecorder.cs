using Starlings.Roost.Abi;
using System.Globalization;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Qwen35.Tests.Parity;

/// <summary>
/// Collects one case's per-layer capture (R-011 layout: <c>forward{n}/embeddings</c>, <c>forward{n}/layer.{i}</c>,
/// <c>forward{n}/final_norm</c>, and for linear-attention layers <c>forward{n}/layer.{i}/recurrent_state</c> and
/// <c>/conv_state</c>) from the calls <see cref="RecordingOps"/> forwards to it. One sequence (batch 1, state slot 0).
/// </summary>
internal sealed unsafe class StageRecorder
{
    private const int Layers = 24;
    private const int NormsBeforeFinal = 2 * Layers;

    private readonly int[] _linearLayers;
    private readonly int _stateForwards;
    private int _forward;
    private int _residuals;
    private int _norms;
    private int _convs;
    private int _deltas;
    private int _embeddings;
    private bool _finalSeen;

    public StageRecorder(IReadOnlyList<string> layerTypes, int stateForwards, Perturbation perturbation)
    {
        ArgumentNullException.ThrowIfNull(layerTypes);
        if (layerTypes.Count != Layers)
        {
            throw new ArgumentException($"{layerTypes.Count} layer types; Qwen3.5-0.8B has {Layers}");
        }

        _linearLayers = [.. Enumerable.Range(0, Layers).Where(i => layerTypes[i] == "linear_attention")];
        _stateForwards = stateForwards;
        Perturbation = perturbation;
    }

    /// <summary>The recorder the next forward reports to; null records nothing.</summary>
    [ThreadStatic]
    public static StageRecorder? Active;

    public Perturbation Perturbation { get; }

    /// <summary>Captured tensors: name to (shape, float32 values).</summary>
    public Dictionary<string, (int[] Shape, float[] Values)> Tensors { get; } = new(StringComparer.Ordinal);

    /// <summary>Checks that one whole forward was seen and moves on to the next.</summary>
    public void EndForward()
    {
        bool states = _forward < _stateForwards;
        if (_embeddings != 1 || _residuals != 2 * Layers || _norms != NormsBeforeFinal + 1 || !_finalSeen
            || _convs != _linearLayers.Length || _deltas != _linearLayers.Length)
        {
            throw new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"forward {_forward}: saw {_embeddings} embeddings, {_residuals} residuals, {_norms} norms, {_convs} convolutions, {_deltas} delta rules (states {states})"));
        }

        _forward++;
        _embeddings = _residuals = _norms = _convs = _deltas = 0;
        _finalSeen = false;
    }

    public void OnEmbedding(in EmbeddingArgs a)
    {
        _embeddings++;
        Hidden("embeddings", a.Output, a.Tokens, a.Dim, a.DType);
    }

    public void OnResidual(in ResidualArgs a)
    {
        int index = _residuals++;
        if (index % 2 == 1)
        {
            int dim = 1024;
            Hidden("layer." + (index / 2).ToString(CultureInfo.InvariantCulture), a.Output, checked((int)(a.Elements / dim)), dim, a.DType);
        }
    }

    public void OnRmsNorm(in RmsNormArgs a)
    {
        int index = _norms++;
        if (index == NormsBeforeFinal)
        {
            _finalSeen = true;
            Hidden("final_norm", a.Output, a.Rows, a.Dim, a.DType);
        }
    }

    public void BeforeCausalConv1d(in CausalConv1dArgs a)
    {
        int layer = _linearLayers[_convs++];
        if (_forward >= _stateForwards)
        {
            return;
        }

        if (a.Batch != 1)
        {
            throw new InvalidOperationException("the recorder handles one sequence");
        }

        long slot = ReadIndex(a.StateSlots, 0);
        int w = a.Width;
        int kept = w - 1;
        DType st = a.ConvState.Dtype == DType.Undefined ? a.DType : a.ConvState.Dtype;
        var values = new float[(long)a.Channels * w];

        // The reference's state is the last Width inputs of [kept history, this step's inputs], oldest first (transformers keeps
        // conv_kernel_size columns; the slot keeps Width - 1). Column 0 is therefore reconstructed here, not stored by Roost.
        for (int c = 0; c < a.Channels; c++)
        {
            long total = kept + a.Tokens;
            for (int j = 0; j < w; j++)
            {
                long i = total - w + j;
                double v = i < 0 ? 0.0
                    : i < kept ? Load(a.ConvState.Address, st, (((slot * a.Channels) + c) * kept) + i)
                    : Load(a.Input.Address, a.DType, ((i - kept) * a.Channels) + c);
                values[((long)c * w) + j] = (float)v;
            }
        }

        Tensors[StateName(layer, "conv_state")] = ([a.Channels, w], values);
    }

    public void OnGatedDeltaRule(in GatedDeltaRuleArgs a)
    {
        int layer = _linearLayers[_deltas++];
        if (_forward >= _stateForwards)
        {
            return;
        }

        long slot = ReadIndex(a.StateSlots, 0);
        long count = (long)a.VHeads * a.KDim * a.VDim;
        float* src = (float*)a.State.Address + (slot * count);
        Tensors[StateName(layer, "recurrent_state")] = ([a.VHeads, a.KDim, a.VDim], new ReadOnlySpan<float>(src, checked((int)count)).ToArray());
    }

    private string StateName(int layer, string kind) =>
        string.Create(CultureInfo.InvariantCulture, $"forward{_forward}/layer.{layer}/{kind}");

    private void Hidden(string stage, DeviceSpan span, int rows, int dim, DType dtype)
    {
        var values = new float[(long)rows * dim];
        for (long i = 0; i < values.Length; i++)
        {
            values[i] = (float)Load(span.Address, dtype, i);
        }

        Tensors[string.Create(CultureInfo.InvariantCulture, $"forward{_forward}/{stage}")] = ([rows, dim], values);
    }

    private static long ReadIndex(DeviceSpan span, long i) => span.Dtype == DType.I64 ? ((long*)span.Address)[i] : ((int*)span.Address)[i];

    internal static double Load(nint address, DType dtype, long i) => dtype switch
    {
        DType.BF16 => BitConverter.Int32BitsToSingle(((ushort*)address)[i] << 16),
        DType.F32 => ((float*)address)[i],
        DType.F64 => ((double*)address)[i],
        _ => throw new NotSupportedException($"the recorder does not read {dtype}"),
    };
}
