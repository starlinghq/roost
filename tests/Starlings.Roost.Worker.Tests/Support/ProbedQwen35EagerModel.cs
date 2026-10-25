using Starlings.Roost.Kernels;
using Starlings.Roost.Ops;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Qwen35;

namespace Starlings.Roost.Worker.Tests.Support;

/// <summary>
/// The product <see cref="Qwen35EagerModel"/> with the per-layer probe: when <see cref="ProbeState.Recorder"/> is set (the
/// per-layer comparison), the forward goes through <see cref="Probe{TInner}"/>; otherwise it is the product forward.
/// </summary>
public sealed class ProbedQwen35EagerModel : IEagerModel
{
    private readonly Qwen35EagerModel _inner = new();

    /// <inheritdoc/>
    public string Name => _inner.Name;

    /// <inheritdoc/>
    public int VocabSize => _inner.VocabSize;

    /// <inheritdoc/>
    public int BlockSize => _inner.BlockSize;

    /// <inheritdoc/>
    public int AttentionGroup => _inner.AttentionGroup;

    /// <inheritdoc/>
    public int GroupCount => _inner.GroupCount;

    /// <inheritdoc/>
    public IReadOnlyList<EagerWeightSlot> WeightSlots => _inner.WeightSlots;

    /// <inheritdoc/>
    public IReadOnlyList<EagerStateSlot> StateSlots => _inner.StateSlots;

    /// <inheritdoc/>
    public EagerLayout Layout(int maxTokens, int maxSequences, int maxKvBlocks) => _inner.Layout(maxTokens, maxSequences, maxKvBlocks);

    /// <inheritdoc/>
    public IReadOnlyList<KernelRequirement> KernelRequirements(int sm) => _inner.KernelRequirements(sm);

    /// <inheritdoc/>
    public void CheckBackend<TOps>()
        where TOps : struct, IRoostOps => _inner.CheckBackend<TOps>();

    /// <inheritdoc/>
    public void Forward<TOps>(in EagerForward input)
        where TOps : struct, IRoostOps
    {
        if (ProbeState.Recorder is null)
        {
            _inner.Forward<TOps>(in input);
            return;
        }

        Qwen35EagerModel.RunForward<Probe<TOps>>(in input);
    }

    /// <inheritdoc/>
    public DeviceSpan Logits(in EagerForward input) => _inner.Logits(in input);
}
