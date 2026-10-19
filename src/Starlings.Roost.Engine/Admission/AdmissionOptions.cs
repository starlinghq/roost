namespace Starlings.Roost.Engine.Admission;

/// <summary>Limits of admission control (R-031).</summary>
public sealed record AdmissionOptions
{
    /// <summary>
    /// The model's context length in tokens. A prompt of this length or more is refused; a request without
    /// <c>max_tokens</c> is reserved up to it, because it may run until a stop token.
    /// </summary>
    public required int ContextLength { get; init; }

    /// <summary>
    /// Bytes of step workspace the worker allocates outside the block pools: kernel workspaces for the largest plan,
    /// activations, and graph memory (R-033). <see cref="AdmissionController.TryComputePoolBudget"/> subtracts it from
    /// the free memory, so the pools never take it.
    /// </summary>
    public long WorkspaceBytes { get; init; }

    /// <summary>Bytes to leave to the driver and to co-tenants' growth, subtracted like the workspace.</summary>
    public long HeadroomBytes { get; init; }

    /// <summary>
    /// Free device bytes below which a request is refused with <see cref="Requests.RoostErrorCode.InsufficientDeviceMemory"/>
    /// even when the pools have room: memory the worker may still allocate lazily (graph capture, library handles) or
    /// that a co-tenant has taken since the pools were sized. Zero disables the check. Needs an <see cref="IDeviceMemorySource"/>.
    /// </summary>
    public long MinFreeDeviceBytes { get; init; }

    /// <summary>Live rows (sequences, counting forked branch rows) allowed at once; zero means only the pools limit it.</summary>
    public int MaxLiveSequences { get; init; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ContextLength);
        ArgumentOutOfRangeException.ThrowIfNegative(WorkspaceBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(HeadroomBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(MinFreeDeviceBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(MaxLiveSequences);
    }
}
