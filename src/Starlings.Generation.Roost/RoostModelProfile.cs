using System.Collections.Immutable;
using Starlings.Roost.Generation;
using Starlings.Roost.State;
using Starlings.Tokenizers;

namespace Starlings.Generation.Roost;

/// <summary>
/// What the host needs to know about the model a worker serves: how to turn a conversation into prompt ids and ids back
/// into text, which ids stop generation, the state groups and pool capacities the worker was sized for, and the digests
/// the worker checks in the handshake. The capacities must equal the worker's (for the eager worker,
/// <c>EagerWorkerOptions.AttentionBlocks</c>, <c>RecurrentSlots</c>, <c>MaxTokens</c>, <c>MaxSequences</c> and
/// <c>MaxKvBlocks</c>): the worker refuses a plan that exceeds them.
/// </summary>
public sealed record RoostModelProfile
{
    /// <summary>The model id reported in responses and accepted in <c>ChatOptions.ModelId</c>.</summary>
    public required string ModelId { get; init; }

    /// <summary>The model's tokenizer.</summary>
    public required Tokenizer Tokenizer { get; init; }

    /// <summary>The model's chat template (for Qwen3.5-0.8B, <see cref="Qwen35RoostChatTemplate.Instance"/>).</summary>
    public required IRoostChatTemplate ChatTemplate { get; init; }

    /// <summary>The model's generation stop ids (for Qwen chat models, <c>&lt;|im_end|&gt;</c> and the end-of-text id).</summary>
    public required ImmutableArray<int> StopTokenIds { get; init; }

    /// <summary>The model's state groups (C4) in index order, as its generated <c>CreateStateGroups()</c> returns them.</summary>
    public required ImmutableArray<StateGroupDescriptor> StateGroups { get; init; }

    /// <summary>Blocks in each paged (non-recurrent) group's pool on the worker.</summary>
    public required int AttentionBlocks { get; init; }

    /// <summary>Slots in each recurrent group's pool on the worker.</summary>
    public int RecurrentSlots { get; init; } = 1;

    /// <summary>The most tokens one step may carry; a longer prompt is prefilled in chunks of at most this many.</summary>
    public int MaxTokensPerStep { get; init; } = 128;

    /// <summary>The most sequences one step may carry. The eager worker runs one (M0).</summary>
    public int MaxSequencesPerStep { get; init; } = 1;

    /// <summary>The most attention block ids one step may name; null means <see cref="AttentionBlocks"/>.</summary>
    public int? MaxKvBlocksPerStep { get; init; }

    /// <summary>SHA-256 of the kernel manifest (C5) the worker must have been built from, 32 bytes.</summary>
    public required ImmutableArray<byte> ManifestDigest { get; init; }

    /// <summary>SHA-256 of the weights as recorded in the profile lock, 32 bytes.</summary>
    public required ImmutableArray<byte> WeightsDigest { get; init; }

    /// <summary>Tokens per block of the first paged group.</summary>
    public int BlockSize => PagedGroup.BlockSize;

    /// <summary>
    /// The longest sequence (prompt plus output) the worker can hold: the paged pool, or the per-step block limit when that
    /// is smaller, in tokens.
    /// </summary>
    public int ContextTokens => Math.Min(AttentionBlocks, MaxKvBlocksPerStep ?? AttentionBlocks) * BlockSize;

    private StateGroupDescriptor PagedGroup =>
        StateGroups.IsDefaultOrEmpty
            ? throw new InvalidOperationException("The profile has no state groups.")
            : StateGroups.FirstOrDefault(static g => g.Kind != StateGroupKind.Recurrent)
                ?? throw new InvalidOperationException("The profile has no paged state group.");

    /// <summary>Checks the profile for values no worker can serve.</summary>
    /// <exception cref="ArgumentException">A value is missing or out of range.</exception>
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentNullException.ThrowIfNull(Tokenizer);
        ArgumentNullException.ThrowIfNull(ChatTemplate);
        if (StopTokenIds.IsDefault)
        {
            throw new ArgumentException("StopTokenIds must be set (empty for none).", nameof(StopTokenIds));
        }

        if (StateGroups.IsDefaultOrEmpty || !StateGroups.Any(static g => g.Kind != StateGroupKind.Recurrent))
        {
            throw new ArgumentException("The profile needs at least one paged state group.", nameof(StateGroups));
        }

        if (AttentionBlocks <= 0 || RecurrentSlots < 0 || MaxTokensPerStep <= 0 || MaxSequencesPerStep <= 0 || MaxKvBlocksPerStep is <= 0)
        {
            throw new ArgumentException("Pool capacities and step limits must be positive.");
        }

        if (RecurrentSlots == 0 && StateGroups.Any(static g => g.Kind == StateGroupKind.Recurrent))
        {
            throw new ArgumentException("The model has a recurrent state group, so the worker needs at least one recurrent slot.", nameof(RecurrentSlots));
        }

        if (ManifestDigest.IsDefault || ManifestDigest.Length != 32 || WeightsDigest.IsDefault || WeightsDigest.Length != 32)
        {
            throw new ArgumentException("ManifestDigest and WeightsDigest are SHA-256 digests: 32 bytes each.");
        }
    }
}
