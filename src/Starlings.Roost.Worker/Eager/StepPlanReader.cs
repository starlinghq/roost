using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Worker.Eager;

/// <summary>Why a step plan was refused. The worker reports it as a recoverable fault: nothing ran.</summary>
public enum PlanRejection
{
    /// <summary>Accepted.</summary>
    None = 0,

    /// <summary>The buffer is shorter than its header, a table lies outside it or a record is shorter than its stride.</summary>
    Malformed = 1000,

    /// <summary>The step is larger than the worker was sized for (tokens, sequences, blocks).</summary>
    TooLarge,

    /// <summary>A token id, block id, position or slot is outside what the worker holds.</summary>
    OutOfRange,

    /// <summary>The plan asks for something the eager worker does not do (sampling other than greedy, cache ops, speculation, heads).</summary>
    Unsupported,
}

/// <summary>A step plan was refused; nothing was launched.</summary>
public sealed class PlanRejectedException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="reason">The class of refusal.</param>
    /// <param name="message">What was wrong; never contains token text.</param>
    public PlanRejectedException(PlanRejection reason, string message)
        : base(message)
    {
        Reason = reason;
    }

    /// <summary>The class of refusal.</summary>
    public PlanRejection Reason { get; }
}

/// <summary>What the worker holds, against which a plan is checked before any index reaches the device.</summary>
/// <param name="MaxTokens">Largest token count of a step.</param>
/// <param name="MaxSequences">Largest sequence count of a step.</param>
/// <param name="MaxKvBlocks">Largest total of attention block ids across a step's sequences.</param>
/// <param name="MaxSelected">Largest count of sequences that sample in a step.</param>
/// <param name="Vocab">Token ids are in [0, Vocab).</param>
/// <param name="BlockSize">Tokens per attention block.</param>
/// <param name="AttentionBlocks">Attention pool capacity in blocks: block ids are in [0, AttentionBlocks).</param>
/// <param name="RecurrentSlots">Recurrent pool capacity: slots are in [0, RecurrentSlots).</param>
/// <param name="AttentionGroup">Index of the attention state group in the plan's group_blocks table.</param>
/// <param name="GroupCount">State groups per row in the plan.</param>
public readonly record struct PlanLimits(
    int MaxTokens,
    int MaxSequences,
    int MaxKvBlocks,
    int MaxSelected,
    int Vocab,
    int BlockSize,
    int AttentionBlocks,
    int RecurrentSlots,
    int AttentionGroup,
    int GroupCount);

/// <summary>One sequence of a validated plan.</summary>
/// <param name="Handle">The engine's id for the sequence, echoed in the result.</param>
/// <param name="TokenStart">First token's index in the flat token table (sequences are contiguous and in order).</param>
/// <param name="TokenCount">Tokens in this chunk.</param>
/// <param name="PositionOffset">Position of the first token.</param>
/// <param name="RecurrentSlot">Recurrent pool slot, or -1.</param>
/// <param name="Sample">Whether to emit a token after this chunk.</param>
/// <param name="Blocks">The first <c>ceil((PositionOffset + TokenCount) / BlockSize)</c> attention blocks, in order.</param>
/// <param name="StopStart">Index into the stop table.</param>
/// <param name="StopCount">Stop ids.</param>
/// <param name="MaxNewTokensLeft">Tokens the request may still emit; 0 for no limit.</param>
public readonly record struct ParsedSequence(
    ulong Handle,
    int TokenStart,
    int TokenCount,
    int PositionOffset,
    int RecurrentSlot,
    bool Sample,
    int[] Blocks,
    int StopStart,
    int StopCount,
    uint MaxNewTokensLeft)
{
    /// <summary>Keys in the cache after this chunk.</summary>
    public int KvLength => PositionOffset + TokenCount;
}

/// <summary>A step plan (C2) checked against <see cref="PlanLimits"/> and copied out of the shared buffer.</summary>
public sealed class ParsedPlan
{
    internal ParsedPlan(ulong stepId, int[] tokenIds, int[] positions, ParsedSequence[] sequences, int[] stopIds)
    {
        StepId = stepId;
        TokenIds = tokenIds;
        Positions = positions;
        Sequences = sequences;
        StopIds = stopIds;
    }

    /// <summary>The step id.</summary>
    public ulong StepId { get; }

    /// <summary>Flat token ids.</summary>
    public int[] TokenIds { get; }

    /// <summary>Flat positions.</summary>
    public int[] Positions { get; }

    /// <summary>The sequences, in plan order.</summary>
    public ParsedSequence[] Sequences { get; }

    /// <summary>The stop-id table.</summary>
    public int[] StopIds { get; }

    /// <summary>Sequences that sample.</summary>
    public int Selected => Sequences.Count(static s => s.Sample);

    /// <summary>Total attention block ids over all sequences.</summary>
    public int KvBlocks => Sequences.Sum(static s => s.Blocks.Length);
}

/// <summary>
/// Reads a C2 step plan for the eager worker (R-025). The plan arrives in shared memory written by another process, so
/// every offset, count, index and id is checked here, on the host, before anything reaches the device: a bad block id
/// or token id would otherwise be an out-of-range device access, which is a sticky fault.
/// </summary>
/// <remarks>
/// The eager worker runs what a single-GPU M0 step needs: greedy sampling, one chunk per sequence, contiguous tokens in
/// sequence order, positions consecutive from <c>position_offset</c>, no cache ops, no speculation, no head rows, no
/// LoRA, grammar or embedding inputs. Anything else is refused with <see cref="PlanRejection.Unsupported"/>.
/// </remarks>
public static class StepPlanReader
{
    /// <summary>Parses and checks a plan.</summary>
    /// <param name="plan">The plan buffer (header first).</param>
    /// <param name="limits">What the worker holds.</param>
    /// <returns>The checked plan.</returns>
    /// <exception cref="PlanRejectedException">The plan is malformed, too large, out of range or unsupported.</exception>
    public static ParsedPlan Parse(ReadOnlySpan<byte> plan, in PlanLimits limits)
    {
        if (plan.Length < Unsafe.SizeOf<StepPlanHeader>())
        {
            throw Reject(PlanRejection.Malformed, "the plan is shorter than its header");
        }

        var h = MemoryMarshal.Read<StepPlanHeader>(plan);
        if (h.Magic != SharedLayout.MagicPlan || h.LayoutVersion != SharedLayout.Version || h.TotalBytes > (ulong)plan.Length)
        {
            throw Reject(PlanRejection.Malformed, "bad magic, layout version or size");
        }

        plan = plan[..(int)h.TotalBytes];
        if (h.SequenceCount == 0 || h.SequenceCount > limits.MaxSequences)
        {
            throw Reject(h.SequenceCount == 0 ? PlanRejection.Malformed : PlanRejection.TooLarge, $"sequence count {h.SequenceCount} (the worker holds {limits.MaxSequences})");
        }

        if (h.TokenCount == 0 || h.TokenCount > limits.MaxTokens)
        {
            throw Reject(h.TokenCount == 0 ? PlanRejection.Malformed : PlanRejection.TooLarge, $"token count {h.TokenCount} (the worker holds {limits.MaxTokens})");
        }

        if (h.GroupCount != limits.GroupCount)
        {
            throw Reject(PlanRejection.Malformed, $"group count {h.GroupCount}, expected {limits.GroupCount}");
        }

        if (h.CacheOps.Count != 0 || h.SpecTreeParents.Count != 0 || h.KvTransfers.Count != 0 || h.HeadRows.Count != 0 || h.EncoderOutputs.Count != 0 || h.EmbeddingInputs.Count != 0)
        {
            throw Reject(PlanRejection.Unsupported, "cache ops, speculation, KV transfers, head rows, encoder outputs and embedding inputs are not run by the eager worker");
        }

        var seqs = Table<PlanSequence>(plan, h.Sequences, h.SequenceCount, "sequences");
        var tokens = Table<int>(plan, h.TokenIds, h.TokenCount, "token_ids");
        var positions = Table<int>(plan, h.Positions, h.TokenCount, "positions");
        var groupBlocks = Table<PlanGroupBlocks>(plan, h.GroupBlocks, h.SequenceCount * h.GroupCount, "group_blocks");
        var blockIds = Table<int>(plan, h.BlockIds, h.BlockIds.Count, "block_ids");
        var sampling = Table<PlanSampling>(plan, h.Sampling, h.Sampling.Count, "sampling");
        var stops = Table<int>(plan, h.StopTokenIds, h.StopTokenIds.Count, "stop_token_ids");

        foreach (int t in tokens)
        {
            if ((uint)t >= (uint)limits.Vocab)
            {
                throw Reject(PlanRejection.OutOfRange, $"a token id is outside [0, {limits.Vocab})");
            }
        }

        var parsed = new ParsedSequence[(int)h.SequenceCount];
        int expectedStart = 0;
        int selected = 0;
        long totalBlocks = 0;
        for (int i = 0; i < parsed.Length; i++)
        {
            var s = seqs[i];
            if (s.TokenStart != expectedStart || s.TokenCount == 0 || (long)s.TokenStart + s.TokenCount > h.TokenCount)
            {
                throw Reject(PlanRejection.Malformed, $"sequence {i}: tokens must be non-empty, contiguous and in sequence order");
            }

            expectedStart += (int)s.TokenCount;
            if ((s.Flags & (SequenceOptions.Verify | SequenceOptions.HeadRow)) != 0 || s.DraftCount != 0 || s.LoraSlot >= 0 || s.GrammarSlot >= 0 || s.EmbeddingSlot >= 0)
            {
                throw Reject(PlanRejection.Unsupported, $"sequence {i}: verification, head rows, drafts, LoRA, grammar and embedding inputs are not run by the eager worker");
            }

            if (s.PositionOffset > int.MaxValue / 2 || s.PositionOffset + s.TokenCount > (ulong)limits.AttentionBlocks * (ulong)limits.BlockSize)
            {
                throw Reject(PlanRejection.OutOfRange, $"sequence {i}: positions reach past the attention pool ({limits.AttentionBlocks} blocks of {limits.BlockSize})");
            }

            for (int t = 0; t < s.TokenCount; t++)
            {
                if (positions[(int)s.TokenStart + t] != (int)s.PositionOffset + t)
                {
                    throw Reject(PlanRejection.Malformed, $"sequence {i}: positions must run consecutively from position_offset");
                }
            }

            bool sample = (s.Flags & SequenceOptions.Sample) != 0;
            if (sample)
            {
                selected++;
                if (s.SamplingIndex >= sampling.Length)
                {
                    throw Reject(PlanRejection.Malformed, $"sequence {i}: sampling index outside the sampling table");
                }

                var sp = sampling[(int)s.SamplingIndex];
                bool greedy = (sp.Flags & SamplingOptions.Greedy) != 0 || sp.Temperature == 0f;
                if (!greedy || sp.Logprobs != 0 || sp.RepetitionPenalty is not (0f or 1f) || sp.PresencePenalty != 0f || sp.FrequencyPenalty != 0f)
                {
                    throw Reject(PlanRejection.Unsupported, $"sequence {i}: only greedy sampling without penalties or logprobs is run by the eager worker");
                }
            }

            int slot = s.RecurrentSlot;
            if ((uint)slot >= (uint)limits.RecurrentSlots && limits.RecurrentSlots > 0)
            {
                throw Reject(PlanRejection.OutOfRange, $"sequence {i}: recurrent slot {slot} is outside [0, {limits.RecurrentSlots})");
            }

            int pages = ((int)s.PositionOffset + (int)s.TokenCount + limits.BlockSize - 1) / limits.BlockSize;
            int gb = (int)s.GroupBlocksStart + limits.AttentionGroup;
            if (s.GroupBlocksStart != (uint)(i * h.GroupCount) || gb >= groupBlocks.Length)
            {
                throw Reject(PlanRejection.Malformed, $"sequence {i}: group_blocks_start must be sequence_index * group_count");
            }

            var g = groupBlocks[gb];
            if (g.BlockCount < pages || (long)g.BlockStart + g.BlockCount > blockIds.Length)
            {
                throw Reject(PlanRejection.OutOfRange, $"sequence {i}: {g.BlockCount} attention blocks listed, {pages} needed");
            }

            var blocks = blockIds.Slice((int)g.BlockStart, pages).ToArray();
            foreach (int b in blocks)
            {
                if ((uint)b >= (uint)limits.AttentionBlocks)
                {
                    throw Reject(PlanRejection.OutOfRange, $"sequence {i}: block id outside [0, {limits.AttentionBlocks})");
                }
            }

            if ((long)s.StopStart + s.StopCount > stops.Length)
            {
                throw Reject(PlanRejection.Malformed, $"sequence {i}: stop ids outside the stop table");
            }

            totalBlocks += pages;
            parsed[i] = new ParsedSequence(s.SequenceHandle, (int)s.TokenStart, (int)s.TokenCount, (int)s.PositionOffset, slot, sample, blocks, (int)s.StopStart, (int)s.StopCount, s.MaxNewTokensLeft);
        }

        if (expectedStart != h.TokenCount)
        {
            throw Reject(PlanRejection.Malformed, "the sequences do not cover the token table");
        }

        if (totalBlocks > limits.MaxKvBlocks)
        {
            throw Reject(PlanRejection.TooLarge, $"{totalBlocks} attention blocks in the step (the worker holds {limits.MaxKvBlocks})");
        }

        if (selected > limits.MaxSelected)
        {
            throw Reject(PlanRejection.TooLarge, $"{selected} sampling sequences (the worker holds {limits.MaxSelected})");
        }

        return new ParsedPlan(h.StepId, tokens.ToArray(), positions.ToArray(), parsed, stops.ToArray());
    }

    private static ReadOnlySpan<T> Table<T>(ReadOnlySpan<byte> buffer, in TableRef table, uint expected, string name)
        where T : unmanaged
    {
        if (table.Count != expected)
        {
            throw Reject(PlanRejection.Malformed, $"table {name} has {table.Count} entries, expected {expected}");
        }

        if (table.Count == 0)
        {
            return default;
        }

        int size = Unsafe.SizeOf<T>();
        if (table.Stride < size || table.Offset % 4 != 0 || table.Offset > (ulong)buffer.Length)
        {
            throw Reject(PlanRejection.Malformed, $"table {name} has a bad stride or offset");
        }

        // Records are packed at their stride; the reader supports stride == size, which is what the writers produce.
        if (table.Stride != size)
        {
            throw Reject(PlanRejection.Malformed, $"table {name} has stride {table.Stride}, expected {size}");
        }

        ulong bytes = (ulong)table.Count * (ulong)size;
        if (table.Offset + bytes > (ulong)buffer.Length)
        {
            throw Reject(PlanRejection.Malformed, $"table {name} extends past the end of the plan");
        }

        // The span is over the shared buffer: copy-free but unaligned reads are fine for these unmanaged types via MemoryMarshal.Cast
        // only when aligned, so copy into an aligned array.
        var copy = new T[table.Count];
        buffer.Slice((int)table.Offset, (int)bytes).CopyTo(MemoryMarshal.AsBytes(copy.AsSpan()));
        return copy;
    }

    private static PlanRejectedException Reject(PlanRejection reason, string message) => new(reason, message);
}
