using System.Runtime.InteropServices;

namespace Starlings.Roost.Native;

/// <summary>
/// The token-mask slots that contract C2's <c>grammar_slot</c> names (<see cref="Plans.PlanSequence.GrammarSlot"/>, -1 for none):
/// each slot holds one sequence's <see cref="GrammarMatcher"/> and a row of a contiguous, 64-byte-aligned native mask table, so a
/// sampler can read the mask of slot <c>s</c> at <c>MaskWords * s</c>. Owned by the worker side; the host never sees an address.
/// </summary>
/// <remarks>
/// <para><b>Order per step.</b> <see cref="ComputeMasks"/> for the slots of the step (on thread-pool threads when there are
/// several, so the work can overlap the forward pass, X10), then sample with the masks
/// (<see cref="GrammarSampler.Sample"/>), then <see cref="Advance"/> each slot by its sampled token. A mask is stale after
/// <see cref="Advance"/> and is recomputed on the next <see cref="ComputeMasks"/>.</para>
/// <para><b>Threading.</b> Assign, release, advance and sampling run on one thread (the worker's step thread);
/// <see cref="ComputeMasks"/> fans out internally and returns when every mask is written.</para>
/// </remarks>
public sealed unsafe class GrammarSlots : IDisposable
{
    private readonly GrammarMatcher?[] _matchers;
    private readonly bool[] _fresh;
    private uint* _masks;

    /// <summary>Creates an empty table.</summary>
    /// <param name="capacity">Slots: the most sequences with a grammar at once.</param>
    /// <param name="vocabSize">Mask rows: the model's logits width (the tokenizer's <see cref="GrammarTokenizer.VocabSize"/>).</param>
    public GrammarSlots(int capacity, int vocabSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        Capacity = capacity;
        VocabSize = vocabSize;
        MaskWords = (vocabSize + 31) / 32;
        _matchers = new GrammarMatcher?[capacity];
        _fresh = new bool[capacity];
        nuint bytes = (nuint)capacity * (nuint)MaskWords * sizeof(uint);
        _masks = (uint*)NativeMemory.AlignedAlloc(bytes, 64);
        NativeMemory.Clear(_masks, bytes);
    }

    /// <summary>Frees the mask table and the matchers if <see cref="Dispose"/> was not called.</summary>
    ~GrammarSlots()
    {
        FreeMasks();
    }

    /// <summary>Slots in the table.</summary>
    public int Capacity { get; }

    /// <summary>Mask rows.</summary>
    public int VocabSize { get; }

    /// <summary>Words per mask row, <c>ceil(VocabSize / 32)</c>.</summary>
    public int MaskWords { get; }

    /// <summary>Slots in use.</summary>
    public int Count { get; private set; }

    /// <summary>Puts a matcher in the lowest free slot; the table owns it from then on.</summary>
    /// <param name="matcher">The matcher, made by a tokenizer whose <see cref="GrammarTokenizer.VocabSize"/> is <see cref="VocabSize"/>.</param>
    /// <returns>The slot, the value for <c>grammar_slot</c>.</returns>
    /// <exception cref="ArgumentException">The matcher's vocabulary differs from the table's.</exception>
    /// <exception cref="InvalidOperationException">Every slot is in use.</exception>
    public int Assign(GrammarMatcher matcher)
    {
        ArgumentNullException.ThrowIfNull(matcher);
        ObjectDisposedException.ThrowIf(_masks == null, this);
        if (matcher.VocabSize != VocabSize)
        {
            throw new ArgumentException($"The matcher has {matcher.VocabSize} mask rows; the table has {VocabSize}.", nameof(matcher));
        }

        for (var slot = 0; slot < _matchers.Length; slot++)
        {
            if (_matchers[slot] is null)
            {
                _matchers[slot] = matcher;
                _fresh[slot] = false;
                Count++;
                return slot;
            }
        }

        throw new InvalidOperationException($"All {Capacity} grammar slots are in use.");
    }

    /// <summary>Disposes a slot's matcher and frees the slot.</summary>
    /// <param name="slot">The slot.</param>
    public void Release(int slot)
    {
        var matcher = Get(slot);
        _matchers[slot] = null;
        _fresh[slot] = false;
        Count--;
        matcher.Dispose();
    }

    /// <summary>The matcher in a slot.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>The matcher.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The slot is outside the table.</exception>
    /// <exception cref="InvalidOperationException">The slot is free.</exception>
    public GrammarMatcher Get(int slot)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(slot);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(slot, Capacity);
        return _matchers[slot] ?? throw new InvalidOperationException($"Grammar slot {slot} is free.");
    }

    /// <summary>Computes the masks of the given slots that are stale (negative entries, meaning no grammar, are skipped).</summary>
    /// <param name="slots">The slots of a step, in any order; repeats are allowed.</param>
    /// <exception cref="GrammarException">A matcher is in its error state.</exception>
    public void ComputeMasks(ReadOnlySpan<int> slots)
    {
        ObjectDisposedException.ThrowIf(_masks == null, this);
        var stale = new List<int>(slots.Length);
        foreach (var slot in slots)
        {
            if (slot >= 0 && !_fresh[slot] && !stale.Contains(slot))
            {
                _ = Get(slot);
                stale.Add(slot);
            }
        }

        if (stale.Count == 1)
        {
            Compute(stale[0]);
        }
        else if (stale.Count > 1)
        {
            // One matcher per task: matchers are independent, and each writes only its own row.
            Parallel.ForEach(stale, Compute);
        }
    }

    /// <summary>A slot's mask as last computed.</summary>
    /// <param name="slot">The slot.</param>
    /// <returns>The <see cref="MaskWords"/> words of its row.</returns>
    /// <exception cref="InvalidOperationException">The slot is free or its mask is stale.</exception>
    public ReadOnlySpan<uint> Mask(int slot) => new(Row(slot), MaskWords);

    /// <summary>Advances a slot's matcher by the token sampled for its sequence; the slot's mask becomes stale.</summary>
    /// <param name="slot">The slot.</param>
    /// <param name="token">The sampled token.</param>
    /// <returns>True when the grammar accepted the token; false when it does not allow it (the matcher is unchanged).</returns>
    /// <exception cref="GrammarException">The token is out of range or the matcher is in its error state.</exception>
    public bool Advance(int slot, int token)
    {
        var matcher = Get(slot);
        _fresh[slot] = false;
        return matcher.TryConsume(token);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        for (var slot = 0; slot < _matchers.Length; slot++)
        {
            _matchers[slot]?.Dispose();
            _matchers[slot] = null;
        }

        Count = 0;
        FreeMasks();
        GC.SuppressFinalize(this);
    }

    /// <summary>The address of a slot's fresh mask row.</summary>
    internal uint* Row(int slot)
    {
        ObjectDisposedException.ThrowIf(_masks == null, this);
        _ = Get(slot);
        if (!_fresh[slot])
        {
            throw new InvalidOperationException($"The mask of grammar slot {slot} is stale; call ComputeMasks first.");
        }

        return _masks + ((long)slot * MaskWords);
    }

    private void Compute(int slot)
    {
        _matchers[slot]!.ComputeMask(_masks + ((long)slot * MaskWords));
        _fresh[slot] = true;
    }

    private void FreeMasks()
    {
        var masks = _masks;
        _masks = null;
        if (masks != null)
        {
            NativeMemory.AlignedFree(masks);
        }
    }
}
