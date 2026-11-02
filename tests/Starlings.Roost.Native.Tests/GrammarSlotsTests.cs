using Starlings.Roost.Native.Tests.Support;

namespace Starlings.Roost.Native.Tests;

/// <summary>The slot table behind C2's grammar_slot: assignment, stale masks and parallel mask computation.</summary>
public sealed class GrammarSlotsTests
{
    [Fact]
    public void SlotsAreAssignedLowestFirstAndReused()
    {
        var tokenizer = Grammar.Require();
        using var slots = new GrammarSlots(2, tokenizer.VocabSize);
        Assert.Equal(0, slots.Assign(tokenizer.CreateMatcher(GrammarSpec.JsonObject)));
        Assert.Equal(1, slots.Assign(tokenizer.CreateMatcher(GrammarSpec.JsonObject)));
        using var extra = tokenizer.CreateMatcher(GrammarSpec.JsonObject);
        _ = Assert.Throws<InvalidOperationException>(() => slots.Assign(extra));
        slots.Release(0);
        Assert.Equal(1, slots.Count);
        Assert.Equal(0, slots.Assign(tokenizer.CreateMatcher(GrammarSpec.JsonObject)));
    }

    [Fact]
    public void AMatcherOfAnotherVocabularyIsRefused()
    {
        var tokenizer = Grammar.Require();
        using var slots = new GrammarSlots(1, 300);
        using var matcher = tokenizer.CreateMatcher(GrammarSpec.JsonObject);
        _ = Assert.Throws<ArgumentException>(() => slots.Assign(matcher));
    }

    [Fact]
    public void AMaskIsStaleAfterAdvanceUntilRecomputed()
    {
        var tokenizer = Grammar.Require();
        using var slots = new GrammarSlots(1, tokenizer.VocabSize);
        int slot = slots.Assign(tokenizer.CreateMatcher(new GrammarSpec(GrammarKind.Regex, "ab")));
        _ = Assert.Throws<InvalidOperationException>(() => slots.Mask(slot).Length);
        slots.ComputeMasks([slot]);
        Assert.Equal(['a'], Grammar.AllowedTokens(slots.Mask(slot), slots.VocabSize));
        Assert.False(slots.Advance(slot, 'b'));
        Assert.True(slots.Advance(slot, 'a'));
        _ = Assert.Throws<InvalidOperationException>(() => slots.Mask(slot).Length);
        slots.ComputeMasks([slot, -1, slot]);
        Assert.Equal(['b'], Grammar.AllowedTokens(slots.Mask(slot), slots.VocabSize));
    }

    [Fact]
    public void MasksOfManySlotsAreComputedInParallelIntoTheirOwnRows()
    {
        var tokenizer = Grammar.Require();
        const int n = 8;
        using var slots = new GrammarSlots(n, tokenizer.VocabSize);
        var expected = new int[n];
        for (var i = 0; i < n; i++)
        {
            // Slot i allows exactly one first byte, 'a' + i.
            char c = (char)('a' + i);
            expected[i] = c;
            Assert.Equal(i, slots.Assign(tokenizer.CreateMatcher(new GrammarSpec(GrammarKind.Regex, c + "z"))));
        }

        slots.ComputeMasks([.. Enumerable.Range(0, n)]);
        for (var i = 0; i < n; i++)
        {
            Assert.Equal([expected[i]], Grammar.AllowedTokens(slots.Mask(i), slots.VocabSize));
        }
    }

    [Fact]
    public void AFreeSlotIsRefused()
    {
        var tokenizer = Grammar.Require();
        using var slots = new GrammarSlots(2, tokenizer.VocabSize);
        _ = Assert.Throws<InvalidOperationException>(() => slots.ComputeMasks([1]));
        _ = Assert.Throws<ArgumentOutOfRangeException>(() => slots.Get(2));
    }
}
