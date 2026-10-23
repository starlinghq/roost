namespace Starlings.Roost.Engine.Caching;

/// <summary>A 128-bit position in a prefix hash chain.</summary>
/// <param name="Lo">Low half.</param>
/// <param name="Hi">High half.</param>
public readonly record struct PrefixKey(ulong Lo, ulong Hi);

/// <summary>
/// The hash chain of the prefix cache (R-030): one chain per state group, started from a seed that names the group, its
/// block size, the LoRA id and a caller salt, and extended one attention block of token ids at a time. A block's key
/// therefore names every token before it, its group and its adapter, so two groups or two adapters never share a key.
/// </summary>
/// <remarks>
/// The mix is two lanes of the SplitMix64 finaliser over each 32-bit token id: deterministic across runs and platforms
/// (no <c>HashCode</c> randomisation) so recorded runs replay. It is not cryptographic. The cache never trusts a key
/// alone: a lookup also compares the parent node and the block's token ids (see <see cref="PrefixCache"/>).
/// </remarks>
public static class PrefixHash
{
    private const ulong LaneA = 0x9E3779B97F4A7C15UL;
    private const ulong LaneB = 0xC2B2AE3D27D4EB4FUL;

    /// <summary>The key a chain starts from.</summary>
    /// <param name="salt">A caller salt (for example a model digest).</param>
    /// <param name="group">State group index.</param>
    /// <param name="blockSize">Tokens per block of the group.</param>
    /// <param name="lora">LoRA id; 0 for none.</param>
    /// <returns>The seed key.</returns>
    public static PrefixKey Seed(ulong salt, int group, int blockSize, ulong lora)
    {
        var a = Mix(salt ^ LaneA);
        var b = Mix(salt ^ LaneB);
        a = Mix(a + (ulong)(uint)group + 1);
        b = Mix(b ^ (((ulong)(uint)group << 32) | (uint)blockSize));
        a = Mix(a ^ lora);
        b = Mix(b + (lora * LaneB));
        return new PrefixKey(a, b);
    }

    /// <summary>Extends a chain by one block.</summary>
    /// <param name="parent">The key of the previous block, or the seed.</param>
    /// <param name="tokens">The block's token ids.</param>
    /// <returns>The key of the block.</returns>
    public static PrefixKey Next(PrefixKey parent, ReadOnlySpan<int> tokens)
    {
        var a = parent.Lo;
        var b = parent.Hi;
        foreach (var t in tokens)
        {
            var v = (ulong)(uint)t;
            a = Mix(a ^ (v + LaneA));
            b = Mix(b + (v * LaneB) + 1);
        }

        // Fold the length so a short block never equals a prefix of a longer one.
        return new PrefixKey(Mix(a ^ (ulong)tokens.Length), Mix(b + (ulong)tokens.Length));
    }

    private static ulong Mix(ulong z)
    {
        z += LaneA;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}
