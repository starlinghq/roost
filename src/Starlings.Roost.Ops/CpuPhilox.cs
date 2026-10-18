namespace Starlings.Roost.Ops;

/// <summary>
/// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC11), the counter-based generator behind
/// <see cref="CpuOps.Sample"/>. The multipliers and Weyl constants are the published ones; the known-answer vectors are in the tests.
/// </summary>
internal static class CpuPhilox
{
    private const uint M0 = 0xD2511F53;
    private const uint M1 = 0xCD9E8D57;
    private const uint W0 = 0x9E3779B9;
    private const uint W1 = 0xBB67AE85;

    /// <summary>Ten rounds over a four-word counter and two-word key.</summary>
    internal static (uint R0, uint R1, uint R2, uint R3) Block(uint c0, uint c1, uint c2, uint c3, uint k0, uint k1)
    {
        for (int i = 0; i < 10; i++)
        {
            ulong p0 = (ulong)M0 * c0;
            ulong p1 = (ulong)M1 * c2;
            uint hi0 = (uint)(p0 >> 32);
            uint lo0 = (uint)p0;
            uint hi1 = (uint)(p1 >> 32);
            uint lo1 = (uint)p1;
            c0 = hi1 ^ c1 ^ k0;
            c1 = lo1;
            c2 = hi0 ^ c3 ^ k1;
            c3 = lo0;
            k0 += W0;
            k1 += W1;
        }

        return (c0, c1, c2, c3);
    }

    /// <summary>
    /// The uniform draw of one sampling row: key = (low, high) words of the seed, counter = (low, high) words of the offset and two zero words;
    /// the first output word's top 24 bits scaled by 2^-24, so the value is in [0, 1) and exactly representable in float32.
    /// </summary>
    internal static double Uniform(ulong seed, ulong offset)
    {
        var (r0, _, _, _) = Block((uint)offset, (uint)(offset >> 32), 0, 0, (uint)seed, (uint)(seed >> 32));
        return (r0 >> 8) * (1.0 / 16777216.0);
    }
}
