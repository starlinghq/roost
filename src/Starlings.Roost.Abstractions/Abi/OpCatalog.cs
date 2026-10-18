namespace Starlings.Roost.Abi;

/// <summary>The phase in which an op of the vocabulary lands (build plan section 5, C3).</summary>
public enum OpPhase
{
    /// <summary>Not an op.</summary>
    None = 0,

    /// <summary>Implemented by every backend from M0.</summary>
    M0,

    /// <summary>Optional fused op, defined as its unfused parts.</summary>
    Fused,

    /// <summary>Reserved for M1.</summary>
    M1,

    /// <summary>Reserved for M2.</summary>
    M2,

    /// <summary>Reserved for M3.</summary>
    M3,

    /// <summary>Reserved for M4.</summary>
    M4,

    /// <summary>Reserved for M5.</summary>
    M5,
}

/// <summary>Classification of <see cref="OpId"/> values by phase, from their permanent numbering.</summary>
public static class OpCatalog
{
    /// <summary>The phase of <paramref name="op"/>.</summary>
    /// <param name="op">The op.</param>
    /// <returns>The phase; <see cref="OpPhase.None"/> for a value that is not an op.</returns>
    public static OpPhase PhaseOf(OpId op) => !Enum.IsDefined(op) || op == OpId.None
        ? OpPhase.None
        : (uint)op switch
        {
            < 64 => OpPhase.M0,
            < 128 => OpPhase.Fused,
            < 160 => OpPhase.M1,
            < 192 => OpPhase.M2,
            < 224 => OpPhase.M3,
            < 256 => OpPhase.M4,
            _ => OpPhase.M5,
        };

    /// <summary>Whether every backend implements <paramref name="op"/> from M0.</summary>
    /// <param name="op">The op.</param>
    /// <returns>True for an M0 op.</returns>
    public static bool IsM0(OpId op) => PhaseOf(op) == OpPhase.M0;
}
