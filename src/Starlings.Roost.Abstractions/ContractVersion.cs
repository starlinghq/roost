namespace Starlings.Roost;

/// <summary>
/// The version of the Roost contracts C1 to C9 (build plan section 5; the ADR is the "Contracts" section of
/// README.md).
/// </summary>
/// <remarks>
/// Version 1 is the first frozen set, the <c>v0</c> stage of the plan. Version 2 (2026-10-19) adds row strides for the
/// GatedDeltaRule q, k and v views (C3) and <c>RoostGatedDeltaRuleParams</c> (C1 ABI 1.1). Until the M0 gate (R-026) a contract may
/// still change through a stream A pull request; each such change raises <see cref="Current"/> and updates every
/// consumer in the same change. The M0 gate promotes the contracts to the <c>v1</c> stage.
/// </remarks>
public static class ContractVersion
{
    /// <summary>The current contract version, carried in the C8 handshake and segment header.</summary>
    public const int Current = 2;

    /// <summary>The plan's name for the current stage: <c>v0</c> until the M0 gate, then <c>v1</c>.</summary>
    public const string Stage = "v0";
}
