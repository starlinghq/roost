namespace Starlings.Roost;

/// <summary>
/// The version of the Roost contracts C1 to C6 (build plan section 5).
/// </summary>
/// <remarks>
/// Zero means the contracts are not frozen yet. R-004 sets it to 1 when the contracts merge as compiling stubs. After
/// that, every contract change is a stream A pull request that raises it and updates every consumer.
/// </remarks>
public static class ContractVersion
{
    /// <summary>The current contract version.</summary>
    public const int Current = 0;
}
