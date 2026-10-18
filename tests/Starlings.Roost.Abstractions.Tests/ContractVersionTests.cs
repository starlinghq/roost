namespace Starlings.Roost.Tests;

public sealed class ContractVersionTests
{
    [Fact]
    public void ContractsAreNotFrozenBeforeR004() => Assert.Equal(0, ContractVersion.Current);
}
