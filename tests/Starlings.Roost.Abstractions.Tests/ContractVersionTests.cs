using Starlings.Roost.Abi;
using Starlings.Roost.Plans;
using Starlings.Roost.Tests.Layout;

namespace Starlings.Roost.Tests;

public sealed class ContractVersionTests
{
    [Fact]
    public void ContractsAreAtVersionThreeStageV0()
    {
        Assert.Equal(3, ContractVersion.Current);
        Assert.Equal("v0", ContractVersion.Stage);
    }

    [Fact]
    public void KernelAbiMatchesHeader()
    {
        Assert.Equal<long>(KernelAbi.Major, NativeLayout.Defines["ROOST_ABI_MAJOR"]);
        Assert.Equal<long>(KernelAbi.Minor, NativeLayout.Defines["ROOST_ABI_MINOR"]);
        Assert.Equal((1u << 16) | 2u, KernelAbi.Version);
    }

    [Fact]
    public void SharedLayoutMatchesHeader()
    {
        Assert.Equal<long>(SharedLayout.Version, NativeLayout.Defines["ROOST_SHM_LAYOUT_VERSION"]);
        Assert.Equal<long>(SharedLayout.Alignment, NativeLayout.Defines["ROOST_SHM_ALIGNMENT"]);
        Assert.Equal<long>(SharedLayout.MagicSegment, NativeLayout.Defines["ROOST_SHM_MAGIC_SEGMENT"]);
        Assert.Equal<long>(SharedLayout.MagicPlan, NativeLayout.Defines["ROOST_SHM_MAGIC_PLAN"]);
        Assert.Equal<long>(SharedLayout.MagicResult, NativeLayout.Defines["ROOST_SHM_MAGIC_RESULT"]);
        Assert.Equal<long>(SharedLayout.MagicRing, NativeLayout.Defines["ROOST_SHM_MAGIC_RING"]);
    }

    [Theory]
    [InlineData(0x0001_0000u, false)]
    [InlineData(0x0001_0001u, false)]
    [InlineData(0x0001_0002u, true)]
    [InlineData(0x0001_0003u, true)]
    [InlineData(0x0002_0000u, false)]
    [InlineData(0x0000_0009u, false)]
    public void LoaderAcceptsEqualMajorAndMinorAtLeastCallers(uint libraryVersion, bool compatible) =>
        Assert.Equal(compatible, KernelAbi.IsCompatible(libraryVersion));
}
