using System.Collections.Immutable;
using Starlings.Roost.Abi;
using Starlings.Roost.Engine.Admission;
using Starlings.Roost.Engine.Blocks;
using Starlings.Roost.Engine.Tests.Blocks;
using Starlings.Roost.Requests;
using Starlings.Roost.State;
using static Starlings.Roost.Engine.Tests.Scheduling.PlannerHarness;

namespace Starlings.Roost.Engine.Tests.Admission;

public sealed class AdmissionControllerTests
{
    private const int Context = 64;

    private static AdmissionController Make(BlockManager m, IDeviceMemorySource? memory = null, int maxLive = 0, long floor = 0, IAdapterSizeSource? adapters = null) =>
        new(m, new AdmissionOptions { ContextLength = Context, MaxLiveSequences = maxLive, MinFreeDeviceBytes = floor }, memory, adapters);

    private static Request WithRows(ulong id, int prompt, int? maxTokens, int n = 1, int beam = 1, int? bestOf = null)
    {
        var r = Req(id, prompt, maxTokens);
        return r with { Sampling = r.Sampling with { N = n, BeamWidth = beam, BestOf = bestOf } };
    }

    [Fact]
    public void WorstCaseCountsPromptPlusMaxTokensPerGroupAndOneSlotPerRow()
    {
        var groups = ImmutableArray.Create(TestGroups.Full(0, 4), TestGroups.Sliding(1, 2, 5), TestGroups.Recurrent(2, snapshotSlots: 0));
        // prompt 10 + 6 new = 16 tokens, +1 for the token being written = 17.
        var r = WorstCaseReservation.Compute(groups, 10, 6, Context);
        Assert.Equal(new GroupReservation(2, 3), r[0]); // 5 blocks of 4, the 2 full prompt blocks shared
        Assert.Equal(new GroupReservation(0, 4), r[1]); // window 5 + block 2 + 1 = 8 tokens = 4 blocks, whatever the length
        Assert.Equal(new GroupReservation(0, 1), r[2]);
        Assert.Equal(14, r[0].For(4)); // the shared blocks once, then three more private tails
    }

    [Fact]
    public void AnUnsetMaxTokensIsReservedUpToTheContextLength()
    {
        var groups = ImmutableArray.Create(TestGroups.Full(0, 4));
        Assert.Equal(Context, WorstCaseReservation.WorstTokens(10, null, Context));
        Assert.Equal(Context, WorstCaseReservation.WorstTokens(10, 1000, Context));
        Assert.Equal(17, (int)WorstCaseReservation.Compute(groups, 10, null, Context)[0].For(1)); // ceil(65 / 4)
    }

    [Fact]
    public void RowsAreTheLargestOfNBestOfAndBeamWidth()
    {
        Assert.Equal(1, WorstCaseReservation.Rows(new SamplingParameters()));
        Assert.Equal(4, WorstCaseReservation.Rows(new SamplingParameters { N = 4 }));
        Assert.Equal(6, WorstCaseReservation.Rows(new SamplingParameters { N = 2, BestOf = 6 }));
        Assert.Equal(5, WorstCaseReservation.Rows(new SamplingParameters { BeamWidth = 5 }));
    }

    [Fact]
    public void AdmitsUntilTheWorstCasesFillAPoolThenRefusesWithInsufficientMemory()
    {
        var m = Hybrid(attentionBlocks: 20, slots: 8); // 20 blocks of 4
        var a = Make(m);
        // prompt 4 + 8 new = 12, +1 = 13 tokens = 4 blocks each: five fit.
        for (ulong i = 1; i <= 5; i++)
        {
            Assert.True(a.TryAdmit(Req(i, 4, 8), out var e), e?.Message);
        }

        Assert.Equal(20, a.Reserved(0));
        Assert.Equal(5, a.Reserved(1));
        Assert.False(a.TryAdmit(Req(6, 4, 8), out var error));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error!.Code);
        Assert.Equal(503, error.HttpStatus);
        Assert.Contains("Retry", error.Message, StringComparison.Ordinal);

        a.Release(3);
        Assert.True(a.TryAdmit(Req(6, 4, 8), out _));
        Assert.Equal(20, a.Reserved(0));
    }

    [Fact]
    public void ASlotPoolSmallerThanTheBlocksRefusesOnTheSlots()
    {
        var a = Make(Hybrid(attentionBlocks: 100, slots: 2));
        Assert.True(a.TryAdmit(Req(1, 4, 4), out _));
        Assert.True(a.TryAdmit(Req(2, 4, 4), out _));
        Assert.False(a.TryAdmit(Req(3, 4, 4), out var error));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error!.Code);
        Assert.Contains("recurrent", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARequestThatCanNeverFitSaysSoAndIsNotRetryable()
    {
        var a = Make(Hybrid(attentionBlocks: 6, slots: 4));
        Assert.False(a.TryAdmit(Req(1, 4, 40), out var error)); // 45 tokens = 12 blocks > 6
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error!.Code);
        Assert.Equal("max_tokens", error.Param);
        Assert.Contains("will not help", error.Message, StringComparison.Ordinal);
        Assert.Equal(0, a.Reserved(0));
    }

    [Fact]
    public void ATooLongPromptIsPromptTooLong()
    {
        var a = Make(Hybrid(attentionBlocks: 100, slots: 4));
        Assert.False(a.TryAdmit(Req(1, Context, 1), out var byContext));
        Assert.Equal(RoostErrorCode.PromptTooLong, byContext!.Code);
        Assert.Equal(400, byContext.HttpStatus);

        var small = Make(Hybrid(attentionBlocks: 3, slots: 4));
        Assert.False(small.TryAdmit(Req(2, 20, 1), out var byPool)); // 21 tokens = 6 blocks > 3
        Assert.Equal(RoostErrorCode.PromptTooLong, byPool!.Code);
    }

    [Fact]
    public void BadOptionsAndDuplicatesAreInvalidOptions()
    {
        var a = Make(Hybrid(20, 4));
        Assert.False(a.TryAdmit(Req(1, 4, 0), out var zero));
        Assert.Equal(RoostErrorCode.InvalidOption, zero!.Code);
        Assert.False(a.TryAdmit(Req(1, 0, 4), out var empty));
        Assert.Equal(RoostErrorCode.InvalidOption, empty!.Code);
        Assert.False(a.TryAdmit(WithRows(1, 4, 4, n: 0), out var n0));
        Assert.Equal(RoostErrorCode.InvalidOption, n0!.Code);

        Assert.True(a.TryAdmit(Req(2, 4, 4), out _));
        var reserved = a.Reserved(0);
        Assert.False(a.TryAdmit(Req(2, 4, 4), out var dup));
        Assert.Equal(RoostErrorCode.InvalidOption, dup!.Code);
        Assert.Equal(reserved, a.Reserved(0)); // the duplicate neither reserved nor released anything
        Assert.Equal(1, a.LiveRows);
    }

    [Fact]
    public void TheLiveRowLimitIsQueueFull()
    {
        var a = Make(Hybrid(200, 16), maxLive: 3);
        Assert.True(a.TryAdmit(Req(1, 4, 4), out _));
        Assert.True(a.TryAdmit(WithRows(2, 4, 4, n: 2), out _));
        Assert.False(a.TryAdmit(Req(3, 4, 4), out var error));
        Assert.Equal(RoostErrorCode.QueueFull, error!.Code);
        Assert.Equal(429, error.HttpStatus);
        a.Release(1);
        Assert.True(a.TryAdmit(Req(3, 4, 4), out _));
    }

    [Fact]
    public void ForkedBranchRowsAreReservedAsSequencesAtAdmission()
    {
        var m = Hybrid(attentionBlocks: 100, slots: 4);
        var a = Make(m);
        // prompt 9, 4 new: 14 tokens = 4 blocks; the 2 full prompt blocks are shared, each row owns 2 more; 3 rows.
        Assert.True(a.TryAdmit(WithRows(1, 9, 4, n: 3), out _));
        Assert.Equal(2 + (3 * 2), a.Reserved(0));
        Assert.Equal(3, a.Reserved(1)); // a recurrent slot per row, forked or not yet
        Assert.Equal(3, a.ReservedRows);
        Assert.Equal(1, a.LiveRows);

        Assert.True(a.TryAdmitFork(1, 2, out _));
        Assert.True(a.TryAdmitFork(2, 3, out _)); // a fork of a branch belongs to the same request
        Assert.Equal(3, a.LiveRows);
        Assert.Equal(8, a.Reserved(0)); // nothing more was reserved: the rows were already counted
        Assert.Equal(3, a.Reserved(1));
    }

    [Fact]
    public void AForkBeyondTheReservedRowsReservesOneMoreOrIsRefused()
    {
        var m = Hybrid(attentionBlocks: 100, slots: 2);
        var a = Make(m);
        Assert.True(a.TryAdmit(Req(1, 9, 4), out _));
        Assert.True(a.TryAdmitFork(1, 2, out _)); // second row: one more slot
        Assert.Equal(2, a.Reserved(1));
        Assert.Equal(2 + (2 * 2), a.Reserved(0));
        Assert.False(a.TryAdmitFork(1, 3, out var error)); // the third needs a third slot
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error!.Code);
        Assert.Equal(2, a.Reserved(1));
        Assert.Equal(2, a.LiveRows);
        Assert.False(a.TryAdmitFork(99, 4, out var unknown));
        Assert.Equal(RoostErrorCode.InvalidOption, unknown!.Code);
    }

    [Fact]
    public void ReleasingRowsReturnsTheirShareAndTheSharedBlocksWithTheLast()
    {
        var a = Make(Hybrid(attentionBlocks: 100, slots: 8));
        Assert.True(a.TryAdmit(WithRows(1, 9, 4, n: 3), out _));
        Assert.True(a.TryAdmitFork(1, 2, out _));
        Assert.Equal(8, a.Reserved(0));
        a.Release(2);
        Assert.Equal(6, a.Reserved(0)); // 2 shared + 2 rows of 2
        Assert.Equal(2, a.Reserved(1));
        a.Release(1);
        Assert.Equal(0, a.Reserved(0)); // the last live row: the shared blocks and the never-forked row go too
        Assert.Equal(0, a.Reserved(1));
        Assert.Equal(0, a.ReservedRows);
        a.Release(1); // a second release is ignored
        a.Release(777);
        Assert.Equal(0, a.Reserved(0));
    }

    [Fact]
    public void TheDeviceFloorRefusesWhenCoTenantsTookTheMemory()
    {
        var source = new FixedDeviceMemorySource(24_000, 10_000);
        var a = Make(Hybrid(100, 8), source, floor: 4_000);
        Assert.True(a.TryAdmit(Req(1, 4, 4), out _));
        source.Set(24_000, 3_999);
        Assert.False(a.TryAdmit(Req(2, 4, 4), out var error));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error!.Code);
        Assert.Equal(1, a.LiveRows);
        Assert.False(a.TryAdmitFork(1, 3, out var forkError));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, forkError!.Code);
        source.Set(24_000, 4_000);
        Assert.True(a.TryAdmit(Req(2, 4, 4), out _)); // the ceiling came back: refusal was never an abort
    }

    [Fact]
    public void AnUnreadableSourceLeavesTheDecisionToThePools()
    {
        var source = new FixedDeviceMemorySource(24_000, 0);
        source.MakeUnreadable();
        var a = Make(Hybrid(100, 8), source, floor: 4_000);
        Assert.True(a.TryAdmit(Req(1, 4, 4), out _));
    }

    [Fact]
    public void ThePoolBudgetIsFreeMemoryLessWorkspaceHeadroomAndFloor()
    {
        var source = new FixedDeviceMemorySource(24_000, 20_000);
        var options = new AdmissionOptions { ContextLength = Context, WorkspaceBytes = 3_000, HeadroomBytes = 1_000, MinFreeDeviceBytes = 500 };
        Assert.True(AdmissionController.TryComputePoolBudget(source, options, out var budget, out var none));
        Assert.Null(none);
        Assert.Equal(15_500, budget);

        // The budget feeds BlockSizing: the pools it makes fit in it.
        var groups = TestGroups.Qwen35Shaped();
        var big = new FixedDeviceMemorySource(0, 4L << 30);
        Assert.True(AdmissionController.TryComputePoolBudget(big, options, out var bytes, out _));
        var capacities = BlockSizing.Compute(groups, bytes, maxSequences: 4);
        var used = capacities.Sum(c => ((long)c.Blocks + c.SnapshotSlots) * groups[c.Group].BytesPerBlock);
        Assert.InRange(used, 1, bytes);

        source.Set(24_000, 4_400);
        Assert.False(AdmissionController.TryComputePoolBudget(source, options, out _, out var error));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, error!.Code);
        source.MakeUnreadable();
        Assert.False(AdmissionController.TryComputePoolBudget(source, options, out _, out var unread));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, unread!.Code);
    }

    [Fact]
    public void AdapterMemoryIsReservedOncePerResidentAdapter()
    {
        var adapter = new StateGroupDescriptor
        {
            Index = 2,
            Name = "adapter",
            Kind = StateGroupKind.Adapter,
            Layers = [],
            BlockSize = 1,
            DType = DType.BF16,
            BytesPerBlock = 1000,
        };
        ImmutableArray<StateGroupDescriptor> groups = [TestGroups.Full(0, 4), TestGroups.Recurrent(1, snapshotSlots: 0), adapter];
        var m = new BlockManager(groups, [new GroupCapacity(0, 100, 0), new GroupCapacity(1, 8, 0), new GroupCapacity(2, 5, 0)]);
        var sizes = new MapAdapterSizes { [7] = 2_500, [8] = 3_000, [9] = 99_000 };
        var a = Make(m, adapters: sizes);

        Request Lora(ulong id, ulong handle) => Req(id, 4, 4) with { Lora = new LoraHandle(handle) };
        Assert.True(a.TryAdmit(Lora(1, 7), out _));
        Assert.Equal(3, a.Reserved(2)); // ceil(2500 / 1000)
        Assert.True(a.TryAdmit(Lora(2, 7), out _));
        Assert.Equal(3, a.Reserved(2)); // the same adapter is not counted twice

        Assert.False(a.TryAdmit(Lora(3, 8), out var full)); // 3 more blocks do not fit beside the first 3 of 5
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, full!.Code);
        Assert.Equal(2, a.LiveRows);
        Assert.False(a.TryAdmit(Lora(4, 9), out var never));
        Assert.Contains("will not help", never!.Message, StringComparison.Ordinal);
        Assert.False(a.TryAdmit(Lora(5, 1234), out var unknown));
        Assert.Equal(RoostErrorCode.ProfileUnsupported, unknown!.Code);

        a.Release(1);
        Assert.Equal(3, a.Reserved(2)); // request 2 still uses it
        a.Release(2);
        Assert.Equal(0, a.Reserved(2));
        Assert.True(a.TryAdmit(Lora(3, 8), out _));
        Assert.Equal(3, a.Reserved(2));
    }

    [Fact]
    public void AGroupKindAdmissionDoesNotKnowIsRejectedAtConstruction()
    {
        var cross = TestGroups.Full(0, 4) with { Kind = StateGroupKind.Cross };
        var m = new BlockManager([cross], [new GroupCapacity(0, 4, 0)]);
        Assert.Throws<NotSupportedException>(() => Make(m));
    }

    private sealed class MapAdapterSizes : Dictionary<ulong, long>, IAdapterSizeSource
    {
        public bool TryGetBytes(LoraHandle adapter, out long bytes) => TryGetValue(adapter.Value, out bytes);
    }
}
