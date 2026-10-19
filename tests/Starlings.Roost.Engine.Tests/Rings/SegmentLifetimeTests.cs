using Starlings.Roost.Engine.Rings;

namespace Starlings.Roost.Engine.Tests.Rings;

/// <summary>
/// The segment lifetime rule: <see cref="RingSegment.Close"/> stops and joins every poller made with
/// <see cref="RingSegment.CreatePoller"/> before it releases the view, and keeps the view mapped if one does not end.
/// Each test holds a poller inside a poll with a gate, so the order is fixed by the test, not by timing. A poll that
/// reads the segment after an unmap would see <see cref="ObjectDisposedException"/> from the header access (kept in
/// <c>Fault</c>), not an access violation, because the unmapped state is published before the view is released.
/// </summary>
public sealed class SegmentLifetimeTests : IDisposable
{
    private static readonly TimeSpan Long = TimeSpan.FromSeconds(30);

    private readonly string _path = Path.Combine(RingSegment.DefaultDirectory(), $"roost-lifetime-test-{Guid.NewGuid():N}");
    private readonly RingSegment _segment;
    private readonly ManualResetEventSlim _inPoll = new();
    private readonly ManualResetEventSlim _gate = new();

    public SegmentLifetimeTests()
    {
        _segment = RingSegment.Create(_path, new SegmentOptions());
    }

    public void Dispose()
    {
        _gate.Set(); // never leave a poller parked
        Assert.True(_segment.Close(Long));
        _inPoll.Dispose();
        _gate.Dispose();
    }

    [Fact]
    public async Task CloseJoinsAPollerThatIsMidPollBeforeItUnmaps()
    {
        var polls = 0;
        ulong bytesSeen = 0;
        var expectedBytes = (ulong)_segment.Bytes;
        var token = TestContext.Current.CancellationToken;
        using var poller = _segment.CreatePoller("lifetime-mid-poll", () =>
        {
            _inPoll.Set();
            _gate.Wait(Long, token);
            bytesSeen = _segment.Header.SegmentBytes; // reads the mapping after the stop was requested
            polls++;
            return false;
        });
        poller.Start();
        Assert.True(_inPoll.Wait(Long, token));

        var close = Task.Run(() => _segment.Close(Long), token);
        while (!poller.StopRequested)
        {
            await Task.Yield(); // Close has taken its poller list and requested the stop once this is true
        }

        // The poller is parked inside a poll: Close must be waiting in the join, with the view still mapped.
        Assert.False(close.IsCompleted);
        Assert.True(_segment.IsMapped);
        Assert.False(poller.HasExited);

        _gate.Set();
        Assert.True(await close);
        Assert.True(poller.HasExited);
        Assert.False(_segment.IsMapped);
        Assert.Null(poller.Fault); // the read after the stop request hit a live mapping
        Assert.Equal(1, polls);
        Assert.Equal(expectedBytes, bytesSeen);
        Assert.Throws<ObjectDisposedException>(() => _segment.Header.SegmentBytes);
    }

    [Fact]
    public void CloseKeepsTheMappingWhenAPollerDoesNotEndInTime()
    {
        var reads = 0;
        var token = TestContext.Current.CancellationToken;
        using var poller = _segment.CreatePoller("lifetime-stuck", () =>
        {
            _inPoll.Set();
            _gate.Wait(Long, token); // ignores the stop request until the gate opens
            _ = _segment.Header.SegmentBytes;
            reads++;
            return false;
        });
        poller.Start();
        Assert.True(_inPoll.Wait(Long, token));

        Assert.False(_segment.Close(TimeSpan.FromMilliseconds(50)));
        Assert.True(_segment.IsMapped);
        Assert.False(poller.HasExited);

        _gate.Set();
        Assert.True(_segment.Close(Long));
        Assert.True(poller.HasExited);
        Assert.Null(poller.Fault);
        Assert.Equal(1, reads);
        Assert.False(_segment.IsMapped);
    }

    [Fact]
    public void APollerThatNeverStartedDoesNotHoldTheSegmentAndCannotStartAfterClose()
    {
        using var poller = _segment.CreatePoller("lifetime-unstarted", () => false);
        Assert.True(_segment.Close(Long));
        Assert.True(poller.HasExited);
        Assert.Throws<ObjectDisposedException>(poller.Start);
        Assert.Throws<ObjectDisposedException>(() => _segment.CreatePoller("late", () => false));
    }
}
