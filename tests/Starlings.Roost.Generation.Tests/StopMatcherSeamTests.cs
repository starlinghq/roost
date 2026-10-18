namespace Starlings.Roost.Generation.Tests;

public sealed class StopMatcherSeamTests
{
    [Fact]
    public void NoStopSequencesPassesTextThrough()
    {
        var matcher = PendingStopMatcherFactory.Instance.Create([]);
        Assert.Equal(new StopMatchResult("hello ", null), matcher.Push("hello "));
        Assert.Equal(new StopMatchResult("", null), matcher.Push(""));
        Assert.Equal(new StopMatchResult("world", null), matcher.Push("world"));
        Assert.Equal("", matcher.Flush());
    }

    [Fact]
    public void MatchersAreIndependent()
    {
        var a = PendingStopMatcherFactory.Instance.Create([]);
        var b = PendingStopMatcherFactory.Instance.Create([]);
        Assert.NotSame(a, b);
    }

    [Fact]
    public void StopSequencesAreRefusedUntilStarlingPublishesItsMatcher()
    {
        var ex = Assert.Throws<NotSupportedException>(() => PendingStopMatcherFactory.Instance.Create(["</s>"]));
        Assert.Contains("starling#124", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NullsAreRejected()
    {
        Assert.Throws<ArgumentNullException>(() => PendingStopMatcherFactory.Instance.Create(null!));
        Assert.Throws<ArgumentNullException>(() => PendingStopMatcherFactory.Instance.Create([]).Push(null!));
    }

    [Fact]
    public void StarlingStillKeepsItsMatcherInternal()
    {
        // Tripwire: when a Starling pin exposes a public matcher, this fails and the pending factory should be replaced.
        var type = typeof(Starlings.Generation.OnnxGenAI.IncrementalDetokenizer).Assembly.GetType("Starlings.Generation.OnnxGenAI.StopSequenceMatcher");
        Assert.NotNull(type);
        Assert.False(type.IsVisible);
    }
}
