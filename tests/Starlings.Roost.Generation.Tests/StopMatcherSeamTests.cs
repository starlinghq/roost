namespace Starlings.Roost.Generation.Tests;

public sealed class StopMatcherSeamTests
{
    private static IStopMatcher Create(params string[] stops) => StarlingStopMatcherFactory.Instance.Create(stops);

    [Fact]
    public void NoStopSequencesPassesTextThrough()
    {
        var matcher = Create();
        Assert.Equal(new StopMatchResult("hello ", null), matcher.Push("hello "));
        Assert.Equal(new StopMatchResult("", null), matcher.Push(""));
        Assert.Equal(new StopMatchResult("world", null), matcher.Push("world"));
        Assert.Equal("", matcher.Flush());
    }

    [Fact]
    public void MatchersAreIndependent()
    {
        var a = Create("</s>");
        var b = Create("</s>");
        Assert.NotSame(a, b);
        a.Push("x</");
        Assert.Equal(new StopMatchResult("y", null), b.Push("y"));
    }

    [Fact]
    public void APartialStopIsHeldBackUntilItMatches()
    {
        var matcher = Create("</s>");
        Assert.Equal(new StopMatchResult("abc", null), matcher.Push("abc</"));
        Assert.Equal(new StopMatchResult("", "</s>"), matcher.Push("s>tail"));
    }

    [Fact]
    public void APartialStopThatFailsIsReleased()
    {
        var matcher = Create("</s>");
        Assert.Equal(new StopMatchResult("a", null), matcher.Push("a</"));
        Assert.Equal(new StopMatchResult("</x", null), matcher.Push("x"));
    }

    [Fact]
    public void HeldBackTextIsFlushedAtTheEnd()
    {
        var matcher = Create("STOP");
        Assert.Equal(new StopMatchResult("end ", null), matcher.Push("end ST"));
        Assert.Equal("ST", matcher.Flush());
        Assert.Equal("", matcher.Flush());
    }

    [Fact]
    public void TheEarliestMatchWinsAndTextAfterItIsDropped()
    {
        var matcher = Create("bb", "a");
        Assert.Equal(new StopMatchResult("x", "a"), matcher.Push("xabb"));
    }

    [Fact]
    public void AtTheSameStartTheLongestMatchWins()
    {
        var matcher = Create("ab", "abc");
        Assert.Equal(new StopMatchResult("", "abc"), matcher.Push("abcd"));
    }

    [Fact]
    public void NullsAndEmptyStopsAreRejected()
    {
        Assert.Throws<ArgumentNullException>(() => StarlingStopMatcherFactory.Instance.Create(null!));
        Assert.Throws<ArgumentNullException>(() => Create().Push(null!));
        Assert.Throws<ArgumentException>(() => Create(""));
    }
}
