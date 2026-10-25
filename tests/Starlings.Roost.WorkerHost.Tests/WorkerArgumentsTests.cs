namespace Starlings.Roost.WorkerHost.Tests;

public sealed class WorkerArgumentsTests
{
    private static readonly string Sha = new('a', 64);

    private static WorkerArguments Sample() => new()
    {
        Model = "qwen3.5-0.8b",
        WeightsPath = "/m/model.safetensors",
        WeightsSize = 1_746_000_000,
        WeightsSha256 = Sha,
        ManifestPath = "/k/manifest.toml",
        KernelDirectory = "/k",
        AllowUntested = true,
        Device = 2,
        MaxTokens = 64,
        MaxSequences = 4,
        MaxKvBlocks = 32,
        AttentionBlocks = 48,
        RecurrentSlots = 4,
    };

    [Fact]
    public void ArgumentsRoundTrip()
    {
        var a = Sample();
        Assert.Equal(a, WorkerArguments.Parse(a.ToArguments()));
    }

    [Fact]
    public void OptionalArgumentsDefault()
    {
        var a = Sample() with { ManifestPath = null, KernelDirectory = null, AllowUntested = false, Device = 0 };
        var args = a.ToArguments();
        Assert.DoesNotContain("--kernel-manifest", args);
        Assert.DoesNotContain("--kernels", args);
        Assert.DoesNotContain("--allow-untested", args);
        Assert.Equal(a, WorkerArguments.Parse(args));
    }

    [Fact]
    public void TheSegmentFlagAndUnknownArgumentsAreSkipped()
    {
        string[] args = [.. Sample().ToArguments(), WorkerEntry.SegmentFlag, "/dev/shm/roost-1", "--something-else"];
        Assert.Equal(Sample(), WorkerArguments.Parse(args));
    }

    [Theory]
    [InlineData("--model")]
    [InlineData("--weights")]
    [InlineData("--weights-size")]
    [InlineData("--weights-sha256")]
    [InlineData("--max-tokens")]
    [InlineData("--recurrent-slots")]
    public void AMissingRequiredArgumentIsNamed(string flag)
    {
        var args = Sample().ToArguments().ToList();
        int at = args.IndexOf(flag);
        args.RemoveRange(at, 2);
        var e = Assert.Throws<ArgumentException>(() => WorkerArguments.Parse(args));
        Assert.Contains(flag, e.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("--weights-size", "-1")]
    [InlineData("--max-tokens", "lots")]
    [InlineData("--weights-sha256", "ABC")]
    public void AMalformedValueIsRefused(string flag, string value)
    {
        var args = Sample().ToArguments().ToList();
        args[args.IndexOf(flag) + 1] = value;
        var e = Assert.Throws<ArgumentException>(() => WorkerArguments.Parse(args));
        Assert.Contains(flag, e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARepeatedOrDanglingArgumentIsRefused()
    {
        string[] repeated = [.. Sample().ToArguments(), "--model", "other"];
        Assert.Contains("twice", Assert.Throws<ArgumentException>(() => WorkerArguments.Parse(repeated)).Message, StringComparison.Ordinal);
        string[] dangling = [.. Sample().ToArguments().Where(static a => a != "--allow-untested"), "--device"];
        Assert.Contains("needs a value", Assert.Throws<ArgumentException>(() => WorkerArguments.Parse(dangling)).Message, StringComparison.Ordinal);
    }
}
