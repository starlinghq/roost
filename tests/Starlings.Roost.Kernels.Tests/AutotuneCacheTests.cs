using System.Text.Json.Nodes;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Kernels.Tests;

public sealed class AutotuneCacheTests : IDisposable
{
    private const string Driver = "580.65.06";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "roost-autotune-" + Guid.NewGuid().ToString("N"));

    private static readonly ShapeBucket Bucket = new(OpId.Gemm, "act=BF16;tokens=256");

    private string CachePath => Path.Combine(_dir, "sub", "autotune.json");

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, recursive: true);
        }
    }

    private static VariantCandidate Candidate(string id, char code = 'c') =>
        new(OpId.Gemm, id, VariantCandidate.DigestOf(OpId.Gemm, id, new string(code, 64)));

    private AutotuneCache Saved(params (VariantCandidate Candidate, double Median)[] entries)
    {
        var cache = AutotuneCache.Open(CachePath, 86, Driver);
        foreach (var (candidate, median) in entries)
        {
            cache.Record(Bucket, candidate, new AutotuneMeasurement(median, 20));
        }

        cache.Save();
        return cache;
    }

    [Fact]
    public void AMissingFileOpensEmpty()
    {
        var cache = AutotuneCache.Open(CachePath, 86, Driver);
        Assert.Equal(AutotuneLoadOutcome.Missing, cache.LoadOutcome);
        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void EntriesRoundTripThroughTheFile()
    {
        var a = Candidate("gemm-a");
        var b = Candidate("gemm-b");
        Saved((b, 9.25), (a, 12.5));

        var reopened = AutotuneCache.Open(CachePath, 86, Driver);
        Assert.Equal(AutotuneLoadOutcome.Loaded, reopened.LoadOutcome);
        Assert.Equal(0, reopened.Discarded);
        Assert.True(reopened.TryGet(Bucket, a, out var ma));
        Assert.Equal(new AutotuneMeasurement(12.5, 20), ma);
        Assert.True(reopened.TryGet(Bucket, b, out var mb));
        Assert.Equal(9.25, mb.MedianMicroseconds);
        Assert.Equal(["gemm-a", "gemm-b"], reopened.Entries.Select(e => e.VariantId));
        Assert.All(reopened.Entries, e => Assert.Equal((86, Driver), (e.Sm, e.DriverVersion)));
        Assert.Empty(Directory.GetFiles(Path.GetDirectoryName(CachePath)!, "*.tmp", SearchOption.AllDirectories));
    }

    [Fact]
    public void TheFileIsVersionedAndDeterministic()
    {
        Saved((Candidate("gemm-b"), 2), (Candidate("gemm-a"), 1));
        var first = File.ReadAllBytes(CachePath);
        var json = JsonNode.Parse(first)!;
        Assert.Equal("roost-autotune", (string?)json["format"]);
        Assert.Equal(AutotuneCache.FormatVersion, (int)json["version"]!);
        Assert.Equal(86, (int)json["sm"]!);
        Assert.Equal(Driver, (string?)json["driver"]);
        Assert.Equal((uint)OpId.Gemm, (uint)json["entries"]![0]!["op"]!);
        Assert.Equal("gemm-a", (string?)json["entries"]![0]!["variant"]);
        Assert.NotNull(json["entries"]![0]!["median_us"]);

        File.Delete(CachePath);
        Saved((Candidate("gemm-a"), 1), (Candidate("gemm-b"), 2));
        Assert.Equal(first, File.ReadAllBytes(CachePath));
    }

    [Theory]
    [InlineData(89, Driver)]
    [InlineData(86, "590.10.01")]
    public void ADifferentSmOrDriverInvalidatesTheWholeFile(int sm, string driver)
    {
        Saved((Candidate("gemm-a"), 1));
        var cache = AutotuneCache.Open(CachePath, sm, driver);
        Assert.Equal(AutotuneLoadOutcome.DeviceChanged, cache.LoadOutcome);
        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void ADigestMismatchInvalidatesTheEntryOnLookup()
    {
        Saved((Candidate("gemm-a"), 1), (Candidate("gemm-b"), 2));
        var cache = AutotuneCache.Open(CachePath, 86, Driver);
        Assert.False(cache.TryGet(Bucket, Candidate("gemm-a", 'd'), out _));
        Assert.Equal(1, cache.Count);
        Assert.Equal(1, cache.Discarded);
        Assert.False(cache.TryGet(Bucket, Candidate("gemm-a"), out _));
        Assert.False(cache.TryGet(new ShapeBucket(OpId.Gemm, "act=BF16;tokens=512"), Candidate("gemm-b"), out _));
        Assert.True(cache.TryGet(Bucket, Candidate("gemm-b"), out _));
    }

    [Theory]
    [InlineData("")]
    [InlineData("{ not json")]
    [InlineData("[]")]
    [InlineData("null")]
    [InlineData("{\"format\":\"something-else\",\"version\":1,\"sm\":86,\"driver\":\"580.65.06\",\"entries\":[]}")]
    [InlineData("{\"format\":\"roost-autotune\",\"version\":\"one\"}")]
    public void ACorruptFileOpensEmptyAndTheNextSaveReplacesIt(string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(CachePath)!);
        File.WriteAllText(CachePath, content);
        var cache = AutotuneCache.Open(CachePath, 86, Driver);
        Assert.Equal(AutotuneLoadOutcome.Corrupt, cache.LoadOutcome);
        Assert.Equal(0, cache.Count);

        cache.Record(Bucket, Candidate("gemm-a"), new AutotuneMeasurement(3, 5));
        cache.Save();
        Assert.Equal(AutotuneLoadOutcome.Loaded, AutotuneCache.Open(CachePath, 86, Driver).LoadOutcome);
    }

    [Fact]
    public void AnotherFormatVersionOpensEmpty()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(CachePath)!);
        File.WriteAllText(CachePath, "{\"format\":\"roost-autotune\",\"version\":2,\"sm\":86,\"driver\":\"580.65.06\",\"entries\":[]}");
        Assert.Equal(AutotuneLoadOutcome.VersionMismatch, AutotuneCache.Open(CachePath, 86, Driver).LoadOutcome);
    }

    [Fact]
    public void InvalidOrDuplicateEntriesAreDroppedAndTheRestKept()
    {
        var good = Candidate("gemm-a");
        var digest = good.Digest;
        var entries = string.Join(',', [
            $"{{\"op\":{(uint)OpId.Gemm},\"bucket\":\"{Bucket.Key}\",\"variant\":\"gemm-a\",\"digest\":\"{digest}\",\"median_us\":4.5,\"samples\":9}}",
            $"{{\"op\":{(uint)OpId.Gemm},\"bucket\":\"{Bucket.Key}\",\"variant\":\"gemm-a\",\"digest\":\"{digest}\",\"median_us\":1.0,\"samples\":9}}",
            $"{{\"op\":{(uint)OpId.Gemm},\"bucket\":\"{Bucket.Key}\",\"variant\":\"gemm-b\",\"digest\":\"XYZ\",\"median_us\":1.0,\"samples\":9}}",
            $"{{\"op\":{(uint)OpId.Gemm},\"bucket\":\"{Bucket.Key}\",\"variant\":\"gemm-c\",\"digest\":\"{digest}\",\"median_us\":-1.0,\"samples\":9}}",
            $"{{\"op\":0,\"bucket\":\"{Bucket.Key}\",\"variant\":\"gemm-d\",\"digest\":\"{digest}\",\"median_us\":1.0,\"samples\":9}}",
            $"{{\"op\":{(uint)OpId.Gemm},\"bucket\":\"\",\"variant\":\"gemm-e\",\"digest\":\"{digest}\",\"median_us\":1.0,\"samples\":9}}",
            "null",
        ]);
        Directory.CreateDirectory(Path.GetDirectoryName(CachePath)!);
        File.WriteAllText(CachePath, $"{{\"format\":\"roost-autotune\",\"version\":1,\"sm\":86,\"driver\":\"{Driver}\",\"entries\":[{entries}]}}");

        var cache = AutotuneCache.Open(CachePath, 86, Driver);
        Assert.Equal(AutotuneLoadOutcome.Loaded, cache.LoadOutcome);
        Assert.Equal(6, cache.Discarded);
        Assert.Equal(1, cache.Count);
        Assert.True(cache.TryGet(Bucket, good, out var m));
        Assert.Equal(4.5, m.MedianMicroseconds);
    }

    [Fact]
    public void SaveReplacesTheFileAtomicallyAndLeavesNoTemporaryFile()
    {
        Saved((Candidate("gemm-a"), 1));
        var cache = AutotuneCache.Open(CachePath, 86, Driver);
        cache.Record(Bucket, Candidate("gemm-b"), new AutotuneMeasurement(2, 3));
        cache.Save();
        Assert.Equal(2, AutotuneCache.Open(CachePath, 86, Driver).Count);
        Assert.Equal([CachePath], Directory.GetFiles(Path.GetDirectoryName(CachePath)!));
    }

    [Fact]
    public void InvalidateRemovesABucketAndInvalidMeasurementsAreRefused()
    {
        var cache = AutotuneCache.InMemory(86, Driver);
        cache.Record(Bucket, Candidate("gemm-a"), new AutotuneMeasurement(1, 1));
        cache.Record(new ShapeBucket(OpId.Gemm, "*"), Candidate("gemm-a"), new AutotuneMeasurement(1, 1));
        Assert.Equal(1, cache.Invalidate(Bucket));
        Assert.Equal(1, cache.Count);
        Assert.Throws<ArgumentException>(() => cache.Record(Bucket, Candidate("gemm-a"), new AutotuneMeasurement(double.NaN, 1)));
        Assert.Throws<ArgumentException>(() => cache.Record(Bucket, Candidate("gemm-a"), new AutotuneMeasurement(1, 0)));
        Assert.Throws<ArgumentException>(() => cache.Record(new ShapeBucket(OpId.Residual, "*"), Candidate("gemm-a"), new AutotuneMeasurement(1, 1)));
        Assert.Throws<InvalidOperationException>(cache.Save);
        Assert.Throws<ArgumentException>(() => AutotuneCache.InMemory(86, "has space"));
    }
}
