using System.Collections.Immutable;
using System.Security.Cryptography;
using Starlings.Roost.Abi;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Weights;
using Starlings.Roost.WorkerHost;
using static Starlings.Roost.Worker.Tests.SyntheticCheckpoint;

namespace Starlings.Roost.Worker.Tests;

/// <summary>R-100, CPU half: host weights released, restored and replaced through the R-024 verified stream.</summary>
public sealed class HostWeightResidencyTests : IDisposable
{
    private readonly DirectoryInfo _dir = Directory.CreateTempSubdirectory("roost-r100-");

    public void Dispose() => _dir.Delete(recursive: true);

    private static SyntheticTensor[] Tensors(int seed) => [Bf16("embed", seed, 10, 4), F32("norm", seed + 1, 4)];

    private static EagerWeightSlot[] Slots() =>
    [
        new("embed", DType.BF16, 40, ["embed"], null),
        new("norm", DType.F32, 4, ["norm"], null),
    ];

    private static byte[] Digest(byte[] file) => SHA256.HashData(file);

    [Fact]
    public void ReleaseAndRestoreRoundTripTheSameVerifiedBytes()
    {
        var file = Build(Tensors(1));
        var path = Write(_dir, file);
        using var residency = new HostWeightResidency(Slots(), path, Lock(file));
        var before = Bytes(residency);

        var released = residency.Release();
        Assert.Equal(ControlStatus.Ok, released.Status);
        Assert.True(released.Bytes > 0);
        Assert.False(residency.IsResident);
        Assert.True(residency.ResidentDigest.IsDefault);

        var restored = residency.Restore(Digest(file));
        Assert.Equal(ControlStatus.Ok, restored.Status);
        Assert.Equal(released.Bytes, restored.Bytes);
        Assert.Equal(Digest(file), residency.ResidentDigest.ToArray());
        Assert.Equal(before, Bytes(residency));
    }

    [Fact]
    public void ASubstitutedFileFailsTheRestoreAndTheWeightsStayReleased()
    {
        var file = Build(Tensors(1));
        var path = Write(_dir, file);
        using var residency = new HostWeightResidency(Slots(), path, Lock(file));
        _ = residency.Release();

        // Same size and header, other tensor bytes: only the digest at the end of the stream can tell.
        var substituted = Build(Tensors(9));
        Assert.Equal(file.Length, substituted.Length);
        File.WriteAllBytes(path, substituted);

        var restored = residency.Restore(Digest(file));
        Assert.Equal(ControlStatus.WeightsDigest, restored.Status);
        Assert.Contains("SHA-256", restored.Message, StringComparison.Ordinal);
        Assert.False(residency.IsResident);
    }

    [Fact]
    public void ARestoreToADigestTheSourceIsNotLockedAtIsRefusedWithoutReading()
    {
        var file = Build(Tensors(1));
        var path = Write(_dir, file);
        using var residency = new HostWeightResidency(Slots(), path, Lock(file));
        var other = Digest(Build(Tensors(2)));

        Assert.Equal(ControlStatus.WeightsDigest, residency.Restore(other).Status); // resident: compared, not reloaded
        _ = residency.Release();
        File.Delete(path); // nothing to read: the lock decides first
        Assert.Equal(ControlStatus.WeightsDigest, residency.Restore(other).Status);
    }

    [Fact]
    public void AReplacementIsSwappedInOnlyAfterItsDigestMatched()
    {
        var file = Build(Tensors(1));
        var path = Write(_dir, file);
        using var residency = new HostWeightResidency(Slots(), path, Lock(file));
        var original = Bytes(residency);

        var next = Build(Tensors(5));
        var nextPath = Write(_dir, next, "next.safetensors");
        var source = new WeightSource(nextPath, next.Length, ImmutableArray.Create(Digest(next)));

        // A lock the file does not match: refused, and the old weights stay resident and unchanged.
        var wrongLock = source with { Sha256 = ImmutableArray.Create(Digest(file)) };
        var refused = residency.Replace(wrongLock);
        Assert.Equal(ControlStatus.WeightsDigest, refused.Status);
        Assert.Equal(original, Bytes(residency));
        Assert.Equal(path, residency.SourcePath);

        var replaced = residency.Replace(source);
        Assert.Equal(ControlStatus.Ok, replaced.Status);
        Assert.Equal(Digest(next), residency.ResidentDigest.ToArray());
        Assert.NotEqual(original, Bytes(residency));
        Assert.Equal(nextPath, residency.SourcePath);

        // A release and restore now bring back the replacement.
        _ = residency.Release();
        Assert.Equal(ControlStatus.WeightsDigest, residency.Restore(Digest(file)).Status);
        Assert.Equal(ControlStatus.Ok, residency.Restore(Digest(next)).Status);
    }

    [Fact]
    public void AReplacementWhileReleasedIsVerifiedAndBecomesWhatTheRestoreLoads()
    {
        var file = Build(Tensors(1));
        var path = Write(_dir, file);
        using var residency = new HostWeightResidency(Slots(), path, Lock(file));
        _ = residency.Release();

        var next = Build(Tensors(5));
        var nextPath = Write(_dir, next, "next.safetensors");
        var replaced = residency.Replace(new WeightSource(nextPath, next.Length, ImmutableArray.Create(Digest(next))));
        Assert.Equal(ControlStatus.Ok, replaced.Status);
        Assert.Equal(0, replaced.Bytes);
        Assert.False(residency.IsResident);

        Assert.Equal(ControlStatus.Ok, residency.Restore(Digest(next)).Status);
        Assert.Equal(Digest(next), residency.ResidentDigest.ToArray());
    }

    private static unsafe byte[] Bytes(HostWeightResidency residency)
    {
        var spans = residency.Weights!.Spans;
        return [.. new ReadOnlySpan<byte>((void*)spans[0].Address, 80), .. new ReadOnlySpan<byte>((void*)spans[1].Address, 16)];
    }
}
