using System.Collections.Immutable;
using System.Diagnostics;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Driver;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Tests;

/// <summary>
/// R-100, CPU half: sleep, wake and in-place weight update through <see cref="RoostChatClient"/>, over the in-process and the
/// process transport, with a worker that runs roost-tiny-hybrid on <c>CpuOps</c> from a locked safetensors file
/// (<see cref="TinyCpuExecutor"/>). The output before a sleep and after the wake is the same; a weights file substituted on disk
/// fails the wake or the update with <see cref="RoostErrorCode.AssetMismatch"/>; an update changes the output.
/// </summary>
public sealed class SleepWakeTests : IDisposable
{
    private static readonly ulong OtherSeed = TinyConfig.Hybrid.Seed + 1;

    private readonly DirectoryInfo _dir = Directory.CreateTempSubdirectory("roost-r100-chat-");

    public static TheoryData<TransportKind> Transports => [TransportKind.InProcess, TransportKind.Process];

    public static TheoryData<TransportKind, RoostSleepLevel> TransportsAndLevels => new()
    {
        { TransportKind.InProcess, RoostSleepLevel.KvCache },
        { TransportKind.InProcess, RoostSleepLevel.KvCacheAndWeights },
        { TransportKind.Process, RoostSleepLevel.KvCache },
        { TransportKind.Process, RoostSleepLevel.KvCacheAndWeights },
    };

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _dir.Delete(recursive: true);

    [Theory]
    [MemberData(nameof(TransportsAndLevels))]
    public async Task SleepThenWakeGivesTheSameOutputAndRecordsTheWakeTime(TransportKind transport, RoostSleepLevel level)
    {
        var (path, locked) = Checkpoint("model.safetensors", TinyConfig.Hybrid.Seed);
        var executors = new List<TinyCpuExecutor>();
        await using var client = await TestRig.TinyAsync(transport, path, locked, executors);
        Assert.Equal(RoostCapabilityStatus.Untested, client.Capabilities.StatusOf(RoostCapabilities.SleepWake));
        var before = await AskAsync(client);
        Assert.Null(client.LastWakeDuration);

        var slept = await client.SleepAsync(level, Ct);
        Assert.Equal(RoostWorkerOperation.Sleep, slept.Operation);
        Assert.True(slept.IsSleeping);
        Assert.True(client.IsSleeping);
        Assert.True(slept.Bytes > 0);
        Assert.Equal(level == RoostSleepLevel.KvCacheAndWeights, slept.WeightsDigest.IsDefault);
        if (transport == TransportKind.InProcess)
        {
            Assert.False(executors.Single().HasState);
            Assert.Equal(level == RoostSleepLevel.KvCache, executors.Single().Weights!.IsResident);
        }

        var asleep = await Assert.ThrowsAsync<RoostChatException>(() => AskAsync(client));
        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, asleep.Error.Code);
        var twice = await Assert.ThrowsAsync<RoostChatException>(() => client.SleepAsync(level, Ct));
        Assert.Equal(RoostErrorCode.InvalidOption, twice.Error.Code);

        var woken = await client.WakeUpAsync(Ct);
        Assert.Equal(RoostWorkerOperation.WakeUp, woken.Operation);
        Assert.False(woken.IsSleeping);
        Assert.False(client.IsSleeping);
        Assert.Equal(slept.Bytes, woken.Bytes);
        Assert.True(woken.Duration > TimeSpan.Zero);
        Assert.True(woken.WorkerDuration <= woken.Duration);
        Assert.Equal(woken.Duration, client.LastWakeDuration);
        Assert.Equal(Convert.FromHexString(locked.Sha256), woken.WeightsDigest.ToArray());

        Assert.Equal(before, await AskAsync(client));
        var awake = await Assert.ThrowsAsync<RoostChatException>(() => client.WakeUpAsync(Ct));
        Assert.Equal(RoostErrorCode.InvalidOption, awake.Error.Code);
    }

    [Theory]
    [MemberData(nameof(Transports))]
    public async Task AWeightsFileSubstitutedDuringSleepFailsTheWakeAndTheWorkerStaysAsleep(TransportKind transport)
    {
        var (path, locked) = Checkpoint("model.safetensors", TinyConfig.Hybrid.Seed);
        var original = await File.ReadAllBytesAsync(path, Ct);
        await using var client = await TestRig.TinyAsync(transport, path, locked);
        var before = await AskAsync(client);
        _ = await client.SleepAsync(RoostSleepLevel.KvCacheAndWeights, Ct);

        // Same tensors, dtypes and size, other values: only the digest checked as the file is read can tell.
        _ = TinyCheckpoint.Write(path, OtherSeed);
        Assert.Equal(original.Length, new FileInfo(path).Length);
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.WakeUpAsync(Ct));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("SHA-256", e.Error.Message, StringComparison.Ordinal);
        Assert.True(client.IsSleeping);

        await File.WriteAllBytesAsync(path, original, Ct);
        _ = await client.WakeUpAsync(Ct);
        Assert.Equal(before, await AskAsync(client));
    }

    [Theory]
    [MemberData(nameof(Transports))]
    public async Task UpdateWeightsReplacesTheModelInPlaceAndASubstitutedFileKeepsTheOldWeights(TransportKind transport)
    {
        var (path, locked) = Checkpoint("model.safetensors", TinyConfig.Hybrid.Seed);
        var (nextPath, nextLock) = Checkpoint("next.safetensors", OtherSeed);
        await using var client = await TestRig.TinyAsync(transport, path, locked);
        Assert.Equal(RoostCapabilityStatus.Untested, client.Capabilities.StatusOf(RoostCapabilities.WeightUpdate));
        var before = await AskAsync(client);

        var updated = await client.UpdateWeightsAsync(TinyCheckpoint.Source(nextPath, nextLock), Ct);
        Assert.Equal(RoostWorkerOperation.UpdateWeights, updated.Operation);
        Assert.Equal(Convert.FromHexString(nextLock.Sha256), updated.WeightsDigest.ToArray());
        Assert.True(updated.Bytes > 0);
        var after = await AskAsync(client);
        Assert.NotEqual(before, after);

        // A third file whose bytes no longer match the lock the caller gives: refused, and the replacement stays in use.
        var (thirdPath, thirdLock) = Checkpoint("third.safetensors", TinyConfig.Hybrid.Seed);
        _ = TinyCheckpoint.Write(thirdPath, OtherSeed + 1);
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.UpdateWeightsAsync(TinyCheckpoint.Source(thirdPath, thirdLock), Ct));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Equal(after, await AskAsync(client));

        // A level 2 sleep and wake restore the replacement, not the launch weights.
        _ = await client.SleepAsync(RoostSleepLevel.KvCacheAndWeights, Ct);
        var woken = await client.WakeUpAsync(Ct);
        Assert.Equal(Convert.FromHexString(nextLock.Sha256), woken.WeightsDigest.ToArray());
        Assert.Equal(after, await AskAsync(client));

        // Back to the original file in place: the first output again.
        _ = await client.UpdateWeightsAsync(TinyCheckpoint.Source(path, locked), Ct);
        Assert.Equal(before, await AskAsync(client));
    }

    [Fact]
    public async Task AWorkerProcessStartedAfterAnUpdateServesTheUpdatedWeights()
    {
        var (path, locked) = Checkpoint("model.safetensors", TinyConfig.Hybrid.Seed);
        var (nextPath, nextLock) = Checkpoint("next.safetensors", OtherSeed);
        await using var client = await TestRig.TinyAsync(TransportKind.Process, path, locked);
        _ = await client.UpdateWeightsAsync(TinyCheckpoint.Source(nextPath, nextLock), Ct);
        var updated = await AskAsync(client);

        var transport = Assert.IsType<ProcessWorkerTransport>(((DriverTransport)((WorkerDriver)client.Backend).Transport).Inner);
        var first = transport.ProcessId;
        using (var worker = Process.GetProcessById((int)first))
        {
            worker.Kill();
            await worker.WaitForExitAsync(Ct);
        }

        // The request that finds the worker gone fails; the next starts a worker, which loads the launch file and then the update.
        var lost = await Assert.ThrowsAsync<RoostChatException>(() => AskAsync(client));
        Assert.Equal(RoostErrorCode.WorkerRestarted, lost.Error.Code);
        Assert.Equal(updated, await AskAsync(client));
        Assert.NotEqual(first, transport.ProcessId);
    }

    [Fact]
    public async Task UpdateWeightsRefusesARelativePath()
    {
        var (path, locked) = Checkpoint("model.safetensors", TinyConfig.Hybrid.Seed);
        await using var client = await TestRig.TinyAsync(TransportKind.InProcess, path, locked);
        var source = TinyCheckpoint.Source(path, locked) with { Path = "model.safetensors" };
        _ = await Assert.ThrowsAsync<ArgumentException>(() => client.UpdateWeightsAsync(source, Ct));
    }

    [Fact]
    public async Task AWorkerWithoutTheLifecycleRefusesWithProfileUnsupported()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") });
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.SleepAsync(cancellationToken: Ct));
        Assert.Equal(RoostErrorCode.ProfileUnsupported, e.Error.Code);
        Assert.False(client.IsSleeping);
        Assert.Equal("ok", (await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], cancellationToken: Ct)).Text);
    }

    [Fact]
    public async Task SleepIsRefusedWhileARequestRuns()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("abc"), NeverStop = true, DelayMs = 20 });
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        await using var stream = client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "Hi")], new ChatOptions { MaxOutputTokens = 200 }, cts.Token).GetAsyncEnumerator(cts.Token);
        Assert.True(await stream.MoveNextAsync());

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.SleepAsync(cancellationToken: Ct));
        Assert.Equal(RoostErrorCode.InvalidOption, e.Error.Code);
        Assert.Contains("requests running", e.Error.Message, StringComparison.Ordinal);
        await cts.CancelAsync();
    }

    [Fact]
    public void ThePackagedCudaWorkerReportsSleepAndUpdateUnsupported()
    {
        var packaged = new RoostCapabilities(weightLifecycle: false);
        Assert.Equal(RoostCapabilityStatus.Unsupported, packaged.StatusOf(RoostCapabilities.SleepWake));
        Assert.Equal(RoostCapabilityStatus.Unsupported, packaged.StatusOf(RoostCapabilities.WeightUpdate));
    }

    private static async Task<string> AskAsync(RoostChatClient client)
    {
        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hello, tiny model")], new ChatOptions { MaxOutputTokens = 12 }, Ct);
        return string.Join(',', response.Text.Select(static c => ((int)c).ToString(System.Globalization.CultureInfo.InvariantCulture)));
    }

    private (string Path, Starlings.Roost.Worker.Weights.LockedFile Lock) Checkpoint(string name, ulong seed)
    {
        var path = Path.Combine(_dir.FullName, name);
        return (path, TinyCheckpoint.Write(path, seed));
    }
}
