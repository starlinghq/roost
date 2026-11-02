using System.Collections.Immutable;
using System.Security.Cryptography;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;
using Starlings.Roost.WorkerHost.Tests.Support;

namespace Starlings.Roost.WorkerHost.Tests;

/// <summary>
/// The C8 control messages of layout 3 (R-100) over both transports: sleep, wake and replace weights reach the executor's
/// <see cref="IWeightLifecycle"/>, the runtime keeps the sleeping state and refuses steps while asleep, and the answers carry
/// status, bytes, state, digest and the worker's duration.
/// </summary>
public abstract class ControlMessageSuite
{
    internal abstract Rig NewRig(bool lifecycle);

    [Fact]
    public async Task SleepRefusesStepsUntilAWakeThatChecksTheDigest()
    {
        await using var rig = NewRig(lifecycle: true);
        _ = await rig.StartAsync();
        _ = rig.Publish();
        Assert.Equal(FaultClass.None, rig.Wait(TimeSpan.FromSeconds(10)).Fault);

        var slept = rig.Control(RingSlotKind.Sleep);
        Assert.Equal(ControlStatus.Ok, slept.Status);
        Assert.Equal(LifecycleFakeExecutor.KvBytes, slept.Bytes);
        Assert.Equal(WorkerState.Sleeping, slept.WorkerState);
        Assert.Equal(FakeExecutor.WeightsDigest, slept.WeightsDigest);
        await WaitForHeartbeatStateAsync(rig, WorkerState.Sleeping);

        _ = rig.Publish();
        var refused = rig.Wait(TimeSpan.FromSeconds(10));
        Assert.Equal(FaultClass.Recoverable, refused.Fault);
        Assert.Equal(WorkerRuntime.ErrorAsleep, refused.Header.ErrorCode);

        Assert.Equal(ControlStatus.RefusedState, rig.Control(RingSlotKind.Sleep).Status);

        var wrong = FakeExecutor.Digest("other weights");
        var mismatch = rig.Control(RingSlotKind.Wake, (b, id) => ControlMessages.WriteWake(b.Span, id, wrong.AsSpan()));
        Assert.Equal(ControlStatus.WeightsDigest, mismatch.Status);
        Assert.Equal(WorkerState.Sleeping, mismatch.WorkerState);
        Assert.NotEmpty(mismatch.ErrorText);

        var woken = rig.Control(RingSlotKind.Wake);
        Assert.Equal(ControlStatus.Ok, woken.Status);
        Assert.Equal(WorkerState.Ready, woken.WorkerState);
        Assert.Equal(LifecycleFakeExecutor.KvBytes, woken.Bytes);
        Assert.True(woken.WorkerDuration >= TimeSpan.Zero);

        _ = rig.Publish();
        var ok = rig.Wait(TimeSpan.FromSeconds(10));
        Assert.Equal(FaultClass.None, ok.Fault);
        Assert.True(ok.EchoOk && ok.ChecksumOk);
        Assert.Equal(ControlStatus.RefusedState, rig.Control(RingSlotKind.Wake).Status);
        await WaitForHeartbeatStateAsync(rig, WorkerState.Ready);
    }

    [Fact]
    public async Task ALevelTwoSleepReleasesTheWeightsUntilTheWake()
    {
        await using var rig = NewRig(lifecycle: true);
        _ = await rig.StartAsync();

        var slept = rig.Control(RingSlotKind.Sleep, (b, id) => ControlMessages.WriteSleep(b.Span, id, SleepLevel.Weights));
        Assert.Equal(ControlStatus.Ok, slept.Status);
        Assert.Equal(LifecycleFakeExecutor.KvBytes + LifecycleFakeExecutor.WeightBytes, slept.Bytes);
        Assert.True(slept.WeightsDigest.IsDefault);

        var woken = rig.Control(RingSlotKind.Wake);
        Assert.Equal(ControlStatus.Ok, woken.Status);
        Assert.Equal(LifecycleFakeExecutor.KvBytes + LifecycleFakeExecutor.WeightBytes, woken.Bytes);
        Assert.Equal(FakeExecutor.WeightsDigest, woken.WeightsDigest);
    }

    [Fact]
    public async Task AReplacementIsVerifiedAgainstItsLockAndASubstitutedFileKeepsTheOldWeights()
    {
        var directory = Path.Combine(Path.GetTempPath(), "roost-r100-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "weights.bin");
            var bytes = RandomNumberGenerator.GetBytes(4096);
            await File.WriteAllBytesAsync(path, bytes, TestContext.Current.CancellationToken);
            var source = new WeightSource(path, bytes.Length, [.. SHA256.HashData(bytes)]);

            await using var rig = NewRig(lifecycle: true);
            _ = await rig.StartAsync();
            var replaced = rig.Control(RingSlotKind.ReplaceWeights, (b, id) => ControlMessages.WriteReplaceWeights(b.Span, id, source));
            Assert.Equal(ControlStatus.Ok, replaced.Status);
            Assert.Equal(source.Sha256, replaced.WeightsDigest);
            Assert.Equal(bytes.Length, replaced.Bytes);

            // The same size, other bytes: only the digest can tell, and the worker keeps what it had.
            bytes[100] ^= 0xFF;
            await File.WriteAllBytesAsync(path, bytes, TestContext.Current.CancellationToken);
            var substituted = rig.Control(RingSlotKind.ReplaceWeights, (b, id) => ControlMessages.WriteReplaceWeights(b.Span, id, source));
            Assert.Equal(ControlStatus.WeightsDigest, substituted.Status);
            Assert.Equal(source.Sha256, substituted.WeightsDigest);

            // A wake checks the digest of the weights the host last verified: the replacement's, not the launch's.
            Assert.Equal(ControlStatus.Ok, rig.Control(RingSlotKind.Sleep).Status);
            Assert.Equal(ControlStatus.WeightsDigest, rig.Control(RingSlotKind.Wake).Status);
            var woken = rig.Control(RingSlotKind.Wake, (b, id) => ControlMessages.WriteWake(b.Span, id, source.Sha256.AsSpan()));
            Assert.Equal(ControlStatus.Ok, woken.Status);
            _ = rig.Publish();
            Assert.Equal(FaultClass.None, rig.Wait(TimeSpan.FromSeconds(10)).Fault);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task AnExecutorWithoutTheLifecycleAnswersUnsupportedAndKeepsStepping()
    {
        await using var rig = NewRig(lifecycle: false);
        _ = await rig.StartAsync();

        var reply = rig.Control(RingSlotKind.Sleep);
        Assert.Equal(ControlStatus.Unsupported, reply.Status);
        Assert.Equal(WorkerState.Ready, reply.WorkerState);
        Assert.Contains("does not implement", reply.ErrorText, StringComparison.Ordinal);

        _ = rig.Publish();
        Assert.Equal(FaultClass.None, rig.Wait(TimeSpan.FromSeconds(10)).Fault);
    }

    [Fact]
    public async Task AMalformedControlMessageIsAnsweredAndTheWorkerKeepsStepping()
    {
        await using var rig = NewRig(lifecycle: true);
        _ = await rig.StartAsync();

        // A sleep message published under a wake slot: the kinds disagree.
        var reply = rig.Control(RingSlotKind.Wake, (b, id) => ControlMessages.WriteSleep(b.Span, id, SleepLevel.Kv));
        Assert.Equal(ControlStatus.Malformed, reply.Status);
        Assert.Equal(WorkerState.Ready, reply.WorkerState);

        _ = rig.Publish();
        Assert.Equal(FaultClass.None, rig.Wait(TimeSpan.FromSeconds(10)).Fault);
    }

    private static async Task WaitForHeartbeatStateAsync(Rig rig, WorkerState state)
    {
        for (var i = 0; i < 500 && rig.Transport.ReadHeartbeat().State != state; i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.Equal(state, rig.Transport.ReadHeartbeat().State);
    }
}

/// <summary>The control-message suite over the worker on a thread of the test process.</summary>
public sealed class InProcessControlMessageTests : ControlMessageSuite
{
    internal override Rig NewRig(bool lifecycle) =>
        new(new InProcessWorkerTransport(() => lifecycle ? new LifecycleFakeExecutor() : new FakeExecutor(), TestSegments.Options()));
}

/// <summary>The control-message suite over a real worker process.</summary>
public sealed class ProcessControlMessageTests : ControlMessageSuite
{
    internal override Rig NewRig(bool lifecycle) =>
        new(new ProcessWorkerTransport(WorkerProcessEntry.Command(lifecycle: lifecycle), TestSegments.Options()));
}

/// <summary>The control-message codec alone.</summary>
public sealed class ControlCodecTests
{
    [Fact]
    public void TheCodecRoundTripsAReplacementAndRefusesARelativePath()
    {
        var digest = ImmutableArray.Create(SHA256.HashData("w"u8));
        var buffer = new byte[8192];
        var source = new WeightSource(Path.Combine(Path.GetTempPath(), "über", "model.safetensors"), 1234, digest);
        _ = ControlMessages.WriteReplaceWeights(buffer, 77, source);

        var read = ControlMessages.Read(buffer, new RingSlot { StepId = 77, Kind = RingSlotKind.ReplaceWeights });
        Assert.Equal(source.Path, read.Source!.Path);
        Assert.Equal(source.Size, read.Source.Size);
        Assert.Equal(digest, read.Source.Sha256);
        Assert.Equal(digest, read.WeightsDigest);
        Assert.Throws<InvalidDataException>(() => ControlMessages.Read(buffer, new RingSlot { StepId = 78, Kind = RingSlotKind.ReplaceWeights }));
        Assert.Throws<ArgumentException>(() => ControlMessages.WriteReplaceWeights(buffer, 1, source with { Path = "relative/model.safetensors" }));
    }
}
