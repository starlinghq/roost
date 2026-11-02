using System.Collections.Immutable;
using System.Security.Cryptography;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.WorkerHost.Tests.Support;

/// <summary>
/// A <see cref="FakeExecutor"/> with <see cref="IWeightLifecycle"/> (R-100), for the control-message tests of both transports.
/// It holds no memory: a sleep reports fixed byte counts, a wake checks the expected digest against the digest it records, and
/// a replacement hashes the named file and refuses it unless it matches the lock in the message.
/// </summary>
internal sealed class LifecycleFakeExecutor : IStepExecutor, IWeightLifecycle
{
    public const long KvBytes = 1000;
    public const long WeightBytes = 2000;

    private readonly FakeExecutor _inner = new();
    private ImmutableArray<byte> _digest = FakeExecutor.WeightsDigest;
    private bool _weightsReleased;

    public WorkerIdentity Identity => _inner.Identity;

    public ImmutableArray<byte> ResidentWeightsDigest => _weightsReleased ? default : _digest;

    public HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid) => _inner.OpenDevice(deviceOrdinal, deviceUuid);

    public void RegisterSegment(nint address, long bytes) => _inner.RegisterSegment(address, bytes);

    public HandshakeReject LoadWeights() => _inner.LoadWeights();

    public StepOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result) => _inner.Execute(plan, result);

    public ControlOutcome Sleep(SleepLevel level)
    {
        _weightsReleased = level == SleepLevel.Weights;
        return ControlOutcome.Ok(KvBytes + (_weightsReleased ? WeightBytes : 0));
    }

    public ControlOutcome Wake(ReadOnlySpan<byte> expectedDigest)
    {
        if (!expectedDigest.SequenceEqual(_digest.AsSpan()))
        {
            return ControlOutcome.Fail(ControlStatus.WeightsDigest, "the weights do not have the expected digest");
        }

        var bytes = KvBytes + (_weightsReleased ? WeightBytes : 0);
        _weightsReleased = false;
        return ControlOutcome.Ok(bytes);
    }

    public ControlOutcome ReplaceWeights(WeightSource source)
    {
        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(source.Path);
        }
        catch (IOException e)
        {
            return ControlOutcome.Fail(ControlStatus.WeightsDigest, e.Message);
        }

        if (bytes.LongLength != source.Size || !SHA256.HashData(bytes).AsSpan().SequenceEqual(source.Sha256.AsSpan()))
        {
            return ControlOutcome.Fail(ControlStatus.WeightsDigest, "the file does not match its lock");
        }

        _digest = source.Sha256;
        return ControlOutcome.Ok(_weightsReleased ? 0 : bytes.LongLength);
    }

    public void Dispose() => _inner.Dispose();
}
