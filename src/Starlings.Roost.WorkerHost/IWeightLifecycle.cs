using System.Collections.Immutable;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// A weights file named by its lock (R-024): the path the worker opens, the size and SHA-256 the lock records. A worker loads
/// it only through a verified stream, so a substituted file fails before any forward reads it.
/// </summary>
/// <param name="Path">The safetensors file, absolute, as the worker process sees it.</param>
/// <param name="Size">Its locked size in bytes.</param>
/// <param name="Sha256">Its locked SHA-256, 32 bytes.</param>
public sealed record WeightSource(string Path, long Size, ImmutableArray<byte> Sha256)
{
    /// <summary>Checks the fields: an absolute path, a non-negative size and a 32-byte digest.</summary>
    /// <exception cref="ArgumentException">A field is invalid.</exception>
    public void Validate()
    {
        if (string.IsNullOrEmpty(Path) || !System.IO.Path.IsPathFullyQualified(Path))
        {
            throw new ArgumentException("The weights path must be absolute.", nameof(Path));
        }

        if (Size < 0)
        {
            throw new ArgumentException("The weights size must not be negative.", nameof(Size));
        }

        if (Sha256.IsDefault || Sha256.Length != 32)
        {
            throw new ArgumentException("The weights digest is SHA-256: 32 bytes.", nameof(Sha256));
        }
    }
}

/// <summary>What an <see cref="IWeightLifecycle"/> call did.</summary>
/// <param name="Status">The outcome; anything but <see cref="ControlStatus.Ok"/> left the executor as it was, as far as it can tell.</param>
/// <param name="Bytes">Sleep: bytes released. Wake and replace: bytes allocated.</param>
/// <param name="ErrorCode">The CUDA, C1 or worker status behind a failure, 0 otherwise.</param>
/// <param name="Message">Why it failed; never contains prompt text. Null on success.</param>
public readonly record struct ControlOutcome(ControlStatus Status, long Bytes, int ErrorCode = 0, string? Message = null)
{
    /// <summary>Done.</summary>
    /// <param name="bytes">Bytes released or allocated.</param>
    /// <returns>The outcome.</returns>
    public static ControlOutcome Ok(long bytes) => new(ControlStatus.Ok, bytes);

    /// <summary>Failed.</summary>
    /// <param name="status">Why, in contract terms.</param>
    /// <param name="message">Why, in words.</param>
    /// <param name="errorCode">The status code behind it, if any.</param>
    /// <returns>The outcome.</returns>
    public static ControlOutcome Fail(ControlStatus status, string message, int errorCode = 0) => new(status, 0, errorCode, message);
}

/// <summary>
/// Sleep, wake and in-place weight replacement (R-100, the C8 control messages of layout 3), implemented by a step executor
/// that can release and restore its memory. An <see cref="IStepExecutor"/> that does not implement it answers every control
/// message with <see cref="ControlStatus.Unsupported"/>. The runtime calls it on the step thread, never during a step, and
/// keeps the sleeping state itself: <see cref="Sleep"/> only while awake, <see cref="Wake"/> only while asleep,
/// <see cref="ReplaceWeights"/> in either state.
/// </summary>
public interface IWeightLifecycle
{
    /// <summary>The verified digest of the resident weights, 32 bytes; default while they are released.</summary>
    ImmutableArray<byte> ResidentWeightsDigest { get; }

    /// <summary>
    /// Releases the KV and recurrent state pools, and the weights at <see cref="SleepLevel.Weights"/>. Their contents are lost:
    /// the host sends a sleep only with no sequence live.
    /// </summary>
    /// <param name="level">What to release.</param>
    /// <returns>The outcome, with the bytes released.</returns>
    ControlOutcome Sleep(SleepLevel level);

    /// <summary>
    /// Restores what <see cref="Sleep"/> released: state pools allocated again (zeroed), and released weights reloaded from
    /// their locked file through the verified stream. Fails with <see cref="ControlStatus.WeightsDigest"/>, staying asleep, when
    /// the weights the worker would hold do not have <paramref name="expectedDigest"/> or the file does not match its lock.
    /// </summary>
    /// <param name="expectedDigest">The SHA-256 the resident weights must have afterwards, 32 bytes.</param>
    /// <returns>The outcome, with the bytes allocated.</returns>
    ControlOutcome Wake(ReadOnlySpan<byte> expectedDigest);

    /// <summary>
    /// Loads weights from <paramref name="source"/> through the verified stream and, only once the digest has matched, swaps
    /// them in for the resident ones (or makes them the ones a wake restores, while asleep). A file that does not match its
    /// lock fails with <see cref="ControlStatus.WeightsDigest"/> and the old weights stay.
    /// </summary>
    /// <param name="source">The new weights.</param>
    /// <returns>The outcome, with the bytes allocated.</returns>
    ControlOutcome ReplaceWeights(WeightSource source);
}
