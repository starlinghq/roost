using System.Collections.Immutable;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost;

/// <summary>What a worker library says about itself before it touches a device (compared with the host's offer).</summary>
/// <param name="AbiVersion">C1 version of the kernel library, <c>(major &lt;&lt; 16) | minor</c>.</param>
/// <param name="ManifestDigest">SHA-256 of the kernel manifest (C5) the library was built from, 32 bytes.</param>
/// <param name="WeightsDigest">SHA-256 of the weights as recorded in the manifest and profile lock, 32 bytes. Read from the lock, never computed at spawn (S8: 9.4 s for 3.44 GB).</param>
public sealed record WorkerIdentity(uint AbiVersion, ImmutableArray<byte> ManifestDigest, ImmutableArray<byte> WeightsDigest);

/// <summary>What one step produced, apart from the result bytes it wrote.</summary>
/// <param name="Fault"><see cref="FaultClass.None"/>, <see cref="FaultClass.Recoverable"/> or <see cref="FaultClass.Sticky"/>; the other classes are the host's conclusions and are refused.</param>
/// <param name="ErrorCode">The CUDA or C1 status behind a fault, 0 otherwise.</param>
/// <param name="ResultBytes">Bytes of the result buffer in use when <paramref name="Fault"/> is none; the executor writes the whole result, header included.</param>
public readonly record struct StepOutcome(FaultClass Fault, int ErrorCode, int ResultBytes)
{
    /// <summary>A successful step.</summary>
    /// <param name="resultBytes">Bytes written to the result buffer.</param>
    /// <returns>The outcome.</returns>
    public static StepOutcome Ok(int resultBytes) => new(FaultClass.None, 0, resultBytes);

    /// <summary>The step failed and the context is usable.</summary>
    /// <param name="errorCode">The status.</param>
    /// <returns>The outcome.</returns>
    public static StepOutcome Recoverable(int errorCode) => new(FaultClass.Recoverable, errorCode, 0);

    /// <summary>The context is lost: the worker publishes the fault and exits.</summary>
    /// <param name="errorCode">The status.</param>
    /// <returns>The outcome.</returns>
    public static StepOutcome Sticky(int errorCode) => new(FaultClass.Sticky, errorCode, 0);
}

/// <summary>
/// What the worker process runs a step with: the device, the segment registration, the weights and the forward pass.
/// The real implementation is the Worker package (stream F, R-02x); tests inject fakes, and the GPU benchmark injects
/// a CUDA spin step. The runtime calls it from one thread, in the order <see cref="Identity"/>, <see cref="OpenDevice"/>,
/// <see cref="RegisterSegment"/>, <see cref="LoadWeights"/>, then <see cref="Execute"/> per plan.
/// </summary>
public interface IStepExecutor : IDisposable
{
    /// <summary>ABI and digests, read without touching the device.</summary>
    WorkerIdentity Identity { get; }

    /// <summary>Creates the context on a device.</summary>
    /// <param name="deviceOrdinal">The CUDA ordinal the host asked for.</param>
    /// <param name="deviceUuid">Receives the 16-byte device UUID.</param>
    /// <returns><see cref="HandshakeReject.None"/> or <see cref="HandshakeReject.Device"/>.</returns>
    HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid);

    /// <summary>Registers the whole shared segment with the device (<c>cudaHostRegister</c>), so plan and result copies read and write ring memory directly.</summary>
    /// <param name="address">The segment's address in this process.</param>
    /// <param name="bytes">The segment's size.</param>
    void RegisterSegment(nint address, long bytes);

    /// <summary>Loads the weights.</summary>
    /// <returns><see cref="HandshakeReject.None"/>, or <see cref="HandshakeReject.WeightsDigest"/> when what was loaded does not match the recorded digest.</returns>
    HandshakeReject LoadWeights();

    /// <summary>
    /// Runs one step. The plan buffer has been validated (magic, sizes, step id). The executor must have finished reading
    /// <paramref name="plan"/>, including any asynchronous copy, when it returns, because the runtime hands the buffer
    /// back to the host then. It should classify a CUDA error with a probe (<c>cudaGetLastError</c>, then a memset and a
    /// sync; a second failure is sticky, S8 condition 2). An exception is treated as sticky.
    /// </summary>
    /// <param name="plan">The plan buffer.</param>
    /// <param name="result">The result buffer.</param>
    /// <returns>The outcome.</returns>
    StepOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result);
}
