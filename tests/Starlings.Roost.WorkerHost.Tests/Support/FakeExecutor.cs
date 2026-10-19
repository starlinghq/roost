using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using Starlings.Roost.Abi;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.WorkerHost.Tests.Support;

/// <summary>What a test plan asks the fake executor to do; carried in the plan header's <c>ShapeBucket</c>.</summary>
internal enum FakeOp : uint
{
    Normal = 0,
    Recoverable = 1,
    Sticky = 2,
    Hang = 3,
    Crash = 4,
    Throw = 5,

    /// <summary>Sleeps <c>SequenceCount</c> milliseconds, then succeeds.</summary>
    Slow = 6,

    /// <summary>Dies with a real fatal signal (<c>Environment.FailFast</c>: SIGABRT on Linux). Only the GITHUB_ACTIONS-gated test publishes it.</summary>
    CrashSignal = 7,
}

/// <summary>
/// The executor the tests inject into both transports. It does no GPU work: it checks the plan payload, writes a result
/// whose payload and checksum depend on the plan, and misbehaves on request, so a test can tell data that went through
/// the rings intact from data that did not.
/// </summary>
internal sealed unsafe class FakeExecutor : IStepExecutor
{
    public const int PayloadBytes = 256;
    public const int EchoBytes = 64;
    public const int ErrorRecoverable = 11;
    public const int ErrorSticky = 700;

    /// <summary>The exit code of an injected crash: the process ends at once, with no cleanup and no signal, so no core dump is written.</summary>
    public const int CrashExitCode = 97;

    public static readonly ImmutableArray<byte> ManifestDigest = Digest("roost-test-manifest");
    public static readonly ImmutableArray<byte> WeightsDigest = Digest("roost-test-weights");

    private readonly ManualResetEventSlim? _hangLatch;
    private readonly int _openDelayMs;
    private readonly bool _failOpen;

    public FakeExecutor(ManualResetEventSlim? hangLatch = null, int openDelayMs = 0, bool failOpen = false)
    {
        _hangLatch = hangLatch;
        _openDelayMs = openDelayMs;
        _failOpen = failOpen;
    }

    public WorkerIdentity Identity => new(KernelAbi.Version, ManifestDigest, WeightsDigest);

    public static ImmutableArray<byte> Digest(string text) => [.. SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(text))];

    public static byte[] UuidFor(int ordinal)
    {
        var uuid = new byte[16];
        for (var i = 0; i < uuid.Length; i++)
        {
            uuid[i] = (byte)(0xA0 + ordinal + i);
        }

        return uuid;
    }

    public HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid)
    {
        if (_failOpen)
        {
            throw new InvalidOperationException("injected: OpenDevice threw");
        }

        if (_openDelayMs > 0)
        {
            Thread.Sleep(_openDelayMs);
        }

        if (deviceOrdinal != 0)
        {
            return HandshakeReject.Device;
        }

        UuidFor(deviceOrdinal).CopyTo(deviceUuid);
        return HandshakeReject.None;
    }

    public void RegisterSegment(nint address, long bytes)
    {
    }

    public HandshakeReject LoadWeights() => HandshakeReject.None;

    public StepOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result)
    {
        var header = Unsafe.ReadUnaligned<StepPlanHeader>(ref Unsafe.AsRef(in plan[0]));
        var payload = plan.Slice(sizeof(StepPlanHeader), PayloadBytes);
        switch ((FakeOp)header.ShapeBucket)
        {
            case FakeOp.Recoverable:
                return StepOutcome.Recoverable(ErrorRecoverable);
            case FakeOp.Sticky:
                return StepOutcome.Sticky(ErrorSticky);
            case FakeOp.Hang:
                if (_hangLatch is null)
                {
                    Thread.Sleep(Timeout.Infinite);
                }
                else
                {
                    _hangLatch.Wait();
                }

                return StepOutcome.Sticky(ErrorSticky);
            case FakeOp.Crash:
                // A crash as the host sees it (the process is gone without a result) but without SIGABRT, which
                // systemd-coredump records and a desktop reports. Skips managed shutdown the way a crash does.
                if (OperatingSystem.IsWindows())
                {
                    Win32.ExitNow(CrashExitCode);
                }
                else
                {
                    Libc.Exit(CrashExitCode);
                }

                return default;
            case FakeOp.CrashSignal:
                Environment.FailFast("injected crash");
                return default;
            case FakeOp.Throw:
                throw new InvalidOperationException("injected: Execute threw");
            case FakeOp.Slow:
                Thread.Sleep((int)header.SequenceCount);
                break;
            default:
                break;
        }

        var checksum = Checksum(payload);
        var resultHeader = new StepResultHeader
        {
            Magic = 0x53525352u,
            LayoutVersion = 1,
            HeaderSize = (uint)sizeof(StepResultHeader),
            Fault = FaultClass.None,
            StepId = header.StepId,
            TotalBytes = (ulong)(sizeof(StepResultHeader) + EchoBytes),
            TokenCount = (uint)header.StepId,
            Reserved1 = checksum,
        };
        Unsafe.WriteUnaligned(ref result[0], resultHeader);
        payload[..EchoBytes].CopyTo(result[sizeof(StepResultHeader)..]);
        return StepOutcome.Ok(sizeof(StepResultHeader) + EchoBytes);
    }

    public void Dispose()
    {
    }

    public static ulong Checksum(ReadOnlySpan<byte> payload)
    {
        ulong sum = 14695981039346656037UL;
        foreach (var b in payload)
        {
            sum = (sum ^ b) * 1099511628211UL;
        }

        return sum;
    }
}
