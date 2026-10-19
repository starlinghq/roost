using System.Runtime.CompilerServices;
using Starlings.Roost.Plans;

namespace Starlings.Roost.WorkerHost.Tests.Support;

/// <summary>Builds the plans the fake executor understands.</summary>
internal static unsafe class TestPlans
{
    public static int TotalBytes => sizeof(StepPlanHeader) + FakeExecutor.PayloadBytes;

    public static void Fill(Span<byte> buffer, ulong stepId, FakeOp op = FakeOp.Normal, uint argument = 0)
    {
        var header = new StepPlanHeader
        {
            Magic = 0x4C505352u,
            LayoutVersion = 1,
            HeaderSize = (uint)sizeof(StepPlanHeader),
            StepId = stepId,
            TotalBytes = (ulong)TotalBytes,
            ShapeBucket = (uint)op,
            SequenceCount = argument,
        };
        Unsafe.WriteUnaligned(ref buffer[0], header);
        var payload = buffer.Slice(sizeof(StepPlanHeader), FakeExecutor.PayloadBytes);
        var x = stepId * 0x9E3779B97F4A7C15UL + 1;
        for (var i = 0; i < payload.Length; i++)
        {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            payload[i] = (byte)x;
        }
    }

    public static ulong ExpectedChecksum(ulong stepId)
    {
        Span<byte> scratch = stackalloc byte[sizeof(StepPlanHeader) + FakeExecutor.PayloadBytes];
        Fill(scratch, stepId);
        return FakeExecutor.Checksum(scratch.Slice(sizeof(StepPlanHeader), FakeExecutor.PayloadBytes));
    }

    public static bool EchoMatches(ReadOnlySpan<byte> result, ulong stepId)
    {
        Span<byte> scratch = stackalloc byte[sizeof(StepPlanHeader) + FakeExecutor.PayloadBytes];
        Fill(scratch, stepId);
        return result.Slice(sizeof(StepResultHeader), FakeExecutor.EchoBytes)
            .SequenceEqual(scratch.Slice(sizeof(StepPlanHeader), FakeExecutor.EchoBytes));
    }
}
