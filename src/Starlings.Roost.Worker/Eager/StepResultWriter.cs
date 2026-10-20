using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Worker.Eager;

/// <summary>Writes a successful C2 step result for the eager worker.</summary>
public static class StepResultWriter
{
    private const int Align = SharedLayout.Alignment;

    /// <summary>The bytes a result for <paramref name="sequences"/> sequences and <paramref name="tokens"/> emitted tokens needs.</summary>
    /// <param name="sequences">Sequences in the plan.</param>
    /// <param name="tokens">Emitted tokens.</param>
    /// <returns>The size.</returns>
    public static int SizeFor(int sequences, int tokens)
    {
        int at = Round(Unsafe.SizeOf<StepResultHeader>());
        at = Round(at + (sequences * Unsafe.SizeOf<ResultSequence>()));
        at = Round(at + (tokens * 4));
        return Round(at + Unsafe.SizeOf<DeviceTimings>());
    }

    /// <summary>Writes the result.</summary>
    /// <param name="result">The result buffer.</param>
    /// <param name="plan">The plan that ran.</param>
    /// <param name="emitted">The token each sampling sequence emitted, in plan order (one per sequence with <see cref="ParsedSequence.Sample"/>).</param>
    /// <param name="timings">Device timings.</param>
    /// <returns>Bytes used.</returns>
    /// <exception cref="ArgumentException">The buffer is too small.</exception>
    public static int Write(Span<byte> result, ParsedPlan plan, ReadOnlySpan<int> emitted, in DeviceTimings timings)
    {
        ArgumentNullException.ThrowIfNull(plan);
        int sequences = plan.Sequences.Length;
        int total = SizeFor(sequences, emitted.Length);
        if (result.Length < total)
        {
            throw new ArgumentException($"the result buffer holds {result.Length} bytes, the result needs {total}", nameof(result));
        }

        result[..total].Clear();
        int seqAt = Round(Unsafe.SizeOf<StepResultHeader>());
        int tokAt = Round(seqAt + (sequences * Unsafe.SizeOf<ResultSequence>()));
        int timeAt = Round(tokAt + (emitted.Length * 4));
        int next = 0;
        for (int i = 0; i < sequences; i++)
        {
            var s = plan.Sequences[i];
            var r = new ResultSequence { SequenceHandle = s.Handle, TokenStart = (uint)next, TokenCount = s.Sample ? 1u : 0u, Finish = FinishReason.None };
            if (s.Sample)
            {
                int token = emitted[next++];
                MemoryMarshal.Write(result.Slice(tokAt + ((next - 1) * 4), 4), token);
                if (s.StopCount > 0 && plan.StopIds.AsSpan(s.StopStart, s.StopCount).Contains(token))
                {
                    r.Finish = FinishReason.Eos;
                }
                else if (s.MaxNewTokensLeft == 1)
                {
                    r.Finish = FinishReason.Length;
                }
            }

            MemoryMarshal.Write(result.Slice(seqAt + (i * Unsafe.SizeOf<ResultSequence>()), Unsafe.SizeOf<ResultSequence>()), r);
        }

        MemoryMarshal.Write(result.Slice(timeAt, Unsafe.SizeOf<DeviceTimings>()), timings);
        var h = new StepResultHeader
        {
            Magic = SharedLayout.MagicResult,
            LayoutVersion = SharedLayout.Version,
            HeaderSize = (uint)Unsafe.SizeOf<StepResultHeader>(),
            Fault = FaultClass.None,
            StepId = plan.StepId,
            TotalBytes = (ulong)total,
            SequenceCount = (uint)sequences,
            TokenCount = (uint)emitted.Length,
            Sequences = new TableRef { Offset = (ulong)seqAt, Count = (uint)sequences, Stride = (uint)Unsafe.SizeOf<ResultSequence>() },
            TokenIds = new TableRef { Offset = (ulong)tokAt, Count = (uint)emitted.Length, Stride = 4 },
            Timings = new TableRef { Offset = (ulong)timeAt, Count = 1, Stride = (uint)Unsafe.SizeOf<DeviceTimings>() },
        };
        MemoryMarshal.Write(result, h);
        return total;
    }

    private static int Round(int n) => (n + Align - 1) / Align * Align;
}
