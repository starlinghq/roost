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
    public static int SizeFor(int sequences, int tokens) => SizeFor(sequences, tokens, 0);

    /// <summary>The bytes a result with <paramref name="logprobRows"/> logprob rows needs.</summary>
    /// <param name="sequences">Sequences in the plan.</param>
    /// <param name="tokens">Emitted tokens.</param>
    /// <param name="logprobRows">Logprob rows: (1 + top-n) per emitted token of a row that asked for logprobs.</param>
    /// <returns>The size.</returns>
    public static int SizeFor(int sequences, int tokens, int logprobRows)
    {
        int at = Round(Unsafe.SizeOf<StepResultHeader>());
        at = Round(at + (sequences * Unsafe.SizeOf<ResultSequence>()));
        at = Round(at + (tokens * 4));
        at = Round(at + (logprobRows * Unsafe.SizeOf<ResultLogprob>()));
        return Round(at + Unsafe.SizeOf<DeviceTimings>());
    }

    /// <summary>Writes the result.</summary>
    /// <param name="result">The result buffer.</param>
    /// <param name="plan">The plan that ran.</param>
    /// <param name="emitted">The token each sampling sequence emitted, in plan order (one per sequence with <see cref="ParsedSequence.Sample"/>).</param>
    /// <param name="timings">Device timings.</param>
    /// <returns>Bytes used.</returns>
    /// <exception cref="ArgumentException">The buffer is too small.</exception>
    public static int Write(Span<byte> result, ParsedPlan plan, ReadOnlySpan<int> emitted, in DeviceTimings timings) =>
        Write(result, plan, emitted, null, timings);

    /// <summary>Writes the result with the logprob rows of the CPU sampler.</summary>
    /// <param name="result">The result buffer.</param>
    /// <param name="plan">The plan that ran.</param>
    /// <param name="emitted">The token each sampling sequence emitted, in plan order.</param>
    /// <param name="logprobs">
    /// Per sampling sequence, in the same order, its (1 + top-n) rows with the chosen token first, or null for a row that
    /// asked for none; null for no logprobs at all.
    /// </param>
    /// <param name="timings">Device timings.</param>
    /// <returns>Bytes used.</returns>
    /// <exception cref="ArgumentException">The buffer is too small, or <paramref name="logprobs"/> does not have one entry per emitted token.</exception>
    public static int Write(Span<byte> result, ParsedPlan plan, ReadOnlySpan<int> emitted, IReadOnlyList<ResultLogprob[]?>? logprobs, in DeviceTimings timings)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (logprobs is not null && logprobs.Count != emitted.Length)
        {
            throw new ArgumentException($"{logprobs.Count} logprob entries for {emitted.Length} emitted tokens", nameof(logprobs));
        }

        int rows = 0;
        for (int i = 0; logprobs is not null && i < logprobs.Count; i++)
        {
            rows += logprobs[i]?.Length ?? 0;
        }

        int sequences = plan.Sequences.Length;
        int total = SizeFor(sequences, emitted.Length, rows);
        if (result.Length < total)
        {
            throw new ArgumentException($"the result buffer holds {result.Length} bytes, the result needs {total}", nameof(result));
        }

        result[..total].Clear();
        int seqAt = Round(Unsafe.SizeOf<StepResultHeader>());
        int tokAt = Round(seqAt + (sequences * Unsafe.SizeOf<ResultSequence>()));
        int lpAt = Round(tokAt + (emitted.Length * 4));
        int timeAt = Round(lpAt + (rows * Unsafe.SizeOf<ResultLogprob>()));
        int next = 0;
        int nextRow = 0;
        for (int i = 0; i < sequences; i++)
        {
            var s = plan.Sequences[i];
            var r = new ResultSequence { SequenceHandle = s.Handle, TokenStart = (uint)next, TokenCount = s.Sample ? 1u : 0u, Finish = FinishReason.None };
            if (s.Sample)
            {
                int token = emitted[next++];
                MemoryMarshal.Write(result.Slice(tokAt + ((next - 1) * 4), 4), token);
                if (logprobs?[next - 1] is { Length: > 0 } lp)
                {
                    r.LogprobStart = (uint)nextRow;
                    r.LogprobCount = (uint)lp.Length;
                    MemoryMarshal.AsBytes(lp.AsSpan()).CopyTo(result[(lpAt + (nextRow * Unsafe.SizeOf<ResultLogprob>()))..]);
                    nextRow += lp.Length;
                }

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
            Logprobs = new TableRef { Offset = (ulong)lpAt, Count = (uint)rows, Stride = (uint)Unsafe.SizeOf<ResultLogprob>() },
            Timings = new TableRef { Offset = (ulong)timeAt, Count = 1, Stride = (uint)Unsafe.SizeOf<DeviceTimings>() },
        };
        MemoryMarshal.Write(result, h);
        return total;
    }

    private static int Round(int n) => (n + Align - 1) / Align * Align;
}
