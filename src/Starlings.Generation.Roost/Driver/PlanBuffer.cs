using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Engine.Scheduling;
using Starlings.Roost.Plans;

namespace Starlings.Generation.Roost.Driver;

/// <summary>
/// Lays a <see cref="PlannedStep"/> out as a C2 step plan in a shared plan buffer (<c>kernels/include/roost_worker_shm.h</c>):
/// the header, then each table at a 64-byte aligned offset, records packed at their own size. The engine has no shared-memory
/// writer yet (stream G); this one is internal to the M0 driver and is replaced with the engine's.
/// </summary>
internal static class PlanBuffer
{
    private const int Align = SharedLayout.Alignment;

    /// <summary>An upper bound on the bytes of any plan within the limits.</summary>
    /// <param name="maxSequences">Rows per step.</param>
    /// <param name="maxTokens">Tokens per step.</param>
    /// <param name="groups">State groups.</param>
    /// <param name="maxBlocksPerTable">Block ids in one row's table of one group.</param>
    /// <param name="stopIdsPerRow">Stop ids per row.</param>
    /// <returns>The size.</returns>
    public static long SizeFor(int maxSequences, int maxTokens, int groups, int maxBlocksPerTable, int stopIdsPerRow)
    {
        long at = Round(Unsafe.SizeOf<StepPlanHeader>());
        at = Round(at + ((long)maxSequences * Unsafe.SizeOf<PlanSequence>()));
        at = Round(at + ((long)maxTokens * 4));
        at = Round(at + ((long)maxTokens * 4));
        at = Round(at + ((long)maxSequences * groups * Unsafe.SizeOf<PlanGroupBlocks>()));
        at = Round(at + ((long)maxSequences * groups * maxBlocksPerTable * 4));
        at = Round(at + ((long)maxSequences * Unsafe.SizeOf<PlanSampling>()));
        at = Round(at + ((long)maxSequences * groups * 2 * Unsafe.SizeOf<PlanCacheOp>()));
        at = Round(at + ((long)maxSequences * stopIdsPerRow * 4));
        return at;
    }

    /// <summary>Writes a plan.</summary>
    /// <param name="buffer">The plan buffer.</param>
    /// <param name="step">The plan.</param>
    /// <returns>Bytes written.</returns>
    /// <exception cref="InvalidOperationException">The buffer is too small for the plan.</exception>
    public static int Write(Span<byte> buffer, PlannedStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        int at = Round(Unsafe.SizeOf<StepPlanHeader>());
        var header = new StepPlanHeader
        {
            Magic = SharedLayout.MagicPlan,
            LayoutVersion = SharedLayout.Version,
            HeaderSize = (uint)Unsafe.SizeOf<StepPlanHeader>(),
            Flags = step.Flags,
            StepId = step.StepId,
            MathMode = step.MathMode,
            SequenceCount = (uint)step.Sequences.Length,
            TokenCount = (uint)step.TokenIds.Length,
            GroupCount = (uint)step.GroupCount,
            Sequences = Place<PlanSequence>(ref at, step.Sequences.Length),
            TokenIds = Place<int>(ref at, step.TokenIds.Length),
            Positions = Place<int>(ref at, step.Positions.Length),
            GroupBlocks = Place<PlanGroupBlocks>(ref at, step.GroupBlocks.Length),
            BlockIds = Place<int>(ref at, step.BlockIds.Length),
            Sampling = Place<PlanSampling>(ref at, step.Sampling.Length),
            CacheOps = Place<PlanCacheOp>(ref at, step.CacheOps.Length),
            StopTokenIds = Place<int>(ref at, step.StopTokenIds.Length),
        };
        if (at > buffer.Length)
        {
            throw new InvalidOperationException($"Plan {step.StepId} needs {at} bytes; the plan buffer holds {buffer.Length}.");
        }

        header.TotalBytes = (ulong)at;
        buffer[..at].Clear();
        MemoryMarshal.Write(buffer, header);
        Copy(buffer, header.Sequences, step.Sequences);
        Copy(buffer, header.TokenIds, step.TokenIds);
        Copy(buffer, header.Positions, step.Positions);
        Copy(buffer, header.GroupBlocks, step.GroupBlocks);
        Copy(buffer, header.BlockIds, step.BlockIds);
        Copy(buffer, header.Sampling, step.Sampling);
        Copy(buffer, header.CacheOps, step.CacheOps);
        Copy(buffer, header.StopTokenIds, step.StopTokenIds);
        return at;
    }

    /// <summary>Reads a successful step result into the planner's form.</summary>
    /// <param name="result">The result buffer.</param>
    /// <param name="expectedStep">The step id the slot named.</param>
    /// <returns>The outcome.</returns>
    /// <exception cref="InvalidDataException">The result is malformed or answers another step.</exception>
    public static StepOutcome ReadResult(ReadOnlySpan<byte> result, ulong expectedStep)
    {
        var h = ReadHeader(result);
        if (h.StepId != expectedStep)
        {
            throw new InvalidDataException($"The result answers step {h.StepId}; step {expectedStep} was expected.");
        }

        var sequences = Table<ResultSequence>(result, h.Sequences, h.SequenceCount);
        var tokens = Table<int>(result, h.TokenIds, h.TokenCount);
        foreach (var s in sequences)
        {
            if ((ulong)s.TokenStart + s.TokenCount > (ulong)tokens.Length)
            {
                throw new InvalidDataException("A result row names tokens outside the token table.");
            }
        }

        return StepOutcome.FromTables(h.StepId, sequences, tokens);
    }

    /// <summary>Reads and checks a result header.</summary>
    /// <param name="result">The result buffer.</param>
    /// <returns>The header.</returns>
    /// <exception cref="InvalidDataException">The header is malformed.</exception>
    public static StepResultHeader ReadHeader(ReadOnlySpan<byte> result)
    {
        if (result.Length < Unsafe.SizeOf<StepResultHeader>())
        {
            throw new InvalidDataException("The result buffer is shorter than a result header.");
        }

        var h = MemoryMarshal.Read<StepResultHeader>(result);
        if (h.Magic != SharedLayout.MagicResult || h.LayoutVersion != SharedLayout.Version || h.TotalBytes > (ulong)result.Length)
        {
            throw new InvalidDataException("The result has a bad magic, layout version or size.");
        }

        return h;
    }

    private static T[] Table<T>(ReadOnlySpan<byte> buffer, in TableRef table, uint expected)
        where T : unmanaged
    {
        if (table.Count != expected)
        {
            throw new InvalidDataException($"A result table has {table.Count} entries; the header says {expected}.");
        }

        if (table.Count == 0)
        {
            return [];
        }

        int size = Unsafe.SizeOf<T>();
        ulong bytes = (ulong)table.Count * (ulong)size;
        if (table.Stride != size || table.Offset + bytes > (ulong)buffer.Length)
        {
            throw new InvalidDataException("A result table has a bad stride or lies outside the buffer.");
        }

        var copy = new T[table.Count];
        buffer.Slice((int)table.Offset, (int)bytes).CopyTo(MemoryMarshal.AsBytes(copy.AsSpan()));
        return copy;
    }

    private static TableRef Place<T>(ref int at, int count)
        where T : unmanaged
    {
        var table = new TableRef { Offset = (ulong)at, Count = (uint)count, Stride = (uint)Unsafe.SizeOf<T>() };
        at = Round(at + (count * Unsafe.SizeOf<T>()));
        return table;
    }

    private static void Copy<T>(Span<byte> buffer, TableRef table, ImmutableArray<T> items)
        where T : unmanaged
    {
        if (!items.IsDefaultOrEmpty)
        {
            MemoryMarshal.AsBytes(items.AsSpan()).CopyTo(buffer[(int)table.Offset..]);
        }
    }

    private static int Round(int n) => (n + Align - 1) / Align * Align;

    private static long Round(long n) => (n + Align - 1) / Align * Align;
}
