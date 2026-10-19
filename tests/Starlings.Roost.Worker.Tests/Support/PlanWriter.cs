using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Plans;

namespace Starlings.Roost.Worker.Tests.Support;

/// <summary>One sequence of a test plan.</summary>
/// <param name="Handle">Sequence handle.</param>
/// <param name="Tokens">The chunk's token ids.</param>
/// <param name="PositionOffset">Position of the first token.</param>
/// <param name="Blocks">Attention block ids (at least enough for the keys after this chunk).</param>
/// <param name="RecurrentSlot">Recurrent slot.</param>
/// <param name="Sample">Whether the chunk samples.</param>
public sealed record TestSequence(ulong Handle, int[] Tokens, int PositionOffset, int[] Blocks, int RecurrentSlot, bool Sample = true)
{
    /// <summary>Stop ids.</summary>
    public int[] Stops { get; init; } = [];

    /// <summary>Tokens the request may still emit.</summary>
    public uint MaxNewTokensLeft { get; init; }

    /// <summary>Sampling record (greedy by default).</summary>
    public PlanSampling Sampling { get; init; } = new() { Temperature = 0f, TopP = 1f, TopK = 0, RepetitionPenalty = 1f, Flags = SamplingOptions.Greedy };
}

/// <summary>
/// A C2 step plan writer for tests and the GPU harness: the layout of <c>roost_worker_shm.h</c> with two state groups
/// (attention first, recurrent second). The engine's own writer belongs to stream G; this one exists so the worker's reader
/// is exercised against bytes laid out table by table.
/// </summary>
public static class PlanWriter
{
    /// <summary>Writes a plan.</summary>
    /// <param name="stepId">The step id.</param>
    /// <param name="groupCount">State groups per row; the attention group is index 0.</param>
    /// <param name="sequences">The sequences, in plan order.</param>
    /// <returns>The plan bytes.</returns>
    public static byte[] Write(ulong stepId, int groupCount, params TestSequence[] sequences)
    {
        var seqs = new List<PlanSequence>();
        var tokens = new List<int>();
        var positions = new List<int>();
        var groups = new List<PlanGroupBlocks>();
        var blockIds = new List<int>();
        var sampling = new List<PlanSampling>();
        var stops = new List<int>();
        foreach (var s in sequences)
        {
            int row = seqs.Count;
            seqs.Add(new PlanSequence
            {
                SequenceHandle = s.Handle,
                TokenStart = (uint)tokens.Count,
                TokenCount = (uint)s.Tokens.Length,
                PositionOffset = (uint)s.PositionOffset,
                GroupBlocksStart = (uint)(row * groupCount),
                RecurrentSlot = s.RecurrentSlot,
                LoraSlot = -1,
                GrammarSlot = -1,
                EmbeddingSlot = -1,
                SamplingIndex = (uint)sampling.Count,
                Flags = (s.PositionOffset == 0 ? SequenceOptions.Prefill : SequenceOptions.None) | (s.Sample ? SequenceOptions.Sample : SequenceOptions.None),
                StopStart = (uint)stops.Count,
                StopCount = (uint)s.Stops.Length,
                MaxNewTokensLeft = s.MaxNewTokensLeft,
                HeadRow = -1,
                EncoderOutput = -1,
            });
            sampling.Add(s.Sampling);
            stops.AddRange(s.Stops);
            for (int i = 0; i < s.Tokens.Length; i++)
            {
                tokens.Add(s.Tokens[i]);
                positions.Add(s.PositionOffset + i);
            }

            groups.Add(new PlanGroupBlocks { BlockStart = (uint)blockIds.Count, BlockCount = (uint)s.Blocks.Length });
            blockIds.AddRange(s.Blocks);
            for (int g = 1; g < groupCount; g++)
            {
                groups.Add(new PlanGroupBlocks { BlockStart = (uint)blockIds.Count, BlockCount = 0 });
            }
        }

        int at = Align(Unsafe.SizeOf<StepPlanHeader>());
        TableRef Place<T>(int count)
            where T : unmanaged
        {
            var t = new TableRef { Offset = (ulong)at, Count = (uint)count, Stride = (uint)Unsafe.SizeOf<T>() };
            at = Align(at + (count * Unsafe.SizeOf<T>()));
            return t;
        }

        var header = new StepPlanHeader
        {
            Magic = SharedLayout.MagicPlan,
            LayoutVersion = SharedLayout.Version,
            HeaderSize = (uint)Unsafe.SizeOf<StepPlanHeader>(),
            StepId = stepId,
            SequenceCount = (uint)seqs.Count,
            TokenCount = (uint)tokens.Count,
            GroupCount = (uint)groupCount,
            Sequences = Place<PlanSequence>(seqs.Count),
            TokenIds = Place<int>(tokens.Count),
            Positions = Place<int>(positions.Count),
            GroupBlocks = Place<PlanGroupBlocks>(groups.Count),
            BlockIds = Place<int>(blockIds.Count),
            Sampling = Place<PlanSampling>(sampling.Count),
            StopTokenIds = Place<int>(stops.Count),
        };
        header.TotalBytes = (ulong)at;
        var buffer = new byte[at];
        MemoryMarshal.Write(buffer, header);
        Copy(buffer, header.Sequences, seqs);
        Copy(buffer, header.TokenIds, tokens);
        Copy(buffer, header.Positions, positions);
        Copy(buffer, header.GroupBlocks, groups);
        Copy(buffer, header.BlockIds, blockIds);
        Copy(buffer, header.Sampling, sampling);
        Copy(buffer, header.StopTokenIds, stops);
        return buffer;
    }

    private static void Copy<T>(byte[] buffer, TableRef table, List<T> items)
        where T : unmanaged
    {
        if (items.Count > 0)
        {
            MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(items)).CopyTo(buffer.AsSpan((int)table.Offset));
        }
    }

    private static int Align(int n) => (n + SharedLayout.Alignment - 1) / SharedLayout.Alignment * SharedLayout.Alignment;
}
