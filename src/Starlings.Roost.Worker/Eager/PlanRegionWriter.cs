using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Worker.Eager;

/// <summary>
/// Fills the model's plan-input region (token ids, positions, cache slot mapping, batch tables, selected rows, state
/// slots) from a validated step plan. The names and element types come from the model's layout, so the writer is the same
/// for every model whose generated forward uses the standard inputs. The region is filled on the host and copied to the
/// device in one piece (the plan copy).
/// </summary>
public static class PlanRegionWriter
{
    /// <summary>Writes the inputs.</summary>
    /// <param name="region">The host mirror of the plan region, <see cref="EagerLayout.PlanBytes"/> long.</param>
    /// <param name="layout">Offsets and element types of the inputs.</param>
    /// <param name="plan">The checked plan.</param>
    /// <param name="blockSize">Tokens per attention block.</param>
    public static void Fill(Span<byte> region, EagerLayout layout, ParsedPlan plan, int blockSize)
    {
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(plan);
        region.Clear();
        int seqCount = plan.Sequences.Length;
        int blockAt = 0;
        int row = 0;
        int qEnd = 0;
        for (int b = 0; b < seqCount; b++)
        {
            var s = plan.Sequences[b];
            Put(region, layout, "QIndptr", b, s.TokenStart);
            Put(region, layout, "CuSeqlens", b, s.TokenStart);
            Put(region, layout, "KvIndptr", b, blockAt);
            for (int i = 0; i < s.Blocks.Length; i++)
            {
                Put(region, layout, "KvIndices", blockAt + i, s.Blocks[i]);
            }

            blockAt += s.Blocks.Length;
            int pages = s.Blocks.Length;
            Put(region, layout, "KvLastPageLength", b, s.KvLength - ((pages - 1) * blockSize));
            Put(region, layout, "StateSlots", b, s.RecurrentSlot);
            for (int t = 0; t < s.TokenCount; t++)
            {
                int index = s.TokenStart + t;
                int position = s.PositionOffset + t;
                Put(region, layout, "Ids", index, plan.TokenIds[index]);
                Put(region, layout, "Positions", index, position);
                long slot = ((long)s.Blocks[position / blockSize] * blockSize) + (position % blockSize);
                Put(region, layout, "SlotMapping", index, slot);
            }

            if (s.Sample)
            {
                Put(region, layout, "Rows", row++, s.TokenStart + s.TokenCount - 1);
            }

            qEnd = s.TokenStart + s.TokenCount;
        }

        Put(region, layout, "QIndptr", seqCount, qEnd);
        Put(region, layout, "CuSeqlens", seqCount, qEnd);
        Put(region, layout, "KvIndptr", seqCount, blockAt);
    }

    private static void Put(Span<byte> region, EagerLayout layout, string name, int index, long value)
    {
        if (!layout.Offsets.TryGetValue(name, out long offset))
        {
            throw new InvalidOperationException($"the model's plan layout has no input {name}");
        }

        switch (layout.Dtypes[name])
        {
            case DType.I32:
                MemoryMarshal.Write(region.Slice(checked((int)offset) + (index * 4), 4), checked((int)value));
                break;
            case DType.I64:
                MemoryMarshal.Write(region.Slice(checked((int)offset) + (index * 8), 8), value);
                break;
            default:
                throw new InvalidOperationException($"plan input {name} has unsupported dtype {layout.Dtypes[name]}");
        }
    }
}
