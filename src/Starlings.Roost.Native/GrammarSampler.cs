using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Native;

/// <summary>
/// The CPU sampler path with grammar masks (R-051): <see cref="CpuOps.Sample"/> over host logits rows, each row masked by the
/// grammar slot of its sequence (contract C2's <c>grammar_slot</c>), then every sampled token committed to its slot's matcher.
/// The device sampler takes the same masks after R-042.
/// </summary>
public static unsafe class GrammarSampler
{
    /// <summary>
    /// Samples one token per row. Rows whose slot is negative are sampled without a mask; every other row only from the tokens
    /// its grammar allows next. Each sampled token is then committed to its slot (<see cref="GrammarSlots.Advance"/>).
    /// </summary>
    /// <param name="args">The sampler arguments, with <see cref="SampleArgs.Mask"/> unset; <see cref="SampleArgs.Vocab"/> must equal <see cref="GrammarSlots.VocabSize"/>.</param>
    /// <param name="rowSlots">The grammar slot of each row (<see cref="SampleArgs.Rows"/> entries), -1 for none.</param>
    /// <param name="slots">The slot table.</param>
    /// <remarks>
    /// A row whose allowed tokens all have logit -inf or NaN gets id -1 (<see cref="CpuOps.Sample"/>) and its slot is not advanced;
    /// the caller ends that sequence. A sampled token the matcher then refuses would mean the mask and the matcher disagree, and is
    /// thrown as <see cref="InvalidOperationException"/> rather than emitted.
    /// </remarks>
    /// <exception cref="ArgumentException">The arguments disagree with the table, or a mask is already set.</exception>
    /// <exception cref="GrammarException">A matcher is in its error state.</exception>
    public static void Sample(in SampleArgs args, ReadOnlySpan<int> rowSlots, GrammarSlots slots)
    {
        ArgumentNullException.ThrowIfNull(slots);
        if (rowSlots.Length != args.Rows)
        {
            throw new ArgumentException($"{rowSlots.Length} grammar slots for {args.Rows} rows.", nameof(rowSlots));
        }

        if (args.Mask.Address != 0)
        {
            throw new ArgumentException("The mask is built from the grammar slots; leave SampleArgs.Mask unset.", nameof(args));
        }

        if (args.Vocab != slots.VocabSize)
        {
            throw new ArgumentException($"The logits have {args.Vocab} columns; the grammar masks have {slots.VocabSize} rows.", nameof(args));
        }

        slots.ComputeMasks(rowSlots);
        int words = slots.MaskWords;
        nuint bytes = (nuint)Math.Max(1, args.Rows) * (nuint)words * sizeof(uint);
        var masks = (uint*)NativeMemory.AlignedAlloc(bytes, 64);
        try
        {
            for (var r = 0; r < args.Rows; r++)
            {
                var row = new Span<uint>(masks + ((long)r * words), words);
                if (rowSlots[r] < 0)
                {
                    row.Fill(uint.MaxValue);
                }
                else
                {
                    slots.Mask(rowSlots[r]).CopyTo(row);
                }
            }

            var masked = args;
            masked.Mask = new DeviceSpan { Address = (nint)masks, Length = (ulong)args.Rows * (ulong)words, Dtype = DType.U32 };
            CpuOps.Sample(in masked);
        }
        finally
        {
            NativeMemory.AlignedFree(masks);
        }

        var ids = (int*)args.Ids.Address;
        for (var r = 0; r < args.Rows; r++)
        {
            int slot = rowSlots[r];
            if (slot < 0 || ids[r] < 0)
            {
                continue;
            }

            if (!slots.Advance(slot, ids[r]))
            {
                throw new InvalidOperationException($"Row {r} sampled token {ids[r]}, which grammar slot {slot} refuses although its mask allowed it.");
            }
        }
    }
}
