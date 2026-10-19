using Starlings.Roost.Abi;

namespace Starlings.Roost.Worker.Weights;

/// <summary>
/// One slot of a model's weight loader, in the order the model package's generated <c>WeightSlots</c> table gives
/// (R-020): its index is its index in <c>StepContext.Weights</c>. The streamer needs only these four facts, so it does
/// not depend on any model package.
/// </summary>
/// <param name="Name">The slot's name, unique within the list.</param>
/// <param name="Source">The checkpoint tensor that fills the slot. Several slots may name one tensor (tied embeddings).</param>
/// <param name="Dtype">The dtype of the checkpoint tensor; the slot is filled byte for byte, so it must match.</param>
/// <param name="Elements">The element count; the checkpoint tensor must have exactly this many.</param>
public readonly record struct WeightSlotSpec(string Name, string Source, DType Dtype, long Elements)
{
    /// <summary>Bytes of the slot.</summary>
    public long Bytes => SafetensorsDType.ByteLength(Dtype, Elements);
}

/// <summary>Where a slot lives in the weight arena once streaming has finished.</summary>
/// <param name="Index">The slot's index in the slot list.</param>
/// <param name="Name">The slot's name.</param>
/// <param name="Offset">Byte offset in the arena; a multiple of <see cref="WeightLayout.Alignment"/>.</param>
/// <param name="Bytes">Byte length.</param>
/// <param name="Dtype">The dtype.</param>
/// <param name="Elements">The element count.</param>
public readonly record struct SlotPlacement(int Index, string Name, long Offset, long Bytes, DType Dtype, long Elements);

/// <summary>The device layout of a streamed checkpoint, in slot order, and the facts the verification established.</summary>
public sealed class WeightLayout
{
    /// <summary>Every slot starts at a multiple of this many bytes (cuBLASLt and the vector loads of the kernels want 256).</summary>
    public const long Alignment = 256;

    internal WeightLayout(SlotPlacement[] slots, long totalBytes, string sha256, long bytesRead, TimeSpan elapsed)
    {
        Slots = slots;
        TotalBytes = totalBytes;
        Sha256 = sha256;
        BytesRead = bytesRead;
        Elapsed = elapsed;
    }

    /// <summary>The placements, indexed by slot.</summary>
    public IReadOnlyList<SlotPlacement> Slots { get; }

    /// <summary>The arena size in bytes.</summary>
    public long TotalBytes { get; }

    /// <summary>The verified SHA-256 of the whole file, lower-case hexadecimal.</summary>
    public string Sha256 { get; }

    /// <summary>Bytes of the file that were read and hashed (the whole file, including tensors no slot takes).</summary>
    public long BytesRead { get; }

    /// <summary>Wall time from the first read to the digest comparison.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>Lays slots out in order, each at the next aligned offset.</summary>
    internal static (SlotPlacement[] Slots, long Total) Plan(IReadOnlyList<WeightSlotSpec> specs)
    {
        var placements = new SlotPlacement[specs.Count];
        long offset = 0;
        for (var i = 0; i < specs.Count; i++)
        {
            var s = specs[i];
            var bytes = s.Bytes;
            placements[i] = new SlotPlacement(i, s.Name, offset, bytes, s.Dtype, s.Elements);
            offset = checked(offset + bytes + (Alignment - 1)) / Alignment * Alignment;
        }

        return (placements, offset);
    }
}
