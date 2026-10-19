namespace Starlings.Roost.Engine.Rings;

/// <summary>What the host chooses when it creates the C8 segment.</summary>
public sealed record SegmentOptions
{
    /// <summary>Slots in the plan ring.</summary>
    public int PlanSlotCount { get; init; } = 8;

    /// <summary>Slots in the result ring.</summary>
    public int ResultSlotCount { get; init; } = 8;

    /// <summary>
    /// Bytes per ring slot: at least the 32 of a <c>RoostRingSlot</c>, a multiple of 8. The default of one cache line
    /// keeps the producer's write of slot n+1 off the line the consumer is reading slot n from.
    /// </summary>
    public int SlotBytes { get; init; } = 64;

    /// <summary>Plan buffers; at least 2 (two steps in flight, S8).</summary>
    public int PlanBufferCount { get; init; } = 2;

    /// <summary>Result buffers; at least 2.</summary>
    public int ResultBufferCount { get; init; } = 2;

    /// <summary>Bytes per plan buffer; a multiple of 64.</summary>
    public int PlanBufferBytes { get; init; } = 64 * 1024;

    /// <summary>Bytes per result buffer; a multiple of 64.</summary>
    public int ResultBufferBytes { get; init; } = 16 * 1024;

    /// <summary>Bytes of the embedding and encoder input region; a multiple of 64, may be 0.</summary>
    public long InputRegionBytes { get; init; }
}
