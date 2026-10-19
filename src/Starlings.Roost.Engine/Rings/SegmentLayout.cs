using System.Runtime.CompilerServices;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// Computes, writes and validates the C8 segment layout. Order in the segment: segment header, handshake, heartbeat,
/// then for each ring its <see cref="RingHeader"/>, its producer cursor, its consumer cursor (each alone on a 64-byte
/// line) and its slots; then the plan buffers, the result buffers and the input region; the total is rounded up to a
/// page. The order is this implementation's choice; readers use only the offsets in the headers.
/// </summary>
internal sealed unsafe class SegmentLayout
{
    private const int Line = RingSegment.Alignment;
    private const long Page = 4096;
    private const uint LayoutVersion = 1; // ROOST_SHM_LAYOUT_VERSION
    private const uint MagicSegment = 0x47455352u;
    private const uint MagicRing = 0x474E5252u;

    private readonly SegmentOptions _options;
    private SegmentHeader _header;
    private RingHeader _planRing;
    private RingHeader _resultRing;

    private SegmentLayout(SegmentOptions options)
    {
        _options = options;
    }

    public long TotalBytes => (long)_header.SegmentBytes;

    public static SegmentLayout Compute(SegmentOptions o)
    {
        if (o.PlanSlotCount < 1 || o.ResultSlotCount < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(o), "A ring needs at least one slot.");
        }

        if (o.SlotBytes < sizeof(RingSlot) || o.SlotBytes % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(o), "SlotBytes must be at least sizeof(RingSlot) and a multiple of 8.");
        }

        if (o.PlanBufferCount < 2 || o.ResultBufferCount < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(o), "At least two plan and two result buffers (two steps in flight).");
        }

        if (o.PlanBufferBytes < Line || o.PlanBufferBytes % Line != 0 || o.ResultBufferBytes < Line || o.ResultBufferBytes % Line != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(o), "Buffer sizes must be positive multiples of 64.");
        }

        if (o.InputRegionBytes < 0 || o.InputRegionBytes % Line != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(o), "InputRegionBytes must be a non-negative multiple of 64.");
        }

        var layout = new SegmentLayout(o);
        layout.Fill();
        return layout;
    }

    public void WriteHeaders(byte* segment)
    {
        Unsafe.Write(segment, _header);
        Unsafe.Write(segment + (long)_header.PlanRingOffset, _planRing);
        Unsafe.Write(segment + (long)_header.ResultRingOffset, _resultRing);
    }

    /// <summary>Checks the whole header chain of a mapped segment of <paramref name="mappedBytes"/> bytes.</summary>
    public static void Validate(byte* segment, long mappedBytes)
    {
        if (mappedBytes < sizeof(SegmentHeader))
        {
            throw Bad("file is smaller than a segment header");
        }

        var h = Unsafe.Read<SegmentHeader>(segment);
        if (h.Magic != MagicSegment)
        {
            throw Bad("segment magic");
        }

        if (h.LayoutVersion != LayoutVersion)
        {
            throw Bad($"layout version {h.LayoutVersion}, expected {LayoutVersion}");
        }

        if (h.HeaderSize < (uint)sizeof(SegmentHeader))
        {
            throw Bad("header_size smaller than the known header");
        }

        if (h.ContractVersion != (uint)ContractVersion.Current)
        {
            throw Bad($"contract version {h.ContractVersion}, expected {ContractVersion.Current}");
        }

        if (h.SegmentBytes == 0 || h.SegmentBytes > (ulong)mappedBytes)
        {
            throw Bad("segment_bytes exceeds the mapping");
        }

        var total = (long)h.SegmentBytes;
        Range("handshake", h.HandshakeOffset, (ulong)sizeof(Handshake), total);
        Range("heartbeat", h.HeartbeatOffset, (ulong)sizeof(Heartbeat), total);
        Range("plan ring header", h.PlanRingOffset, (ulong)sizeof(RingHeader), total);
        Range("result ring header", h.ResultRingOffset, (ulong)sizeof(RingHeader), total);
        if (h.PlanBufferCount < 2 || h.ResultBufferCount < 2)
        {
            throw Bad("fewer than two plan or result buffers");
        }

        if (h.PlanBufferBytes == 0 || h.PlanBufferBytes % Line != 0 || h.ResultBufferBytes == 0 || h.ResultBufferBytes % Line != 0)
        {
            throw Bad("buffer bytes not a positive multiple of 64");
        }

        if (h.PlanBufferBytes > int.MaxValue || h.ResultBufferBytes > int.MaxValue)
        {
            throw Bad("buffer larger than 2 GiB");
        }

        Range("plan buffers", h.PlanBuffersOffset, h.PlanBufferBytes * h.PlanBufferCount, total);
        Range("result buffers", h.ResultBuffersOffset, h.ResultBufferBytes * h.ResultBufferCount, total);
        Range("input region", h.InputRegionOffset, h.InputRegionBytes, total);
        ValidateRing("plan", segment, h.PlanRingOffset, total);
        ValidateRing("result", segment, h.ResultRingOffset, total);
    }

    private static void ValidateRing(string name, byte* segment, ulong offset, long total)
    {
        var r = Unsafe.Read<RingHeader>(segment + (long)offset);
        if (r.Magic != MagicRing || r.LayoutVersion != LayoutVersion)
        {
            throw Bad($"{name} ring magic or layout version");
        }

        if (r.SlotCount == 0 || r.SlotBytes < (uint)sizeof(RingSlot) || r.SlotBytes % 8 != 0)
        {
            throw Bad($"{name} ring slot_count or slot_bytes");
        }

        Range($"{name} ring slots", r.SlotsOffset, (ulong)r.SlotCount * r.SlotBytes, total);
        Range($"{name} producer cursor", r.ProducerCursorOffset, 8, total);
        Range($"{name} consumer cursor", r.ConsumerCursorOffset, 8, total);
        if (r.ProducerCursorOffset / Line == r.ConsumerCursorOffset / Line)
        {
            throw Bad($"{name} cursors share a cache line");
        }

        var slotsEnd = r.SlotsOffset + ((ulong)r.SlotCount * r.SlotBytes);
        foreach (var cursor in new[] { r.ProducerCursorOffset, r.ConsumerCursorOffset })
        {
            if (cursor >= r.SlotsOffset && cursor < slotsEnd)
            {
                throw Bad($"{name} cursor inside the slots");
            }
        }
    }

    private static void Range(string what, ulong offset, ulong length, long total)
    {
        if (offset % Line != 0)
        {
            throw Bad($"{what} offset {offset} is not 64-byte aligned");
        }

        if (offset > (ulong)total || length > (ulong)total - offset)
        {
            throw Bad($"{what} lies outside the segment");
        }
    }

    private static InvalidDataException Bad(string why) => new($"Invalid C8 segment: {why}.");

    private static long Up(long value, long to) => (value + to - 1) / to * to;

    private void Fill()
    {
        var o = _options;
        var at = Up(sizeof(SegmentHeader), Line);
        var handshake = at;
        at += Up(sizeof(Handshake), Line);
        var heartbeat = at;
        at += Up(sizeof(Heartbeat), Line);

        _planRing = PlaceRing(ref at, o.PlanSlotCount, o.SlotBytes, out var planRingOffset);
        _resultRing = PlaceRing(ref at, o.ResultSlotCount, o.SlotBytes, out var resultRingOffset);

        var planBuffers = at;
        at += (long)o.PlanBufferBytes * o.PlanBufferCount;
        var resultBuffers = at;
        at += (long)o.ResultBufferBytes * o.ResultBufferCount;
        var input = at;
        at += o.InputRegionBytes;

        _header = new SegmentHeader
        {
            Magic = MagicSegment,
            LayoutVersion = LayoutVersion,
            HeaderSize = (uint)sizeof(SegmentHeader),
            ContractVersion = (uint)ContractVersion.Current,
            SegmentBytes = (ulong)Up(at, Page),
            HandshakeOffset = (ulong)handshake,
            HeartbeatOffset = (ulong)heartbeat,
            PlanRingOffset = (ulong)planRingOffset,
            ResultRingOffset = (ulong)resultRingOffset,
            PlanBuffersOffset = (ulong)planBuffers,
            PlanBufferBytes = (ulong)o.PlanBufferBytes,
            ResultBuffersOffset = (ulong)resultBuffers,
            ResultBufferBytes = (ulong)o.ResultBufferBytes,
            PlanBufferCount = (uint)o.PlanBufferCount,
            ResultBufferCount = (uint)o.ResultBufferCount,
            InputRegionOffset = (ulong)input,
            InputRegionBytes = (ulong)o.InputRegionBytes,
        };
    }

    private static RingHeader PlaceRing(ref long at, int slotCount, int slotBytes, out long headerOffset)
    {
        headerOffset = at;
        at += Line;
        var producer = at;
        at += Line;
        var consumer = at;
        at += Line;
        var slots = at;
        at += Up((long)slotCount * slotBytes, Line);
        return new RingHeader
        {
            Magic = MagicRing,
            LayoutVersion = LayoutVersion,
            SlotCount = (uint)slotCount,
            SlotBytes = (uint)slotBytes,
            SlotsOffset = (ulong)slots,
            ProducerCursorOffset = (ulong)producer,
            ConsumerCursorOffset = (ulong)consumer,
        };
    }
}
