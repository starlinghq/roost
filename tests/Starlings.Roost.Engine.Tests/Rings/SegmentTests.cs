using Starlings.Roost.Engine.Rings;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Tests.Rings;

public sealed unsafe class SegmentTests : IDisposable
{
    private readonly string _path = Path.Combine(RingSegment.DefaultDirectory(), $"roost-seg-test-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (File.Exists(_path))
        {
            File.Delete(_path);
        }
    }

    [Fact]
    public void LayoutIsAlignedDisjointAndInsideTheSegment()
    {
        using var s = RingSegment.Create(_path, new SegmentOptions { InputRegionBytes = 4096 });
        var h = s.Header;
        var ranges = new List<(string Name, ulong Start, ulong Length)>
        {
            ("header", 0, (ulong)sizeof(SegmentHeader)),
            ("handshake", h.HandshakeOffset, (ulong)sizeof(Handshake)),
            ("heartbeat", h.HeartbeatOffset, (ulong)sizeof(Heartbeat)),
            ("plan buffers", h.PlanBuffersOffset, h.PlanBufferBytes * h.PlanBufferCount),
            ("result buffers", h.ResultBuffersOffset, h.ResultBufferBytes * h.ResultBufferCount),
            ("input", h.InputRegionOffset, h.InputRegionBytes),
        };
        foreach (var (name, ringOffset) in new[] { ("plan", h.PlanRingOffset), ("result", h.ResultRingOffset) })
        {
            var r = System.Runtime.CompilerServices.Unsafe.Read<RingHeader>(s.Base + (long)ringOffset);
            ranges.Add(($"{name} ring header", ringOffset, (ulong)sizeof(RingHeader)));
            ranges.Add(($"{name} producer cursor", r.ProducerCursorOffset, 8));
            ranges.Add(($"{name} consumer cursor", r.ConsumerCursorOffset, 8));
            ranges.Add(($"{name} slots", r.SlotsOffset, (ulong)r.SlotCount * r.SlotBytes));
            Assert.NotEqual(r.ProducerCursorOffset / 64, r.ConsumerCursorOffset / 64);
        }

        foreach (var (name, start, length) in ranges)
        {
            Assert.True(start % 64 == 0, $"{name} is not 64-byte aligned");
            Assert.True(start + length <= h.SegmentBytes, $"{name} is outside the segment");
        }

        // Cursors sit alone on a cache line: no other range intrudes on the 64 bytes that contain them.
        foreach (var cursor in ranges.Where(r => r.Name.Contains("cursor", StringComparison.Ordinal)))
        {
            var line = cursor.Start / 64;
            Assert.DoesNotContain(ranges.Where(r => r.Name != cursor.Name && r.Length > 0), r => r.Start / 64 <= line && (r.Start + r.Length - 1) / 64 >= line);
        }

        var sorted = ranges.Where(r => r.Length > 0).OrderBy(r => r.Start).ToList();
        for (var i = 1; i < sorted.Count; i++)
        {
            Assert.True(sorted[i - 1].Start + sorted[i - 1].Length <= sorted[i].Start, $"{sorted[i - 1].Name} overlaps {sorted[i].Name}");
        }

        Assert.Equal(0UL, h.SegmentBytes % 4096);
        Assert.Equal((uint)ContractVersion.Current, h.ContractVersion);
    }

    [Fact]
    public void OpenSeesWhatCreateWrote()
    {
        using var created = RingSegment.Create(_path, new SegmentOptions());
        using var opened = RingSegment.Open(_path);
        Assert.Equal(created.Header.SegmentBytes, opened.Header.SegmentBytes);
        Assert.Equal(created.Header.PlanBuffersOffset, opened.Header.PlanBuffersOffset);

        // Same pages: a write through one mapping is visible through the other.
        var host = created.OpenHost();
        var worker = opened.OpenWorker();
        Assert.True(host.CanPublish());
        host.PlanBuffer(0)[0] = 0x5a;
        Assert.True(host.TryPublish(new RingSlot { StepId = 7, Kind = RingSlotKind.Step, BufferIndex = 0 }));
        Assert.True(worker.TryTakePlan(out var slot));
        Assert.Equal(7UL, slot.StepId);
        Assert.Equal(0x5a, worker.PlanBuffer(0)[0]);
    }

    [Theory]
    [InlineData(0, 0x12345678u)] // segment magic
    [InlineData(4, 99u)] // layout version
    [InlineData(12, 77u)] // contract version
    public void OpenRejectsABadHeaderWord(int byteOffset, uint value)
    {
        using (RingSegment.Create(_path, new SegmentOptions()))
        {
            using var raw = new FileStream(_path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
            raw.Position = byteOffset;
            raw.Write(BitConverter.GetBytes(value));
            raw.Flush();
            Assert.Throws<InvalidDataException>(() => RingSegment.Open(_path));
        }
    }

    [Fact]
    public void OpenRejectsATruncatedFile()
    {
        using (RingSegment.Create(_path, new SegmentOptions()))
        {
            using var raw = new FileStream(_path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
            raw.SetLength(8192);
            Assert.Throws<InvalidDataException>(() => RingSegment.Open(_path));
        }
    }

    [Theory]
    [InlineData(0, 64, 2, 2)]
    [InlineData(8, 16, 2, 2)]
    [InlineData(8, 60, 2, 2)]
    [InlineData(8, 64, 1, 2)]
    [InlineData(8, 64, 2, 1)]
    public void CreateRejectsBadOptions(int slots, int slotBytes, int planBuffers, int resultBuffers)
    {
        var options = new SegmentOptions { PlanSlotCount = slots, SlotBytes = slotBytes, PlanBufferCount = planBuffers, ResultBufferCount = resultBuffers };
        Assert.Throws<ArgumentOutOfRangeException>(() => RingSegment.Create(_path, options));
        Assert.False(File.Exists(_path));
    }
}
