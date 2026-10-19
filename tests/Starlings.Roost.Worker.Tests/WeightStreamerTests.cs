using System.Buffers.Binary;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;
using Starlings.Roost.Worker.Weights;
using static Starlings.Roost.Worker.Tests.SyntheticCheckpoint;

namespace Starlings.Roost.Worker.Tests;

public sealed class WeightStreamerTests : IDisposable
{
    private readonly DirectoryInfo _dir = Directory.CreateTempSubdirectory("roost-r024-");

    public void Dispose() => _dir.Delete(recursive: true);

    /// <summary>File order: a F32 pair first, then BF16 sorted by name, the way the real checkpoint is laid out.</summary>
    private static SyntheticTensor[] FileOrder() =>
    [
        F32("layers.0.a_log", 1, 16),
        F32("layers.0.norm", 2, 128),
        Bf16("embed", 3, 100, 8),
        Bf16("layers.0.proj", 4, 24, 8),
        Bf16("layers.1.proj", 5, 24, 8),
        Bf16("layers.10.proj", 6, 24, 8),
        Bf16("unused.mtp", 7, 33, 3),
    ];

    /// <summary>Slot order is layer by layer, which is not file order; the unused tensor has no slot.</summary>
    private static WeightSlotSpec[] Slots() =>
    [
        Slot("embed", "embed", DType.BF16, 800),
        Slot("l0.proj", "layers.0.proj", DType.BF16, 192),
        Slot("l0.norm", "layers.0.norm", DType.F32, 128),
        Slot("l0.a_log", "layers.0.a_log", DType.F32, 16),
        Slot("l1.proj", "layers.1.proj", DType.BF16, 192),
        Slot("l10.proj", "layers.10.proj", DType.BF16, 192),
    ];

    [Theory]
    [InlineData(16)]
    [InlineData(37)]
    [InlineData(1000)]
    [InlineData(WeightStreamer.DefaultChunkBytes)]
    public void StreamsEachTensorToItsSlotInSlotOrderAndVerifies(int chunk)
    {
        var tensors = FileOrder();
        var file = Build(tensors);
        var path = Write(_dir, file);
        using var arena = new HostWeightArena();
        var done = new List<string>();

        var layout = WeightStreamer.Load(path, Lock(file), Slots(), arena, chunk, s => done.Add(s.Name));

        Assert.Equal(Sha256(file), layout.Sha256);
        Assert.Equal(file.Length, layout.BytesRead);
        var slots = Slots();
        Assert.Equal(slots.Length, layout.Slots.Count);
        long previousEnd = 0;
        for (var i = 0; i < slots.Length; i++)
        {
            var p = layout.Slots[i];
            Assert.Equal(slots[i].Name, p.Name);
            Assert.Equal(0, p.Offset % WeightLayout.Alignment);
            Assert.True(p.Offset >= previousEnd, "slots are laid out in slot order without overlap");
            previousEnd = p.Offset + p.Bytes;
            Assert.Equal(tensors.Single(t => t.Name == slots[i].Source).Data, arena.Bytes(p).ToArray());
            var span = arena.Span(p);
            Assert.Equal((ulong)slots[i].Elements, span.Length);
            Assert.Equal(slots[i].Dtype, span.Dtype);
        }

        Assert.True(layout.TotalBytes >= previousEnd);
        // Copies finish in file order: the F32 tensors, embed, then the layers; the tensor no slot takes is only hashed.
        Assert.Equal(["l0.a_log", "l0.norm", "embed", "l0.proj", "l1.proj", "l10.proj"], done);
    }

    [Fact]
    public void TwoSlotsMayShareOneTensor()
    {
        var file = Build(FileOrder());
        var path = Write(_dir, file);
        using var arena = new HostWeightArena();
        WeightSlotSpec[] slots = [Slot("embed", "embed", DType.BF16, 800), Slot("lm_head", "embed", DType.BF16, 800)];

        var layout = WeightStreamer.Load(path, Lock(file), slots, arena);

        Assert.Equal(arena.Bytes(layout.Slots[0]).ToArray(), arena.Bytes(layout.Slots[1]).ToArray());
        Assert.NotEqual(layout.Slots[0].Offset, layout.Slots[1].Offset);
    }

    [Fact]
    public void ASameSizeSubstitutionFailsAfterTheLastByteAndHandsOutNothing()
    {
        var file = Build(FileOrder());
        var lockedFile = Lock(file);
        file[^5] ^= 0x01; // inside the last tensor, no slot takes it: only the digest can tell
        var path = Write(_dir, file);
        using var arena = new SpyArena();

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, lockedFile, Slots(), arena, 64));

        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("SHA-256", e.Message);
        Assert.True(arena.Allocated);
        Assert.True(arena.Discarded);
        Assert.False(arena.Committed);
        Assert.Throws<InvalidOperationException>(() => arena.Inner.Span(new SlotPlacement(0, "embed", 0, 8, DType.BF16, 4)));
    }

    [Fact]
    public void AFlippedByteInsideASlottedTensorIsDetected()
    {
        var file = Build(FileOrder());
        var lockedFile = Lock(file);
        file[^200] ^= 0x80;
        var path = Write(_dir, file);
        using var arena = new SpyArena();

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, lockedFile, Slots(), arena));

        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.True(arena.Discarded);
    }

    [Fact]
    public void AFlippedByteInTheHeaderNameIsDetected()
    {
        var tensors = FileOrder();
        var file = Build(tensors);
        var lockedFile = Lock(file);
        var text = Utf8(file);
        var at = text.IndexOf("unused.mtp", StringComparison.Ordinal);
        file[at] ^= 0x01; // renames a tensor no slot takes: header still valid, size unchanged
        var path = Write(_dir, file);
        using var arena = new SpyArena();

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, lockedFile, Slots(), arena));

        Assert.Contains("SHA-256", e.Message);
        Assert.True(arena.Discarded);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void AFileOfAnotherSizeFailsBeforeAnyReadOrAllocation(int delta)
    {
        var file = Build(FileOrder());
        var lockedFile = Lock(file);
        var other = delta > 0 ? [.. file, 0] : file[..^1];
        var path = Write(_dir, other);
        using var arena = new SpyArena();

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, lockedFile, Slots(), arena));

        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("lock records", e.Message);
        Assert.False(arena.Allocated);
    }

    [Fact]
    public void AMissingFileIsAnAssetMismatch()
    {
        using var arena = new SpyArena();
        var e = Assert.Throws<WeightLoadException>(() =>
            WeightStreamer.Load(Path.Combine(_dir.FullName, "absent.safetensors"), new LockedFile("m", 10, new string('0', 64)), Slots(), arena));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.False(arena.Allocated);
    }

    [Fact]
    public void AHostileHeaderLengthFailsBeforeAllocation()
    {
        var file = Build(FileOrder());
        BinaryPrimitives.WriteUInt64LittleEndian(file, ulong.MaxValue >> 1);
        var path = Write(_dir, file);
        using var arena = new SpyArena();

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, Lock(file), Slots(), arena));

        Assert.Contains("header length", e.Message);
        Assert.False(arena.Allocated);
    }

    [Fact]
    public void AGapBetweenTensorRangesFailsBeforeAllocation()
    {
        var file = Build(FileOrder(), (name, r) => name == "embed" ? (r.Begin + 2, r.End + 2) : r);
        var path = Write(_dir, file);
        using var arena = new SpyArena();

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, Lock(file), Slots(), arena));

        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.False(arena.Allocated);
    }

    [Fact]
    public void ARangeThatDisagreesWithTheShapeFailsBeforeAllocation()
    {
        var file = Build(FileOrder(), (name, r) => name == "embed" ? (r.Begin, r.End - 2) : r);
        var path = Write(_dir, file);
        using var arena = new SpyArena();

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, Lock(file), Slots(), arena));

        Assert.Contains("does not match its shape", e.Message);
        Assert.False(arena.Allocated);
    }

    [Fact]
    public void AMissingTensorAWrongDtypeOrAWrongCountFailsBeforeAllocation()
    {
        var file = Build(FileOrder());
        var path = Write(_dir, file);
        foreach (var bad in new[]
        {
            Slot("x", "layers.9.proj", DType.BF16, 192),
            Slot("x", "layers.1.proj", DType.F32, 192),
            Slot("x", "layers.1.proj", DType.BF16, 191),
        })
        {
            using var arena = new SpyArena();
            var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, Lock(file), [bad], arena));
            Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
            Assert.False(arena.Allocated);
        }
    }

    [Fact]
    public void ADeviceThatCannotHoldTheWeightsIsATypedFailure()
    {
        var file = Build(FileOrder());
        var path = Write(_dir, file);
        using var arena = new SpyArena { FailAllocate = true };

        var e = Assert.Throws<WeightLoadException>(() => WeightStreamer.Load(path, Lock(file), Slots(), arena));

        Assert.Equal(RoostErrorCode.InsufficientDeviceMemory, e.Error.Code);
        Assert.True(arena.Discarded);
    }

    [Fact]
    public void LockEntriesAreReadFromTomlAndMalformedLocksAreRefused()
    {
        var sha = new string('a', 64);
        var text = System.Text.Encoding.UTF8.GetBytes(LockText("model.safetensors", 123, sha));
        var parsed = WeightLock.Parse(text);
        Assert.Equal(new LockedFile("model.safetensors", 123, sha), parsed.Get("model.safetensors"));
        Assert.Equal("0123456789abcdef0123456789abcdef01234567", parsed.Revision);
        Assert.Throws<WeightLoadException>(() => parsed.Get("other"));
        Assert.Throws<WeightLoadException>(() => WeightLock.Parse(System.Text.Encoding.UTF8.GetBytes(LockText("m", 1, "XYZ"))));
        Assert.Throws<WeightLoadException>(() => WeightLock.Parse("not = [toml"u8));
    }

    [Fact]
    public void TheCommittedQwen35LockNamesTheSingleSafetensorsFile()
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "Roost.slnx")))
        {
            root = root.Parent;
        }

        Assert.NotNull(root);
        var lockFile = WeightLock.Load(Path.Combine(root.FullName, "evidence", "registry", "qwen3.5-0.8b.lock.toml"));
        var entry = lockFile.Get("model.safetensors-00001-of-00001.safetensors");
        Assert.Equal(1746942600, entry.Size);
        Assert.Equal("04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696", entry.Sha256);
        Assert.Equal("2fc06364715b967f1860aea9cf38778875588b17", lockFile.Revision);
    }

    /// <summary>Records which sink calls the streamer made.</summary>
    private sealed class SpyArena : IWeightSink
    {
        public HostWeightArena Inner { get; } = new();

        public bool Allocated { get; private set; }

        public bool Committed { get; private set; }

        public bool Discarded { get; private set; }

        public bool FailAllocate { get; init; }

        public void Allocate(long bytes)
        {
            Allocated = true;
            if (FailAllocate)
            {
                throw new WeightLoadException(RoostErrorCode.InsufficientDeviceMemory, "no memory");
            }

            Inner.Allocate(bytes);
        }

        public void Write(long offset, ReadOnlySpan<byte> data) => Inner.Write(offset, data);

        public void Commit()
        {
            Committed = true;
            Inner.Commit();
        }

        public void Discard()
        {
            Discarded = true;
            Inner.Discard();
        }

        public void Dispose() => Inner.Dispose();
    }
}
