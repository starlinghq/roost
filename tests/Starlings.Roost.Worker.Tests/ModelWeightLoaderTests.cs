using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Weights;
using static Starlings.Roost.Worker.Tests.SyntheticCheckpoint;

namespace Starlings.Roost.Worker.Tests;

public sealed class ModelWeightLoaderTests : IDisposable
{
    private readonly DirectoryInfo _dir = Directory.CreateTempSubdirectory("roost-r025-");

    public void Dispose() => _dir.Delete(recursive: true);

    private static SyntheticTensor[] Tensors() =>
    [
        F32("a_log", 1, 4),
        Bf16("embed", 2, 10, 4),
        Bf16("gate", 3, 6, 4),
        Bf16("norm", 4, 8),
        Bf16("up", 5, 6, 4),
        Bf16("unused", 6, 5),
    ];

    private static EagerWeightSlot[] Slots() =>
    [
        new("embed", DType.BF16, 40, ["embed"], null),
        new("l0.gate_up", DType.BF16, 48, ["gate", "up"], static (sources, dst) =>
        {
            int at = 0;
            foreach (var s in sources)
            {
                s.Span.CopyTo(dst[at..]);
                at += s.Length;
            }
        }),
        new("l0.norm", DType.BF16, 8, ["norm"], static (sources, dst) =>
        {
            var src = MemoryMarshal.Cast<byte, ushort>(sources[0].Span);
            var d = MemoryMarshal.Cast<byte, ushort>(dst);
            for (int i = 0; i < src.Length; i++)
            {
                d[i] = (ushort)(src[i] ^ 1);
            }
        }),
        new("l0.a_log", DType.F32, 4, ["a_log"], null),
        new("tied_head", DType.BF16, 40, ["embed"], null),
    ];

    [Fact]
    public void LoadsVerifiesAndAssemblesSlotsInSlotOrder()
    {
        var tensors = Tensors();
        var file = Build(tensors);
        var path = Write(_dir, file);

        using var weights = ModelWeightLoader.LoadHost(path, Lock(file), Slots());

        Assert.Equal(Sha256(file), weights.Verification.Sha256);
        Assert.Equal(5, weights.Spans.Count);
        ReadOnlySpan<byte> Bytes(int i, int n) => weights.Spans[i].Dtype == DType.F32 ? Read(weights.Spans[i], n * 4) : Read(weights.Spans[i], n * 2);
        Assert.True(Bytes(0, 40).SequenceEqual(tensors[1].Data));
        var gateUp = Bytes(1, 48).ToArray();
        Assert.True(gateUp.AsSpan(0, 48).SequenceEqual(tensors[2].Data));
        Assert.True(gateUp.AsSpan(48, 48).SequenceEqual(tensors[4].Data));
        var norm = Bytes(2, 8).ToArray();
        Assert.Equal(tensors[3].Data.Where((_, i) => i % 2 == 0).Select(static b => (byte)(b ^ 1)), norm.Where((_, i) => i % 2 == 0));
        Assert.True(Bytes(3, 4).SequenceEqual(tensors[0].Data));
        // A tied slot is a second copy of the same tensor.
        Assert.True(Bytes(4, 40).SequenceEqual(tensors[1].Data));
        Assert.All(weights.Spans, s => Assert.Equal(0, s.Address % WeightLayout.Alignment));
    }

    [Fact]
    public void RefusesAFileThatIsNotTheLockedOne()
    {
        var file = Build(Tensors());
        var path = Write(_dir, file);
        var other = (byte[])file.Clone();
        other[^1] ^= 1;
        var e = Assert.Throws<WeightLoadException>(() => ModelWeightLoader.LoadHost(path, Lock(other), Slots()));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
    }

    [Fact]
    public void RefusesASlotWhoseTensorIsMissingOrWronglySized()
    {
        var file = Build(Tensors());
        var path = Write(_dir, file);
        var missing = new EagerWeightSlot[] { new("x", DType.BF16, 4, ["no.such.tensor"], null) };
        var e = Assert.Throws<WeightLoadException>(() => ModelWeightLoader.LoadHost(path, Lock(file), missing));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("no.such.tensor", e.Message, StringComparison.Ordinal);

        // A plain-copy slot must be the size of its one source: the checkpoint tensor is 40 elements, the slot says 41.
        var wrong = new EagerWeightSlot[] { new("embed", DType.BF16, 41, ["embed"], null) };
        var mismatch = Assert.Throws<WeightLoadException>(() => ModelWeightLoader.LoadHost(path, Lock(file), wrong));
        Assert.Equal(RoostErrorCode.AssetMismatch, mismatch.Error.Code);
    }

    private static unsafe ReadOnlySpan<byte> Read(DeviceSpan span, int bytes) => new((void*)span.Address, bytes);
}
