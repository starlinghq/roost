using System.Globalization;
using Starlings.Roost.Abi;
using Starlings.Roost.Models.Qwen35;

namespace Starlings.Roost.Models.Qwen35.Tests;

/// <summary>The weight map against the locked checkpoint's tensor listing (names, dtypes, shapes), and the transforms.</summary>
public class Qwen35WeightMapTests
{
    private static long[] SlotShape(Qwen35Model.WeightSlot s) => s.Shape.Split(',').Select(x => long.Parse(x, CultureInfo.InvariantCulture)).ToArray();

    private static string DtypeName(DType t) => t switch
    {
        DType.BF16 => "BF16",
        DType.F32 => "F32",
        _ => t.ToString(),
    };

    [Fact]
    public void EverySlotsTensorsExistWithTheShapeAndDtypeTheSlotNeeds()
    {
        var header = CheckpointHeader.Tensors;
        foreach (var slot in Qwen35Model.WeightSlots)
        {
            var src = Qwen35WeightMap.For(slot);
            var tensors = src.Tensors.Select(t => header.TryGetValue(t, out var h) ? h : throw new Xunit.Sdk.XunitException($"{slot.Name}: no tensor {t} in the checkpoint")).ToArray();
            Assert.All(tensors, t => Assert.Equal(DtypeName(slot.Dtype), t.Dtype));
            var shape = SlotShape(slot);
            switch (src.Transform)
            {
                case WeightTransform.None or WeightTransform.OnePlus:
                    Assert.Equal(shape, Assert.Single(tensors).Shape);
                    break;
                case WeightTransform.SqueezeConv:
                    Assert.Equal([shape[0], 1, shape[1]], Assert.Single(tensors).Shape);
                    break;
                case WeightTransform.QueryRows or WeightTransform.GateRows:
                    // [heads * 2 * head_dim, hidden] holds the query rows and the gate rows of every head.
                    Assert.Equal([shape[0] * 2, shape[1]], Assert.Single(tensors).Shape);
                    break;
                case WeightTransform.ConcatRows:
                    Assert.Equal(2, tensors.Length);
                    Assert.All(tensors, t => Assert.Equal([shape[0] / 2, shape[1]], t.Shape));
                    break;
            }

            Assert.Equal(slot.Elements, shape.Aggregate(1L, (a, d) => a * d));
        }
    }

    [Fact]
    public void EveryTextTensorFillsASlotAndTheMtpHeadFillsNone()
    {
        var header = CheckpointHeader.Tensors;
        var used = Qwen35WeightMap.All().SelectMany(s => s.Tensors).ToHashSet(StringComparer.Ordinal);

        var text = header.Keys.Where(k => k.StartsWith("model.language_model.", StringComparison.Ordinal)).ToArray();
        Assert.Equal(320, text.Length);
        Assert.Equal(text.Order().ToArray(), used.Order().ToArray());

        var mtp = header.Keys.Where(k => k.StartsWith("mtp.", StringComparison.Ordinal)).Order().ToArray();
        Assert.Equal(15, mtp.Length);
        Assert.Equal(mtp, Qwen35WeightMap.Mtp.Order().ToArray());
        Assert.DoesNotContain(Qwen35WeightMap.Mtp, used.Contains);

        // Nothing else is in the listing: the vision tower (153 more tensors in the file) is not part of it, and there is no lm_head.
        Assert.Equal(text.Length + mtp.Length, header.Count);
        Assert.DoesNotContain("lm_head.weight", header.Keys);
        Assert.DoesNotContain(Qwen35Model.WeightSlots, s => s.Source!.StartsWith(Qwen35WeightMap.VisionPrefix, StringComparison.Ordinal));
    }

    [Fact]
    public void TransformsFollowTheSlotRoles()
    {
        WeightTransform T(string name) => Qwen35WeightMap.For(Qwen35Model.WeightSlots.Single(s => s.Name == name)).Transform;
        Assert.Equal(WeightTransform.None, T("embed"));
        Assert.Equal(WeightTransform.OnePlus, T("l0.input_norm"));
        Assert.Equal(WeightTransform.OnePlus, T("l23.post_norm"));
        Assert.Equal(WeightTransform.OnePlus, T("l3.q_norm"));
        Assert.Equal(WeightTransform.OnePlus, T("l3.k_norm"));
        Assert.Equal(WeightTransform.OnePlus, T("final_norm"));
        // The linear layers' gated norm is NOT zero-centred (Qwen3_5RMSNormGated initialises to ones and multiplies by w).
        Assert.Equal(WeightTransform.None, T("l0.linear_norm"));
        Assert.Equal(WeightTransform.QueryRows, T("l3.q_proj"));
        Assert.Equal(WeightTransform.GateRows, T("l3.q_gate_proj"));
        Assert.Equal(WeightTransform.SqueezeConv, T("l0.conv"));
        Assert.Equal(WeightTransform.ConcatRows, T("l5.gate_up"));
        Assert.Equal(WeightTransform.None, T("l0.in_proj_qkv"));
        Assert.Equal(
            ["model.language_model.layers.5.mlp.gate_proj.weight", "model.language_model.layers.5.mlp.up_proj.weight"],
            Qwen35WeightMap.For(Qwen35Model.WeightSlots.Single(s => s.Name == "l5.gate_up")).Tensors.ToArray());
    }

    [Fact]
    public void OnePlusIsComputedInFloatAndRoundedToNearestEven()
    {
        ushort Bits(float v) => Qwen35WeightMap.ToBFloat16Bits(v);
        ushort[] src = [Bits(0f), Bits(-0.5f), Bits(0.25f), Bits(0.00390625f), Bits(-1f)];
        var dst = new ushort[src.Length];
        Qwen35WeightMap.OnePlus(src, dst);
        Assert.Equal([1f, 0.5f, 1.25f, 1f, 0f], dst.Select(Qwen35WeightMap.BFloat16ToSingle).ToArray());

        // 1 + 2^-8 is half a bfloat16 step above 1: ties go to the even mantissa (1.0); 1 + 3 * 2^-8 goes up to 1 + 2^-6... (1.015625).
        Assert.Equal(1f, Qwen35WeightMap.BFloat16ToSingle(Bits(1f + 0.00390625f)));
        Assert.Equal(1.015625f, Qwen35WeightMap.BFloat16ToSingle(Bits(1f + (3 * 0.00390625f))));
        Assert.Throws<ArgumentException>(() => Qwen35WeightMap.OnePlus(src, new ushort[2]));
    }

    [Fact]
    public void QueryAndGateRowsAreTheTwoHalvesOfEachHeadsRows()
    {
        // 3 heads, head size 2, hidden 2: checkpoint rows are q0 q0 g0 g0 q1 q1 g1 g1 q2 q2 g2 g2 (each row two columns, row r holds 10 * r, 10 * r + 1).
        int heads = 3;
        int headDim = 2;
        int hidden = 2;
        var src = Enumerable.Range(0, heads * 2 * headDim * hidden).Select(i => (ushort)((i / hidden * 10) + (i % hidden))).ToArray();
        var query = new ushort[heads * headDim * hidden];
        var gate = new ushort[query.Length];
        Qwen35WeightMap.SplitQueryRows<ushort>(src, query, heads, headDim, hidden, gate: false);
        Qwen35WeightMap.SplitQueryRows<ushort>(src, gate, heads, headDim, hidden, gate: true);
        Assert.Equal(new ushort[] { 0, 1, 10, 11, 40, 41, 50, 51, 80, 81, 90, 91 }, query);
        Assert.Equal(new ushort[] { 20, 21, 30, 31, 60, 61, 70, 71, 100, 101, 110, 111 }, gate);
        Assert.Throws<ArgumentException>(() => Qwen35WeightMap.SplitQueryRows<ushort>(src, new ushort[3], heads, headDim, hidden, gate: false));
    }
}
