using System.Globalization;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Qwen35.Tests;

/// <summary>
/// A small run of the real checkpoint's weights through the generated forward on <see cref="CpuOps"/>. It needs about 2 GB and a
/// minute, so it runs only when ROOST_QWEN35_CHECKPOINT names a directory holding
/// <c>model.safetensors-00001-of-00001.safetensors</c> (the locked file; nothing is copied). It checks that the weight map, state
/// groups and views work end to end on real data: the forward runs, the logits are finite and distinct, and prefill followed by a
/// decode step agrees with one longer prefill. It is NOT a parity test (per-layer parity is R-022); its top tokens are development evidence.
/// </summary>
public sealed unsafe class Qwen35SmokeTests : IDisposable
{
    private const string FileName = "model.safetensors-00001-of-00001.safetensors";
    private const int BlockSize = 16;
    private const int Vocab = 248320;

    private static readonly int[] Prompt = [760, 6511, 314, 9338, 369];
    private static readonly Qwen35Model.ForwardLayout Layout = new(maxTokens: 8, maxSequences: 1, maxKvBlocks: 2);

    private readonly List<nint> _blocks = [];

    public static bool HasCheckpoint =>
        Environment.GetEnvironmentVariable("ROOST_QWEN35_CHECKPOINT") is { Length: > 0 } dir && File.Exists(Path.Combine(dir, FileName));

    public void Dispose()
    {
        foreach (nint p in _blocks)
        {
            NativeMemory.AlignedFree((void*)p);
        }
    }

    private DeviceSpan Alloc(long elements, DType dtype)
    {
        long bytes = Math.Max(elements * (dtype is DType.BF16 ? 2 : dtype is DType.F32 or DType.I32 ? 4 : dtype is DType.I64 ? 8 : 1), 64);
        void* p = NativeMemory.AlignedAlloc((nuint)bytes, 64);
        new Span<byte>(p, checked((int)Math.Min(bytes, int.MaxValue))).Clear();
        if (bytes > int.MaxValue)
        {
            NativeMemory.Clear((byte*)p + int.MaxValue, (nuint)(bytes - int.MaxValue));
        }

        _blocks.Add((nint)p);
        return new DeviceSpan { Address = (nint)p, Length = (ulong)elements, Dtype = dtype };
    }

    private DeviceSpan[] LoadWeights(SafetensorsFile file)
    {
        var weights = new DeviceSpan[Qwen35Model.WeightSlots.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            var slot = Qwen35Model.WeightSlots[i];
            var src = Qwen35WeightMap.For(slot);
            var span = Alloc(slot.Elements, slot.Dtype);
            weights[i] = span;
            int elementBytes = slot.Dtype == DType.F32 ? 4 : 2;
            var dst = new Span<byte>((void*)span.Address, checked((int)(slot.Elements * elementBytes)));
            switch (src.Transform)
            {
                case WeightTransform.None or WeightTransform.SqueezeConv:
                    file.Bytes(src.Tensors[0]).CopyTo(dst);
                    break;
                case WeightTransform.ConcatRows:
                    int at = 0;
                    foreach (string t in src.Tensors)
                    {
                        file.Bytes(t).CopyTo(dst[at..]);
                        at += file.Bytes(t).Length;
                    }

                    Assert.Equal(dst.Length, at);
                    break;
                case WeightTransform.OnePlus:
                    Qwen35WeightMap.OnePlus(MemoryMarshal.Cast<byte, ushort>(file.Bytes(src.Tensors[0])), MemoryMarshal.Cast<byte, ushort>(dst));
                    break;
                case WeightTransform.QueryRows or WeightTransform.GateRows:
                    Qwen35WeightMap.SplitQueryRows(
                        MemoryMarshal.Cast<byte, ushort>(file.Bytes(src.Tensors[0])),
                        MemoryMarshal.Cast<byte, ushort>(dst),
                        heads: 8,
                        headDim: 256,
                        hidden: 1024,
                        gate: src.Transform == WeightTransform.GateRows);
                    break;
            }
        }

        return weights;
    }

    private DeviceSpan[] NewState()
    {
        var state = new DeviceSpan[Qwen35Model.StateSlots.Length];
        for (int i = 0; i < state.Length; i++)
        {
            var slot = Qwen35Model.StateSlots[i];
            state[i] = Alloc((slot.Role is "k" or "v" ? 2 : 1) * slot.ElementsPerBlock, slot.Dtype);
        }

        return state;
    }

    private static void SetInt(DeviceSpan span, int index, long value)
    {
        if (span.Dtype == DType.I64)
        {
            ((long*)span.Address)[index] = value;
        }
        else
        {
            ((int*)span.Address)[index] = (int)value;
        }
    }

    /// <summary>Runs <paramref name="ids"/> at positions start.. of the one sequence and returns the last token's logits.</summary>
    private static float[] Step(DeviceSpan[] weights, DeviceSpan[] state, DeviceSpan arena, DeviceSpan planRegion, int[] ids, int start)
    {
        int n = ids.Length;
        int context = start + n;
        int pages = (context + BlockSize - 1) / BlockSize;
        var plan = Qwen35Model.PlanInputs.Bind(planRegion, Layout, n, 1, 1, pages);
        for (int t = 0; t < n; t++)
        {
            SetInt(plan.Ids, t, ids[t]);
            SetInt(plan.Positions, t, start + t);
            SetInt(plan.SlotMapping, t, start + t);
        }

        SetInt(plan.Rows, 0, n - 1);
        SetInt(plan.QIndptr, 0, 0);
        SetInt(plan.QIndptr, 1, n);
        SetInt(plan.KvIndptr, 0, 0);
        SetInt(plan.KvIndptr, 1, pages);
        for (int p = 0; p < pages; p++)
        {
            SetInt(plan.KvIndices, p, p);
        }

        SetInt(plan.KvLastPageLength, 0, context - ((pages - 1) * BlockSize));
        SetInt(plan.StateSlots, 0, 0);
        SetInt(plan.CuSeqlens, 0, 0);
        SetInt(plan.CuSeqlens, 1, n);

        var ctx = new Qwen35Model.StepContext(Layout, arena, weights, state, plan, n, 1, 1, pages);
        Qwen35Model.Forward<CpuOps>(in ctx);
        return new ReadOnlySpan<float>((void*)Qwen35Model.Output_logits(in ctx).Address, Vocab).ToArray();
    }

    [Fact(Skip = "Set ROOST_QWEN35_CHECKPOINT to a directory with the locked model.safetensors-00001-of-00001.safetensors (about 2 GB of memory, about a minute).", SkipUnless = nameof(HasCheckpoint))]
    public void RealWeightsRunThroughTheGeneratedForwardOnCpuOps()
    {
        string dir = Environment.GetEnvironmentVariable("ROOST_QWEN35_CHECKPOINT")!;
        var log = new List<string>();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        using var file = new SafetensorsFile(Path.Combine(dir, FileName));
        var weights = LoadWeights(file);
        log.Add($"weights loaded: {Qwen35Model.WeightSlots.Length} slots, {clock.Elapsed.TotalSeconds:F1} s");

        var arena = Alloc(Layout.ArenaBytes, DType.U8);
        var planRegion = Alloc(Layout.PlanBytes, DType.U8);

        // One prefill of five tokens.
        var state = NewState();
        clock.Restart();
        var whole = Step(weights, state, arena, planRegion, Prompt, 0);
        log.Add($"prefill of {Prompt.Length} tokens: {clock.Elapsed.TotalSeconds:F1} s");

        // Four tokens, then the fifth as a decode step on the kept state.
        state = NewState();
        Step(weights, state, arena, planRegion, Prompt[..4], 0);
        clock.Restart();
        var stepped = Step(weights, state, arena, planRegion, Prompt[4..], 4);
        log.Add($"decode step: {clock.Elapsed.TotalSeconds:F1} s");

        Assert.All(whole, v => Assert.True(float.IsFinite(v)));
        Assert.All(stepped, v => Assert.True(float.IsFinite(v)));
        double min = whole.Min();
        double max = whole.Max();
        Assert.True(max - min > 1e-3, $"logits are flat: [{min}, {max}]");

        int topWhole = Array.IndexOf(whole, whole.Max());
        int topStepped = Array.IndexOf(stepped, stepped.Max());
        var top = whole.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(10).ToArray();
        log.Add("top 10 (prefill): " + string.Join(", ", top.Select(x => string.Create(CultureInfo.InvariantCulture, $"{x.i}:{x.v:F3}"))));
        double dot = 0;
        double nw = 0;
        double ns = 0;
        for (int i = 0; i < Vocab; i++)
        {
            dot += (double)whole[i] * stepped[i];
            nw += (double)whole[i] * whole[i];
            ns += (double)stepped[i] * stepped[i];
        }

        double cosine = dot / Math.Sqrt(nw * ns);
        log.Add(string.Create(CultureInfo.InvariantCulture, $"logits range [{min:F3}, {max:F3}]; argmax prefill {topWhole}, argmax prefill+decode {topStepped}; cosine {cosine:F6}"));
        if (Environment.GetEnvironmentVariable("ROOST_QWEN35_SMOKE_LOG") is { Length: > 0 } path)
        {
            File.WriteAllLines(path, log);
        }

        // The state groups carry across steps: the decode step on kept state matches the longer prefill closely.
        Assert.True(cosine > 0.99, $"cosine {cosine}");
    }
}
