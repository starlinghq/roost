using System.Globalization;
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
public sealed class Qwen35SmokeTests
{
    // "The capital of France is" (ids decoded from the snapshot's tokenizer.json).
    private static readonly int[] Prompt = [760, 6511, 314, 9338, 369];

    // The top two next tokens of the locked snapshot, " Paris" (14.875) then " the" (14.596); transformers fp32 gives 14.871 and
    // 14.587 (R-022). Before R-022 bound WindowLeft = -1 the attention saw only the current token and " the" won.
    private const int TheToken = 279;
    private const int ParisToken = 11751;
    private const int Vocab = Qwen35CpuRunner.Vocab;

    public static bool HasCheckpoint =>
        Environment.GetEnvironmentVariable("ROOST_QWEN35_CHECKPOINT") is { Length: > 0 } dir && File.Exists(Path.Combine(dir, Qwen35CpuRunner.FileName));

    [Fact(Skip = "Set ROOST_QWEN35_CHECKPOINT to a directory with the locked model.safetensors-00001-of-00001.safetensors (about 2 GB of memory, about a minute).", SkipUnless = nameof(HasCheckpoint))]
    public void RealWeightsRunThroughTheGeneratedForwardOnCpuOps()
    {
        string dir = Environment.GetEnvironmentVariable("ROOST_QWEN35_CHECKPOINT")!;
        var log = new List<string>();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        using var file = new SafetensorsFile(Path.Combine(dir, Qwen35CpuRunner.FileName));
        using var runner = new Qwen35CpuRunner(file, maxTokens: 8, maxContext: 32);
        log.Add($"weights loaded: {Qwen35Model.WeightSlots.Length} slots, {clock.Elapsed.TotalSeconds:F1} s");

        // One prefill of five tokens.
        clock.Restart();
        var whole = runner.Step<CpuOps>(Prompt, 0);
        log.Add($"prefill of {Prompt.Length} tokens: {clock.Elapsed.TotalSeconds:F1} s");

        // Four tokens, then the fifth as a decode step on the kept state.
        runner.ResetState();
        runner.Step<CpuOps>(Prompt.AsSpan(0, 4), 0);
        clock.Restart();
        var stepped = runner.Step<CpuOps>(Prompt.AsSpan(4), 4);
        log.Add($"decode step: {clock.Elapsed.TotalSeconds:F1} s");

        Assert.All(whole, v => Assert.True(float.IsFinite(v)));
        Assert.All(stepped, v => Assert.True(float.IsFinite(v)));
        double min = whole.Min();
        double max = whole.Max();
        Assert.True(max - min > 1e-3, $"logits are flat: [{min}, {max}]");

        int topWhole = Array.IndexOf(whole, whole.Max());
        int topStepped = Array.IndexOf(stepped, stepped.Max());
        var top = whole.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(10).ToArray();
        log.Add("prompt \"The capital of France is\" ids " + string.Join(" ", Prompt));
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

        // Regression guard for the gating ops (before them the argmax was a nonsense token) and for the attention window (R-022).
        Assert.Equal(ParisToken, topWhole);
        Assert.Equal(ParisToken, topStepped);
        Assert.Equal(TheToken, top[1].i);

        // The state groups carry across steps: the decode step on kept state matches the longer prefill closely.
        Assert.True(cosine > 0.99, $"cosine {cosine}");
    }
}
