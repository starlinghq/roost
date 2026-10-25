using System.Text.Json;

namespace Starlings.Roost.Models.Qwen35.Tests.Parity;

/// <summary>CI checks of the R-022 harness pieces that need no checkpoint: the capture writer, the log-softmax and the prompt set.</summary>
public sealed class ParityHarnessTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "roost-r022-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, recursive: true);
        }
    }

    [Fact]
    public void CaptureFileIsSafetensorsInNameOrderWithMetadata()
    {
        string path = Path.Combine(_dir, "layers", "case.safetensors");
        var metadata = new Dictionary<string, string> { ["format"] = "roost.layer-capture", ["prompt_ids"] = "[1, 2]" };
        var tensors = new Dictionary<string, (int[] Shape, float[] Values)>
        {
            ["forward0/layer.1"] = ([1, 2], [3f, 4f]),
            ["forward0/embeddings"] = ([1, 2], [1f, 2f]),
            ["forward0/layer.0/conv_state"] = ([2, 1], [5f, -6f]),
        };

        string digest = LayerCaptureFile.Write(path, metadata, tensors);

        Assert.Equal(64, digest.Length);
        Assert.Equal(metadata, LayerCaptureFile.ReadMetadata(path));
        byte[] bytes = File.ReadAllBytes(path);
        long headerLength = (long)BitConverter.ToUInt64(bytes, 0);
        Assert.Equal(0, headerLength % 8);
        using var header = JsonDocument.Parse(bytes.AsMemory(8, (int)headerLength));
        string[] names = [.. header.RootElement.EnumerateObject().Select(p => p.Name).Where(n => n != "__metadata__")];
        Assert.Equal(["forward0/embeddings", "forward0/layer.0/conv_state", "forward0/layer.1"], names.Order(StringComparer.Ordinal));
        Assert.Equal([8, 16], header.RootElement.GetProperty("forward0/layer.0/conv_state").GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt32()));
        float[] data = new float[6];
        Buffer.BlockCopy(bytes, 8 + (int)headerLength, data, 0, 24);
        Assert.Equal([1f, 2f, 5f, -6f, 3f, 4f], data);

        using var file = new SafetensorsFile(path);
        Assert.Equal("F32", file.Info("forward0/layer.1").Dtype);
        Assert.Equal([1L, 2L], file.Info("forward0/layer.1").Shape);
    }

    [Fact]
    public void LogSoftmaxReportsArgmaxMarginAndNormalisedValues()
    {
        var (best, margin, logprobs) = Qwen35CpuParityRun.LogSoftmax([1f, 3f, 2.5f, -1f]);

        Assert.Equal(1, best);
        Assert.Equal(0.5, margin, 6);
        Assert.Equal(1.0, logprobs.Sum(v => Math.Exp(v)), 6);
        Assert.Equal(0.5, logprobs[1] - logprobs[2], 5);
    }

    [Fact]
    public void PromptSetRendersChatPromptsWithAClosedThinkBlock()
    {
        Assert.Equal(5, R022PromptSet.Cases.Count);
        Assert.Equal(R022PromptSet.Cases.Count, R022PromptSet.Cases.Select(c => c.Id).Distinct().Count());
        foreach (var (_, messages) in R022PromptSet.Cases)
        {
            string text = R022PromptSet.Render(messages);
            Assert.StartsWith("<|im_start|>", text, StringComparison.Ordinal);
            Assert.EndsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n", text, StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData(null, "none")]
    [InlineData("drop-attention-gate", "drop-attention-gate")]
    [InlineData("drop-z-gate", "drop-z-gate")]
    [InlineData("norm-eps-1e-5", "norm-eps-1e-5")]
    public void PerturbationNamesRoundTrip(string? text, string name) => Assert.Equal(name, Perturbations.Name(Perturbations.Parse(text)));
}
