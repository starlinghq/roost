using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Roost.Models.Qwen35.Tests.Parity;
using Starlings.Tokenizers;

namespace Starlings.Roost.Models.Qwen35.Tests;

/// <summary>
/// R-022: the generated Qwen3.5-0.8B forward on <see cref="Ops.CpuOps"/> (double arithmetic, bfloat16 weights and activations as
/// the definition declares them), teacher-forced on a transformers fp32 reference run and captured per layer in the R-011 layout,
/// so that <c>roost-evidence evaluate-layers</c> compares the two. Both facts are opt-in: they read the locked checkpoint
/// (ROOST_QWEN35_CHECKPOINT) and write run artifacts outside the repository (ROOST_R022_OUT). The procedure, revisions and results
/// are in <c>docs/m0/roost-r022-cpu-parity.md</c>.
/// </summary>
public sealed class Qwen35CpuParityRun
{
    private const string SkipPrompts = "Set ROOST_QWEN35_CHECKPOINT (the locked snapshot directory) and ROOST_R022_OUT (an output directory outside the repository).";
    private const string SkipCapture = "Set ROOST_QWEN35_CHECKPOINT, ROOST_R022_REFERENCE (a reference-run directory with --capture-layers) and ROOST_R022_OUT; about 20 minutes and 4 GB.";

    private static readonly JsonSerializerOptions Compact = new() { WriteIndented = false };

    private static string? Env(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } v ? v : null;

    public static bool CanWritePrompts =>
        Env("ROOST_QWEN35_CHECKPOINT") is { } dir && File.Exists(Path.Combine(dir, "tokenizer.json")) && Env("ROOST_R022_OUT") is not null;

    public static bool CanCapture =>
        Env("ROOST_QWEN35_CHECKPOINT") is { } dir && File.Exists(Path.Combine(dir, Qwen35CpuRunner.FileName))
        && Env("ROOST_R022_REFERENCE") is { } reference && File.Exists(Path.Combine(reference, "reference.jsonl"))
        && Env("ROOST_R022_OUT") is not null;

    /// <summary>Renders the prompt set with Starling's template and tokenises it with Starling's tokenizer over the snapshot's
    /// tokenizer.json; writes <c>prompts.jsonl</c> (the reference runner's input, prompt ids only) and <c>rendered.jsonl</c>
    /// (messages and text, for the comparison with the checkpoint's own Jinja template).</summary>
    [Fact(Skip = SkipPrompts, SkipUnless = nameof(CanWritePrompts))]
    public void WritesPromptSet()
    {
        string dir = Env("ROOST_QWEN35_CHECKPOINT")!;
        string outDir = Env("ROOST_R022_OUT")!;
        Directory.CreateDirectory(outDir);
        var tokenizer = Tokenizer.FromFile(Path.Combine(dir, "tokenizer.json"));
        var prompts = new List<string>();
        var rendered = new List<string>();
        foreach (var (id, messages) in R022PromptSet.Cases)
        {
            string text = R022PromptSet.Render(messages);
            int[] ids = [.. tokenizer.Encode(text, false).Ids];
            Assert.NotEmpty(ids);
            prompts.Add(new JsonObject
            {
                ["id"] = id,
                ["prompt_ids"] = new JsonArray([.. ids.Select(i => (JsonNode)i)]),
                ["max_new_tokens"] = R022PromptSet.MaxNewTokens,
            }.ToJsonString(Compact));
            rendered.Add(new JsonObject
            {
                ["id"] = id,
                ["enable_thinking"] = R022PromptSet.EnableThinking,
                ["messages"] = new JsonArray([.. messages.Select(m => (JsonNode)new JsonObject { ["role"] = m.Role.ToString().ToLowerInvariant(), ["content"] = m.Content })]),
                ["rendered"] = text,
                ["prompt_ids"] = new JsonArray([.. ids.Select(i => (JsonNode)i)]),
            }.ToJsonString(Compact));
        }

        File.WriteAllLines(Path.Combine(outDir, "prompts.jsonl"), prompts, LayerCaptureFile.Utf8);
        File.WriteAllLines(Path.Combine(outDir, "rendered.jsonl"), rendered, LayerCaptureFile.Utf8);
    }

    /// <summary>Teacher-forces every reference case (or ROOST_R022_CASES, comma-separated) through the generated forward on
    /// <see cref="RecordingOps"/> and writes <c>layers/&lt;id&gt;.safetensors</c>, <c>step0/&lt;id&gt;.f32</c> (log-softmax),
    /// <c>candidate.jsonl</c> (argmax and margin per forward) and <c>run.json</c>. ROOST_R022_PERTURB names a deliberate defect.</summary>
    [Fact(Skip = SkipCapture, SkipUnless = nameof(CanCapture))]
    public void TeacherForcedCapture()
    {
        string dir = Env("ROOST_QWEN35_CHECKPOINT")!;
        string referenceDir = Env("ROOST_R022_REFERENCE")!;
        string outDir = Env("ROOST_R022_OUT")!;
        var perturbation = Perturbations.Parse(Env("ROOST_R022_PERTURB"));
        var only = Env("ROOST_R022_CASES")?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet(StringComparer.Ordinal);
        Qwen35Model.CheckBackend<RecordingOps>();

        var cases = new List<(string Id, int[] Prompt, int[] Output)>();
        foreach (string line in File.ReadLines(Path.Combine(referenceDir, "reference.jsonl")))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            string id = root.GetProperty("id").GetString()!;
            if (only is null || only.Contains(id))
            {
                cases.Add((id, [.. root.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32())], [.. root.GetProperty("output_ids").EnumerateArray().Select(e => e.GetInt32())]));
            }
        }

        Assert.NotEmpty(cases);
        int maxTokens = cases.Max(c => c.Prompt.Length);
        int maxContext = cases.Max(c => c.Prompt.Length + c.Output.Length);
        var clock = Stopwatch.StartNew();
        using var file = new SafetensorsFile(Path.Combine(dir, Qwen35CpuRunner.FileName));
        using var runner = new Qwen35CpuRunner(file, maxTokens, maxContext);
        double loadSeconds = clock.Elapsed.TotalSeconds;
        Directory.CreateDirectory(Path.Combine(outDir, "step0"));
        var summaries = new List<string>();
        var digests = new JsonObject();
        foreach (var (id, prompt, output) in cases)
        {
            var metadata = LayerCaptureFile.ReadMetadata(Path.Combine(referenceDir, "layers", id + ".safetensors"));
            int[] decode = LayerCaptureFile.Ints(metadata["decode_ids"]);
            Assert.Equal(prompt, LayerCaptureFile.Ints(metadata["prompt_ids"]));
            Assert.Equal(output[..^1], decode);
            int stateForwards = int.Parse(metadata["state_forwards"], CultureInfo.InvariantCulture);
            var recorder = new StageRecorder(LayerCaptureFile.Strings(metadata["layer_types"]), stateForwards, perturbation);
            runner.ResetState();
            var argmax = new JsonArray();
            var margins = new JsonArray();
            var seconds = new JsonArray();
            StageRecorder.Active = recorder;
            try
            {
                int position = 0;
                for (int forward = 0; forward <= decode.Length; forward++)
                {
                    int[] ids = forward == 0 ? prompt : [decode[forward - 1]];
                    clock.Restart();
                    float[] logits = runner.Step<RecordingOps>(ids, position);
                    recorder.EndForward();
                    seconds.Add(Math.Round(clock.Elapsed.TotalSeconds, 2));
                    position += ids.Length;
                    var (best, margin, logprobs) = LogSoftmax(logits);
                    argmax.Add(best);
                    margins.Add(margin);
                    if (forward == 0)
                    {
                        using var step0 = File.Create(Path.Combine(outDir, "step0", id + ".f32"));
                        LayerCaptureFile.WriteFloats(step0, logprobs);
                    }
                }
            }
            finally
            {
                StageRecorder.Active = null;
            }

            string digest = LayerCaptureFile.Write(Path.Combine(outDir, "layers", id + ".safetensors"), metadata, recorder.Tensors);
            digests[$"layers/{id}.safetensors"] = digest;
            summaries.Add(new JsonObject
            {
                ["id"] = id,
                ["perturbation"] = Perturbations.Name(perturbation),
                ["prompt_tokens"] = prompt.Length,
                ["forwards"] = decode.Length + 1,
                ["reference_output_ids"] = new JsonArray([.. output.Select(i => (JsonNode)i)]),
                ["argmax"] = argmax,
                ["margin"] = margins,
                ["seconds"] = seconds,
                ["layers_sha256"] = digest,
            }.ToJsonString(Compact));
        }

        File.WriteAllLines(Path.Combine(outDir, "candidate.jsonl"), summaries, LayerCaptureFile.Utf8);
        var run = new JsonObject
        {
            ["candidate"] = "Qwen35Model.Forward<RecordingOps> over CpuOps (double arithmetic; bfloat16 weights, activations and convolution state; float32 recurrent state)",
            ["perturbation"] = Perturbations.Name(perturbation),
            ["checkpoint_file"] = Path.Combine(dir, Qwen35CpuRunner.FileName),
            ["reference"] = referenceDir,
            ["weights_load_seconds"] = Math.Round(loadSeconds, 1),
            ["dotnet"] = Environment.Version.ToString(),
            ["captures"] = digests,
        };
        File.WriteAllText(Path.Combine(outDir, "run.json"), run.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n", LayerCaptureFile.Utf8);
    }

    /// <summary>Log-softmax in double, rounded to float32; the argmax (lowest id on a tie) and the gap to the runner-up.</summary>
    internal static (int Best, double Margin, float[] LogProbs) LogSoftmax(float[] logits)
    {
        double max = double.NegativeInfinity;
        int best = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
                best = i;
            }
        }

        double second = double.NegativeInfinity;
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += Math.Exp(logits[i] - max);
            if (i != best && logits[i] > second)
            {
                second = logits[i];
            }
        }

        double log = max + Math.Log(sum);
        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = (float)(logits[i] - log);
        }

        return (best, max - second, result);
    }
}
