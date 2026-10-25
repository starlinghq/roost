using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Generation;
using Starlings.Roost.Kernels;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.Worker.Tests.Support;
using Starlings.Roost.Worker.Weights;
using Starlings.Roost.WorkerHost;
using Starlings.Tokenizers;

// R-036 GPU run: Qwen3.5-0.8B through RoostChatClient, with the eager worker (R-025) in a worker process (the default transport,
// decision 21). This executable is both the host and, re-run with WorkerEntry.SegmentFlag, the worker. Run it through
// tools/gpu-lease (run.sh) so CUDA_VISIBLE_DEVICES selects the leased device; the worker inherits it as ordinal 0.
//   usage: <exe> --ckpt <dir> --lock <lock.toml> --manifest <manifest.toml> --kernels <dir> --tokenizer <tokenizer.json> [--allow-untested]
const string WeightsFile = "model.safetensors-00001-of-00001.safetensors";

string Opt(string name) => args.SkipWhile(a => a != name).Skip(1).FirstOrDefault() ?? throw new ArgumentException($"missing {name}");

// Capacities shared by the host profile and the worker: they must agree, or the worker refuses the plans.
var eager = new EagerWorkerOptions
{
    Device = 0,
    MaxTokens = 64,
    MaxSequences = 1,
    MaxKvBlocks = 32,
    AttentionBlocks = 32,
    RecurrentSlots = 1,
    KernelDirectory = Opt("--kernels"),
    AllowUntested = args.Contains("--allow-untested"),
};
var manifest = KernelManifest.Load(Opt("--manifest"));
var locked = WeightLock.Load(Opt("--lock")).Get(WeightsFile);

if (WorkerEntry.IsWorkerInvocation(args))
{
    return WorkerEntry.Run(args, _ => new EagerStepExecutor(new Qwen35EagerModel(), eager, manifest, Path.Combine(Opt("--ckpt"), WeightsFile), locked));
}

// Provenance: what this run loaded, by digest (the worker picks the library of its device's SM from --kernels).
foreach (var lib in Directory.GetFiles(Opt("--kernels"), "libroost_kernels_sm*.so").Order(StringComparer.Ordinal))
{
    Console.WriteLine($"kernel library {Path.GetFileName(lib)} sha256 {Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(lib)))}");
}

Console.WriteLine($"manifest sha256 {manifest.Sha256.ToLowerInvariant()}; weights sha256 {locked.Sha256.ToLowerInvariant()}; CUDA_VISIBLE_DEVICES {Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}");
var tokenizer = Tokenizer.FromFile(Opt("--tokenizer"));
int StopId(string token) => tokenizer.TokenToId(token) ?? throw new InvalidOperationException($"{token} is not in the tokenizer");
var profile = new RoostModelProfile
{
    ModelId = "qwen3.5-0.8b",
    Tokenizer = tokenizer,
    ChatTemplate = Qwen35RoostChatTemplate.Instance,
    StopTokenIds = [StopId("<|im_end|>"), StopId("<|endoftext|>")],
    StateGroups = Qwen35Model.CreateStateGroups(),
    AttentionBlocks = eager.AttentionBlocks,
    RecurrentSlots = eager.RecurrentSlots,
    MaxTokensPerStep = eager.MaxTokens,
    MaxSequencesPerStep = eager.MaxSequences,
    MaxKvBlocksPerStep = eager.MaxKvBlocks,
    ManifestDigest = ImmutableArray.Create(Convert.FromHexString(manifest.Sha256)),
    WeightsDigest = ImmutableArray.Create(Convert.FromHexString(locked.Sha256)),
};

var clock = Stopwatch.StartNew();
await using var client = await RoostChatClient.StartAsync(new RoostChatClientOptions
{
    Model = profile,
    Worker = new RoostWorkerOptions { Command = WorkerCommand.ForCurrentProcess([.. args]) },
});
Console.WriteLine($"worker ready in {clock.Elapsed.TotalSeconds:F1} s; context {profile.ContextTokens} tokens");

string[] prompts =
[
    "What is the capital of France? Answer in one word.",
    "Write one sentence about the sea.",
    "List three prime numbers.",
];
var failures = 0;
foreach (var prompt in prompts)
{
    ChatMessage[] messages = [new(ChatRole.User, prompt)];
    var options = new ChatOptions { MaxOutputTokens = 48 };
    clock.Restart();
    var response = await client.GetResponseAsync(messages, options);
    var seconds = clock.Elapsed.TotalSeconds;
    var streamed = new System.Text.StringBuilder();
    ChatFinishReason? finish = null;
    await foreach (var update in client.GetStreamingResponseAsync(messages, options))
    {
        streamed.Append(update.Text);
        finish ??= update.FinishReason;
    }

    var same = streamed.ToString() == response.Text && finish == response.FinishReason;
    failures += same ? 0 : 1;
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"[{(same ? "ok" : "MISMATCH")}] {prompt}\n  -> {response.Text.ReplaceLineEndings(" ")}\n  finish {response.FinishReason}, usage {response.Usage?.InputTokenCount}+{response.Usage?.OutputTokenCount}, {seconds:F2} s"));
}

// Cancellation on the GPU path: cancel after the first update; the stream must end with OperationCanceledException, and a
// following request must run. A cancel that is not observed fails the run.
var cancelled = false;
using (var cts = new CancellationTokenSource())
{
    try
    {
        await foreach (var _ in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "Count from one to one hundred.")], new ChatOptions { MaxOutputTokens = 200 }, cts.Token))
        {
            await cts.CancelAsync();
        }
    }
    catch (OperationCanceledException)
    {
        Console.WriteLine("cancelled mid-stream");
        cancelled = true;
    }
}

if (!cancelled)
{
    Console.WriteLine("FAIL: the cancel was not observed (no OperationCanceledException)");
    failures++;
}

var after = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Say hello.")], new ChatOptions { MaxOutputTokens = 8 });
Console.WriteLine($"after cancel: {after.Text.ReplaceLineEndings(" ")} ({after.FinishReason})");
if (string.IsNullOrWhiteSpace(after.Text))
{
    Console.WriteLine("FAIL: the request after the cancel returned no text");
    failures++;
}

Console.WriteLine(failures == 0 ? "PASS" : $"FAIL: {failures} failed check(s)");
return failures == 0 ? 0 : 1;
