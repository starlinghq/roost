// Roost chat console (R-062 sample): Qwen3.5-0.8B through Microsoft.Extensions.AI's IChatClient, served by RoostChatClient.
// The client starts the packaged roost-worker as a separate process (the default, decision 21), so CUDA, the kernel library
// and the weights load there and never in this process.
//
//   dotnet run -- --model <dir> --kernels <dir> [--device <ordinal>] [--allow-untested-sm]
//
// <dir> holds the locked Qwen/Qwen3.5-0.8B checkpoint (tokenizer.json and model.safetensors-00001-of-00001.safetensors);
// --kernels holds libroost_kernels_sm<N>.so for this GPU. Type a message and press Enter; /reset clears the conversation,
// /exit (or an empty line) quits.
using System.Collections.Immutable;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Generation;
using Starlings.Roost.Models.Qwen35;
using Starlings.Tokenizers;

const string WeightsFile = "model.safetensors-00001-of-00001.safetensors";

// The weights Roost's Qwen3.5-0.8B profile is locked to: Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17,
// size and SHA-256 from evidence/registry/qwen3.5-0.8b.lock.toml. The worker checks the file against this digest as it reads it.
const long WeightsSize = 1_746_942_600;
const string WeightsSha256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";

string? Option(string name) => args.SkipWhile(a => a != name).Skip(1).FirstOrDefault();

string? modelDirectory = Option("--model");
string? kernelDirectory = Option("--kernels");
if (modelDirectory is null || kernelDirectory is null)
{
    Console.Error.WriteLine("usage: ChatConsole --model <checkpoint dir> --kernels <kernel library dir> [--device <ordinal>] [--allow-untested-sm]");
    return 2;
}

string weightsPath = Path.Combine(modelDirectory, WeightsFile);
if (!File.Exists(weightsPath) || new FileInfo(weightsPath).Length != WeightsSize)
{
    Console.Error.WriteLine($"{weightsPath} is missing or is not the locked Qwen3.5-0.8B weights ({WeightsSize} bytes).");
    return 2;
}

var tokenizer = Tokenizer.FromFile(Path.Combine(modelDirectory, "tokenizer.json"));
int StopId(string token) => tokenizer.TokenToId(token) ?? throw new InvalidOperationException($"{token} is not in the tokenizer.");

// The host's view of the model. The packaged worker is started with these capacities, so the two always agree.
var profile = new RoostModelProfile
{
    ModelId = "qwen3.5-0.8b",
    Tokenizer = tokenizer,
    // Starling's published Qwen35ChatTemplate, through Roost's message mapping.
    ChatTemplate = Qwen35RoostChatTemplate.Instance,
    StopTokenIds = [StopId("<|im_end|>"), StopId("<|endoftext|>")],
    StateGroups = Qwen35Model.CreateStateGroups(),
    AttentionBlocks = 128, // 128 blocks of 16 tokens: a 2048-token conversation
    RecurrentSlots = 1,
    MaxTokensPerStep = 128,
    MaxSequencesPerStep = 1,
    // The kernel manifest shipped beside the packaged worker; the worker's handshake checks it.
    ManifestDigest = ImmutableArray.Create(RoostWorkerRuntime.ManifestDigest()),
    WeightsDigest = ImmutableArray.Create(Convert.FromHexString(WeightsSha256)),
};

var options = new RoostChatClientOptions
{
    Model = profile,
    Worker = new RoostWorkerOptions
    {
        // Transport is left at its default: a separate worker process (decision 21).
        DeviceOrdinal = int.TryParse(Option("--device"), out var device) ? device : 0,
        Launch = new RoostWorkerLaunch
        {
            WeightsPath = weightsPath,
            WeightsSize = WeightsSize,
            KernelDirectory = kernelDirectory,
            AllowUntestedSm = args.Contains("--allow-untested-sm"),
        },
    },
};

Console.WriteLine("Starting the worker (process spawn, handshake, weight load)...");
await using var roost = await RoostChatClient.StartAsync(options);
IChatClient client = roost;
Console.WriteLine($"Ready: {profile.ModelId}, {profile.ContextTokens}-token context. Greedy decoding only; see the package README for what is supported.");

List<ChatMessage> history = [];
var chatOptions = new ChatOptions { MaxOutputTokens = 256 };
while (true)
{
    Console.Write("> ");
    string? line = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(line) || line.Trim() == "/exit")
    {
        return 0;
    }

    if (line.Trim() == "/reset")
    {
        history.Clear();
        Console.WriteLine("(conversation cleared)");
        continue;
    }

    history.Add(new ChatMessage(ChatRole.User, line));
    var reply = new System.Text.StringBuilder();
    try
    {
        await foreach (var update in client.GetStreamingResponseAsync(history, chatOptions))
        {
            Console.Write(update.Text);
            reply.Append(update.Text);
        }
    }
    catch (RoostChatException e)
    {
        // A typed error (C6): for example the conversation no longer fits the context. The turn is dropped.
        history.RemoveAt(history.Count - 1);
        Console.WriteLine($"\n[{e.Error.Code}] {e.Message} (type /reset to start over)");
        continue;
    }

    Console.WriteLine();
    history.Add(new ChatMessage(ChatRole.Assistant, reply.ToString()));
}
