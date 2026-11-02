using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Starlings.Generation.Roost;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Tokenizers;

namespace Starlings.Roost.Server.Tests.Support;

/// <summary>
/// The server over a <see cref="RoostChatClient"/> on the in-process transport with a <see cref="ScriptedExecutor"/> (the
/// pattern of Starlings.Generation.Roost.Tests' TestRig): the tiny byte-level tokenizer (<c>&lt;|im_end|&gt;</c> is 263),
/// Starling's Qwen3.5 template and roost-tiny-hybrid's state groups, with a 256-token context. The HTTP server is ASP.NET
/// Core's in-memory TestServer.
/// </summary>
internal sealed class ServerRig : IAsyncDisposable
{
    public const int StopId = 263;
    public const int AttentionBlocks = 16;
    public const int MaxTokensPerStep = 32;
    public const string ModelId = "roost-tiny-hybrid-test";

    private static readonly Lazy<Tokenizer> s_tokenizer = new(() =>
        Tokenizer.FromFile(Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json")));

    private readonly ScriptedExecutor? _executor;

    private ServerRig(RoostChatClient client, WebApplication app, ScriptedExecutor? executor)
    {
        Client = client;
        App = app;
        _executor = executor;
        Http = app.GetTestClient();
    }

    public static Tokenizer Tokenizer => s_tokenizer.Value;

    public RoostChatClient Client { get; }

    public WebApplication App { get; }

    public ScriptedExecutor Executor => _executor ?? throw new InvalidOperationException("This rig runs the tiny CPU model, not a scripted worker.");

    public HttpClient Http { get; }

    public static int[] Encode(string text) => [.. Tokenizer.Encode(text, false).Ids];

    public static async Task<ServerRig> StartAsync(Script script, RoostServerOptions? options = null)
    {
        ScriptedExecutor? executor = null;
        var client = await RoostChatClient.StartAsync(new RoostChatClientOptions
        {
            Model = Profile(),
            Worker = new RoostWorkerOptions
            {
                Transport = TransportKind.InProcess,
                InProcessExecutor = () => executor = new ScriptedExecutor(script, Limits(), StopId),
                StepDeadline = TimeSpan.FromSeconds(10),
            },
        }, TestContext.Current.CancellationToken);
        return await ServeAsync(client, options, () => executor);
    }

    /// <summary>
    /// The server over a worker that runs roost-tiny-hybrid on <c>CpuOps</c> from a locked weights file (<see cref="TinyCpuExecutor"/>),
    /// for the sleep-mode routes (R-100).
    /// </summary>
    public static async Task<ServerRig> StartTinyAsync(string weightsPath, Starlings.Roost.Worker.Weights.LockedFile locked, RoostServerOptions options)
    {
        var client = await RoostChatClient.StartAsync(new RoostChatClientOptions
        {
            Model = Profile() with { WeightsDigest = [.. Convert.FromHexString(locked.Sha256)] },
            Worker = new RoostWorkerOptions
            {
                Transport = TransportKind.InProcess,
                InProcessExecutor = () => new TinyCpuExecutor(weightsPath, locked, Limits()),
                StepDeadline = TimeSpan.FromSeconds(30),
            },
        }, TestContext.Current.CancellationToken);
        return await ServeAsync(client, options, () => null);
    }

    private static async Task<ServerRig> ServeAsync(RoostChatClient client, RoostServerOptions? options, Func<ScriptedExecutor?> executor)
    {
        try
        {
            var app = RoostServer.Create(client, options, web => web.UseTestServer());
            await app.StartAsync(TestContext.Current.CancellationToken);
            return new ServerRig(client, app, executor());
        }
        catch
        {
            await client.DisposeAsync();
            throw;
        }
    }

    private static RoostModelProfile Profile() => new()
    {
        ModelId = ModelId,
        Tokenizer = Tokenizer,
        ChatTemplate = Qwen35RoostChatTemplate.Instance,
        StopTokenIds = [StopId],
        StateGroups = RoostTinyHybrid.CreateStateGroups(),
        AttentionBlocks = AttentionBlocks,
        RecurrentSlots = 1,
        MaxTokensPerStep = MaxTokensPerStep,
        MaxSequencesPerStep = 1,
        ManifestDigest = ScriptedExecutor.ManifestDigest,
        WeightsDigest = ScriptedExecutor.WeightsDigest,
    };

    /// <summary>The eager worker's limits for the profile above.</summary>
    public static PlanLimits Limits() => new(
        MaxTokens: MaxTokensPerStep,
        MaxSequences: 1,
        MaxKvBlocks: AttentionBlocks,
        MaxSelected: 1,
        Vocab: Tokenizer.GetVocabularySize(true),
        BlockSize: 16,
        AttentionBlocks: AttentionBlocks,
        RecurrentSlots: 1,
        AttentionGroup: 0,
        GroupCount: 2);

    public async ValueTask DisposeAsync()
    {
        Http.Dispose();
        await App.StopAsync(CancellationToken.None);
        await App.DisposeAsync();
        await Client.DisposeAsync();
    }
}
