using System.ClientModel;
using System.ClientModel.Primitives;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Models;
using OpenAI.Responses;
using Starlings.Generation.Roost;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Server;
using Starlings.Roost.Server.Pooling;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Tokenizers;

namespace Starlings.Roost.ApiCompat.Tests.Support;

/// <summary>
/// The real server on Kestrel (loopback, port 0, as <c>roost serve</c> listens) over a <see cref="RoostChatClient"/> on the
/// in-process transport with a <see cref="ScriptedExecutor"/>, and the official OpenAI .NET client pointed at it. The profile is
/// the one of Starlings.Roost.Server.Tests' ServerRig: the tiny byte-level tokenizer (<c>&lt;|im_end|&gt;</c> is 263),
/// Starling's Qwen3.5 template and roost-tiny-hybrid's state groups.
/// </summary>
internal sealed class ApiCompatRig : IAsyncDisposable
{
    public const int StopId = 263;
    public const int AttentionBlocks = 16;
    public const int MaxTokensPerStep = 32;
    public const string ModelId = "roost-tiny-hybrid-test";

    private static readonly Lazy<Tokenizer> s_tokenizer = new(() =>
        Tokenizer.FromFile(Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json")));

    private ApiCompatRig(RoostChatClient client, WebApplication app, ScriptedExecutor executor, Uri endpoint)
    {
        Client = client;
        App = app;
        Executor = executor;
        Endpoint = endpoint;
        OpenAi = new OpenAIClient(new ApiKeyCredential("unused"), new OpenAIClientOptions
        {
            // The client appends route paths to the endpoint, as it does for https://api.openai.com/v1.
            Endpoint = new Uri(endpoint, "v1"),
            // No retries: a refusal is asserted as the server sent it, once.
            RetryPolicy = new ClientRetryPolicy(maxRetries: 0),
            NetworkTimeout = TimeSpan.FromSeconds(30),
        });
    }

    public static Tokenizer Tokenizer => s_tokenizer.Value;

    public RoostChatClient Client { get; }

    public WebApplication App { get; }

    public ScriptedExecutor Executor { get; }

    /// <summary>The Kestrel listener, for example <c>http://127.0.0.1:41234/</c>.</summary>
    public Uri Endpoint { get; }

    /// <summary>The official client, with its endpoint at <c>/v1</c> of <see cref="Endpoint"/>.</summary>
    public OpenAIClient OpenAi { get; }

    public ChatClient Chat => OpenAi.GetChatClient(ModelId);

    public ResponsesClient Responses => OpenAi.GetResponsesClient();

    public OpenAIModelClient Models => OpenAi.GetOpenAIModelClient();

    public static int[] Encode(string text) => [.. Tokenizer.Encode(text, false).Ids];

    /// <summary>
    /// Starts the server; <paramref name="attentionBlocks"/> enlarges the context (a tools prompt needs about 2,000 byte tokens) and
    /// <paramref name="pooling"/> sets the pooling models (R-052), none by default.
    /// </summary>
    public static async Task<ApiCompatRig> StartAsync(Script script, int attentionBlocks = AttentionBlocks, RoostPoolingModels? pooling = null)
    {
        ScriptedExecutor? executor = null;
        var client = await RoostChatClient.StartAsync(new RoostChatClientOptions
        {
            Model = new RoostModelProfile
            {
                ModelId = ModelId,
                Tokenizer = Tokenizer,
                ChatTemplate = Qwen35RoostChatTemplate.Instance,
                StopTokenIds = [StopId],
                StateGroups = RoostTinyHybrid.CreateStateGroups(),
                AttentionBlocks = attentionBlocks,
                RecurrentSlots = 1,
                MaxTokensPerStep = MaxTokensPerStep,
                MaxSequencesPerStep = 1,
                ManifestDigest = ScriptedExecutor.ManifestDigest,
                WeightsDigest = ScriptedExecutor.WeightsDigest,
            },
            Worker = new RoostWorkerOptions
            {
                Transport = TransportKind.InProcess,
                InProcessExecutor = () => executor = new ScriptedExecutor(script, Limits(attentionBlocks), StopId),
                StepDeadline = TimeSpan.FromSeconds(10),
            },
        }, TestContext.Current.CancellationToken);

        try
        {
            // Kestrel's request log at Information would flood the test output; warnings and errors still show.
            var app = RoostServer.Create(client, new RoostServerOptions { Port = 0, Pooling = pooling ?? RoostPoolingModels.None }, web =>
                web.ConfigureServices(services => services.AddLogging(l => l.SetMinimumLevel(LogLevel.Warning))));
            await app.StartAsync(TestContext.Current.CancellationToken);
            var address = Assert.Single(app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses);
            return new ApiCompatRig(client, app, executor!, new Uri(address.TrimEnd('/') + "/"));
        }
        catch
        {
            await client.DisposeAsync();
            throw;
        }
    }

    /// <summary>The eager worker's limits for the profile above.</summary>
    public static PlanLimits Limits(int attentionBlocks = AttentionBlocks) => new(
        MaxTokens: MaxTokensPerStep,
        MaxSequences: 1,
        MaxKvBlocks: attentionBlocks,
        MaxSelected: 1,
        Vocab: Tokenizer.GetVocabularySize(true),
        BlockSize: 16,
        AttentionBlocks: attentionBlocks,
        RecurrentSlots: 1,
        AttentionGroup: 0,
        GroupCount: 2);

    public async ValueTask DisposeAsync()
    {
        await App.StopAsync(CancellationToken.None);
        await App.DisposeAsync();
        await Client.DisposeAsync();
    }
}
