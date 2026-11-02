using OpenAI.Embeddings;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.ApiCompat.Tests.Support;
using Starlings.Roost.Server.Pooling;
using static Starlings.Roost.ApiCompat.Tests.Support.Refusals;

namespace Starlings.Roost.ApiCompat.Tests;

/// <summary>
/// <c>POST /v1/embeddings</c> (R-052) through the official client's <see cref="EmbeddingClient"/>, over a fake embedding model: the
/// client's request (one input or a batch, <c>dimensions</c>, <c>user</c>) is accepted and Roost's response parses.
/// </summary>
public sealed class EmbeddingsCompatTests : IAsyncLifetime
{
    private const string EmbeddingModel = "fake/embedder";
    private const int Dimensions = 8;

    private ApiCompatRig? _rig;

    private ApiCompatRig Rig => _rig!;

    private EmbeddingClient Embeddings => Rig.OpenAi.GetEmbeddingClient(EmbeddingModel);

    public async ValueTask InitializeAsync() => _rig = await ApiCompatRig.StartAsync(
        new Script { Tokens = ApiCompatRig.Encode("ok") },
        pooling: new RoostPoolingModels { Embeddings = [new FakeEmbedder()] });

    public async ValueTask DisposeAsync()
    {
        if (_rig is not null)
        {
            await _rig.DisposeAsync();
        }
    }

    [Fact]
    public async Task OneEmbeddingParsesWithItsVector()
    {
        OpenAIEmbedding embedding = await Embeddings.GenerateEmbeddingAsync("query: hello", cancellationToken: Ct);

        Assert.Equal(0, embedding.Index);
        Assert.Equal(FakeEmbedder.Vector("query: hello"), embedding.ToFloats().ToArray());
    }

    [Fact]
    public async Task ABatchKeepsInputOrderAndReportsUsage()
    {
        string[] inputs = ["a", "bb", "ccc"];
        OpenAIEmbeddingCollection embeddings = await Embeddings.GenerateEmbeddingsAsync(inputs, new EmbeddingGenerationOptions { Dimensions = Dimensions, EndUserId = "u" }, Ct);

        Assert.Equal(EmbeddingModel, embeddings.Model);
        Assert.Equal([0, 1, 2], embeddings.Select(e => e.Index));
        for (var i = 0; i < inputs.Length; i++)
        {
            Assert.Equal(FakeEmbedder.Vector(inputs[i]), embeddings[i].ToFloats().ToArray());
        }

        Assert.Equal(6, embeddings.Usage.InputTokenCount);
        Assert.Equal(6, embeddings.Usage.TotalTokenCount);
    }

    [Fact]
    public async Task OtherDimensionsAndAnotherModelAreRefusedByName()
    {
        await AssertRefusedAsync(() => Embeddings.GenerateEmbeddingAsync("x", new EmbeddingGenerationOptions { Dimensions = 4 }, Ct), 400, "invalid_option", "dimensions");
        await AssertRefusedAsync(() => Rig.OpenAi.GetEmbeddingClient("other").GenerateEmbeddingAsync("x", cancellationToken: Ct), 400, "invalid_option", "model");
    }

    /// <summary>Deterministic vectors from the text; one token per character.</summary>
    private sealed class FakeEmbedder : IRoostEmbeddingModel
    {
        public string ModelId => EmbeddingModel;

        int IRoostEmbeddingModel.Dimensions => Dimensions;

        public static float[] Vector(string text) => [.. Enumerable.Range(0, Dimensions).Select(i => (text.Length * 0.5f) - (i * 0.25f))];

        public RoostEmbeddingResult Embed(IReadOnlyList<string> inputs, CancellationToken cancellationToken) =>
            new([.. inputs.Select(Vector)], inputs.Sum(t => t.Length));
    }
}
