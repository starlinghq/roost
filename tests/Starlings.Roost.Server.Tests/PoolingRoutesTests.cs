using System.Buffers.Binary;
using System.Net;
using Starlings.Generation.Roost;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Requests;
using Starlings.Roost.Server.Pooling;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>
/// The pooling routes (R-052) over fake models behind <see cref="IRoostEmbeddingModel"/> and <see cref="IRoostExtractionModel"/>:
/// no model file is read. The Starling-backed models are exercised only for refusing a directory without the locked files.
/// </summary>
public sealed class PoolingRoutesTests
{
    private const string Sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    [Fact]
    public async Task EmbeddingsAnswerOpenAisListWithVllmsIdAndCreated()
    {
        var embedder = new FakeEmbedder("fake/e5", 4);
        await using var rig = await StartAsync(embeddings: [embedder]);

        var (response, body) = await PostAsync(rig.Http, "/v1/embeddings", """{"model":"fake/e5","input":"query: hello"}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(["id", "object", "created", "model", "data", "usage"], body.EnumerateObject().Select(p => p.Name));
        Assert.StartsWith("embd-", body.GetProperty("id").GetString(), StringComparison.Ordinal);
        Assert.Equal("list", body.GetProperty("object").GetString());
        Assert.Equal("fake/e5", body.GetProperty("model").GetString());
        var item = Assert.Single(body.GetProperty("data").EnumerateArray());
        Assert.Equal(["index", "object", "embedding"], item.EnumerateObject().Select(p => p.Name));
        Assert.Equal(0, item.GetProperty("index").GetInt32());
        Assert.Equal("embedding", item.GetProperty("object").GetString());
        Assert.Equal(FakeEmbedder.Vector("query: hello", 4), item.GetProperty("embedding").EnumerateArray().Select(x => x.GetSingle()));
        Assert.Equal(12, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(12, body.GetProperty("usage").GetProperty("total_tokens").GetInt32());
        Assert.Equal(["query: hello"], Assert.Single(embedder.Calls));
    }

    [Fact]
    public async Task EmbeddingsOfAnArrayKeepInputOrderAndEncodeBase64AsLittleEndianFloat32()
    {
        await using var rig = await StartAsync(embeddings: [new FakeEmbedder("fake/e5", 3)]);

        var (response, body) = await PostAsync(rig.Http, "/v1/embeddings", """{"input":["a","bb","ccc"],"encoding_format":"base64","dimensions":3,"user":"u"}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var data = body.GetProperty("data").EnumerateArray().ToList();
        Assert.Equal([0, 1, 2], data.Select(d => d.GetProperty("index").GetInt32()));
        string[] texts = ["a", "bb", "ccc"];
        for (var i = 0; i < 3; i++)
        {
            var bytes = Convert.FromBase64String(data[i].GetProperty("embedding").GetString()!);
            var floats = Enumerable.Range(0, bytes.Length / 4).Select(j => BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(j * 4)));
            Assert.Equal(FakeEmbedder.Vector(texts[i], 3), floats);
        }

        Assert.Equal(6, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
    }

    [Fact]
    public async Task EmbeddingsSelectTheModelByIdAndRequireItWhenMoreThanOneIsServed()
    {
        var first = new FakeEmbedder("fake/e5", 2);
        var second = new FakeEmbedder("fake/gemma", 3);
        await using var rig = await StartAsync(embeddings: [first, second]);

        var (ok, body) = await PostAsync(rig.Http, "/v1/embeddings", """{"model":"fake/gemma","input":"x"}""");
        Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
        Assert.Equal("fake/gemma", body.GetProperty("model").GetString());
        Assert.Equal(3, Assert.Single(body.GetProperty("data").EnumerateArray()).GetProperty("embedding").GetArrayLength());
        Assert.Empty(first.Calls);

        var (missing, missingBody) = await PostAsync(rig.Http, "/v1/embeddings", """{"input":"x"}""");
        Assert.Equal(HttpStatusCode.BadRequest, missing.StatusCode);
        AssertErrorBody(missingBody, "invalid_option", "model");

        var (unknown, unknownBody) = await PostAsync(rig.Http, "/v1/embeddings", """{"model":"other","input":"x"}""");
        Assert.Equal(HttpStatusCode.BadRequest, unknown.StatusCode);
        Assert.Contains("fake/e5, fake/gemma", AssertErrorBody(unknownBody, "invalid_option", "model").GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"input":[1,2,3]}""", "input[0]")]
    [InlineData("""{"input":[[1,2]]}""", "input[0]")]
    [InlineData("""{"input":[]}""", "input")]
    [InlineData("""{"input":5}""", "input")]
    [InlineData("""{"model":"fake/e5"}""", "input")]
    [InlineData("""{"input":"x","encoding_format":"float16"}""", "encoding_format")]
    [InlineData("""{"input":"x","dimensions":2}""", "dimensions")]
    [InlineData("""{"input":"x","dimensions":0}""", "dimensions")]
    [InlineData("""{"input":"x","truncate_prompt_tokens":8}""", "truncate_prompt_tokens")]
    [InlineData("""{"input":"x","normalize":true}""", "normalize")]
    public async Task EmbeddingsRefuseABadOrUnsupportedFieldByName(string json, string param)
    {
        var embedder = new FakeEmbedder("fake/e5", 4);
        await using var rig = await StartAsync(embeddings: [embedder]);

        var (response, body) = await PostAsync(rig.Http, "/v1/embeddings", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
        Assert.Empty(embedder.Calls);
    }

    [Fact]
    public async Task EmbeddingsRefuseMoreInputsThanOpenAisLimit()
    {
        await using var rig = await StartAsync(embeddings: [new FakeEmbedder("fake/e5", 1)]);
        var json = "{\"input\":[" + string.Join(',', Enumerable.Repeat("\"x\"", PoolingRoutes.MaxInputs + 1)) + "]}";

        var (response, body) = await PostAsync(rig.Http, "/v1/embeddings", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", "input");
    }

    [Fact]
    public async Task AnInputTheModelRefusesAsTooLongIsPromptTooLongNamingIt()
    {
        await using var rig = await StartAsync(embeddings: [new FakeEmbedder("fake/e5", 2) { TooLongIndex = 1 }], extraction: [new FakeExtractor("fake/gliner") { TooLongIndex = 0 }]);

        var (embed, embedBody) = await PostAsync(rig.Http, "/v1/embeddings", """{"input":["short","long"]}""");
        Assert.Equal(HttpStatusCode.BadRequest, embed.StatusCode);
        AssertErrorBody(embedBody, "prompt_too_long", "input[1]");

        var (extract, extractBody) = await PostAsync(rig.Http, "/v1/extract", """{"input":"long","labels":["person"]}""");
        Assert.Equal(HttpStatusCode.BadRequest, extract.StatusCode);
        AssertErrorBody(extractBody, "prompt_too_long", "input[0]");
    }

    [Fact]
    public async Task ExtractAnswersLabelledSpansPerInput()
    {
        var extractor = new FakeExtractor("fake/gliner");
        await using var rig = await StartAsync(extraction: [extractor]);

        var (response, body) = await PostAsync(rig.Http, "/v1/extract", """{"model":"fake/gliner","input":["Ada met Bob.","none"],"labels":["person","place"],"threshold":0.4}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(["id", "object", "created", "model", "data", "usage"], body.EnumerateObject().Select(p => p.Name));
        Assert.StartsWith("extract-", body.GetProperty("id").GetString(), StringComparison.Ordinal);
        Assert.Equal("fake/gliner", body.GetProperty("model").GetString());
        var data = body.GetProperty("data").EnumerateArray().ToList();
        Assert.Equal(2, data.Count);
        Assert.Equal("extraction", data[0].GetProperty("object").GetString());
        var entity = Assert.Single(data[0].GetProperty("entities").EnumerateArray());
        Assert.Equal(["label", "text", "start", "end", "confidence"], entity.EnumerateObject().Select(p => p.Name));
        Assert.Equal("person", entity.GetProperty("label").GetString());
        Assert.Equal("Ada", entity.GetProperty("text").GetString());
        Assert.Equal(0, entity.GetProperty("start").GetInt32());
        Assert.Equal(3, entity.GetProperty("end").GetInt32());
        Assert.Equal(0.9f, entity.GetProperty("confidence").GetSingle());
        Assert.Equal(1, data[1].GetProperty("index").GetInt32());
        Assert.Empty(data[1].GetProperty("entities").EnumerateArray());
        Assert.Equal(16, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());

        var call = Assert.Single(extractor.Calls);
        Assert.Equal(["Ada met Bob.", "none"], call.Inputs);
        Assert.Equal(["person", "place"], call.Labels);
        Assert.Equal(0.4f, call.Threshold);
    }

    [Theory]
    [InlineData("""{"input":"x"}""", "labels")]
    [InlineData("""{"input":"x","labels":[]}""", "labels")]
    [InlineData("""{"input":"x","labels":["a","a"]}""", "labels[1]")]
    [InlineData("""{"input":"x","labels":["a",""]}""", "labels[1]")]
    [InlineData("""{"input":"x","labels":[3]}""", "labels[0]")]
    [InlineData("""{"input":"x","labels":["a"],"threshold":1.5}""", "threshold")]
    [InlineData("""{"labels":["a"]}""", "input")]
    [InlineData("""{"input":"x","labels":["a"],"relations":["works_for"]}""", "relations")]
    [InlineData("""{"model":"other","input":"x","labels":["a"]}""", "model")]
    public async Task ExtractRefusesABadOrUnsupportedFieldByName(string json, string param)
    {
        var extractor = new FakeExtractor("fake/gliner");
        await using var rig = await StartAsync(extraction: [extractor]);

        var (response, body) = await PostAsync(rig.Http, "/v1/extract", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
        Assert.Empty(extractor.Calls);
    }

    [Theory]
    [InlineData("/v1/embeddings", "'embeddings'")]
    [InlineData("/v1/extract", "'extract'")]
    public async Task ARouteWithoutAConfiguredModelIsProfileUnsupportedNamingTheCapability(string path, string capability)
    {
        await using var rig = await StartAsync();

        var (response, body) = await PostAsync(rig.Http, path, """{"input":"x","labels":["a"]}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains(capability, AssertErrorBody(body, "profile_unsupported", null).GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("/v1/rerank", "'rerank'", """{"query":"q","documents":["a","b"]}""")]
    [InlineData("/v1/classify", "'classify'", """{"input":"x"}""")]
    public async Task RerankAndClassifyAreProfileUnsupportedBecauseNoPackageProvidesThem(string path, string capability, string json)
    {
        await using var rig = await StartAsync(embeddings: [new FakeEmbedder("fake/e5", 2)], extraction: [new FakeExtractor("fake/gliner")]);

        var (response, body) = await PostAsync(rig.Http, path, json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains(capability, AssertErrorBody(body, "profile_unsupported", null).GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TwoModelsOfOneKindWithOneIdAreRefusedWhenTheServerIsBuilt()
    {
        var options = new RoostServerOptions { Pooling = new RoostPoolingModels { Embeddings = [new FakeEmbedder("same", 2), new FakeEmbedder("same", 3)] } };

        var e = await Assert.ThrowsAsync<ArgumentException>(async () => await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") }, options));

        Assert.Contains("'same'", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ServeParsesThePoolingFlags()
    {
        var c = RoostCli.Parse(["serve", "--model", "qwen3.5-0.8b", "--weights", "w", "--weights-sha256", Sha, "--tokenizer", "t",
            "--pooling-e5", "e5", "--pooling-embeddinggemma2", "gemma", "--pooling-embeddinggemma2-dimensions", "256", "--pooling-gliner2", "gliner"]);

        Assert.Equal(new PoolingPaths("e5", "gemma", 256, "gliner"), c.Pooling);
        Assert.Null(RoostCli.Parse(["serve", "--model", "qwen3.5-0.8b", "--weights", "w", "--weights-sha256", Sha, "--tokenizer", "t"]).Pooling);
    }

    [Theory]
    [InlineData(new[] { "--pooling-embeddinggemma2-dimensions", "256" }, "needs --pooling-embeddinggemma2")]
    [InlineData(new[] { "--pooling-embeddinggemma2", "g", "--pooling-embeddinggemma2-dimensions", "300" }, "is not one of 768, 512, 256 or 128")]
    [InlineData(new[] { "--pooling-bge", "b" }, "unknown option '--pooling-bge'")]
    public void ServeRefusesBadPoolingFlags(string[] extra, string message)
    {
        string[] args = ["serve", "--model", "qwen3.5-0.8b", "--weights", "w", "--weights-sha256", Sha, "--tokenizer", "t", .. extra];

        var e = Assert.Throws<ArgumentException>(() => RoostCli.Parse(args));

        Assert.Contains(message, e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ServeReportsAPoolingSnapshotWithoutItsLockedFilesAndDownloadsNothing()
    {
        var empty = Directory.CreateTempSubdirectory("roost-pooling-");
        try
        {
            using var error = new StringWriter();

            var code = await RoostCli.RunAsync(["serve", "--model", "qwen3.5-0.8b", "--weights", "missing.safetensors", "--weights-sha256", Sha, "--tokenizer", "missing.json",
                "--pooling-gliner2", empty.FullName], error);

            Assert.Equal(1, code);
            Assert.Contains("a pooling model did not load", error.ToString(), StringComparison.Ordinal);
            Assert.DoesNotContain("the worker did not start", error.ToString(), StringComparison.Ordinal);
            Assert.Empty(empty.EnumerateFileSystemInfos());
        }
        finally
        {
            empty.Delete(true);
        }
    }

    [Fact]
    public void EveryStarlingModelRefusesADirectoryWithoutItsLockedFiles()
    {
        var empty = Directory.CreateTempSubdirectory("roost-pooling-");
        try
        {
            Assert.ThrowsAny<Exception>(() => E5EmbeddingModel.Open(empty.FullName));
            Assert.ThrowsAny<Exception>(() => EmbeddingGemma2EmbeddingModel.Open(empty.FullName));
            Assert.ThrowsAny<Exception>(() => Gliner2ExtractionModel.Open(empty.FullName));
            Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2EmbeddingModel.Open(empty.FullName, 300));
            Assert.Empty(empty.EnumerateFileSystemInfos());
        }
        finally
        {
            empty.Delete(true);
        }
    }

    private static Task<ServerRig> StartAsync(IRoostEmbeddingModel[]? embeddings = null, IRoostExtractionModel[]? extraction = null) =>
        ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") }, new RoostServerOptions
        {
            Pooling = new RoostPoolingModels { Embeddings = embeddings ?? [], Extraction = extraction ?? [] },
        });

    /// <summary>Deterministic vectors from the text; one token per character.</summary>
    private sealed class FakeEmbedder(string modelId, int dimensions) : IRoostEmbeddingModel
    {
        public string ModelId => modelId;

        public int Dimensions => dimensions;

        public int? TooLongIndex { get; init; }

        public List<IReadOnlyList<string>> Calls { get; } = [];

        public static float[] Vector(string text, int dimensions) => [.. Enumerable.Range(0, dimensions).Select(i => (text.Length * 0.25f) + (i * 0.125f) - 0.5f)];

        public RoostEmbeddingResult Embed(IReadOnlyList<string> inputs, CancellationToken cancellationToken)
        {
            if (TooLongIndex is { } i)
            {
                throw new RoostChatException(new RoostError(RoostErrorCode.PromptTooLong, "Input too long.", $"input[{i}]"));
            }

            lock (Calls)
            {
                Calls.Add([.. inputs]);
            }

            return new RoostEmbeddingResult([.. inputs.Select(t => Vector(t, dimensions))], inputs.Sum(t => t.Length));
        }
    }

    /// <summary>Finds "Ada" as a person; every text reads eight tokens.</summary>
    private sealed class FakeExtractor(string modelId) : IRoostExtractionModel
    {
        public string ModelId => modelId;

        public int? TooLongIndex { get; init; }

        public List<(IReadOnlyList<string> Inputs, IReadOnlyList<string> Labels, float? Threshold)> Calls { get; } = [];

        public IReadOnlyList<RoostExtraction> Extract(IReadOnlyList<string> inputs, IReadOnlyList<string> labels, float? threshold, CancellationToken cancellationToken)
        {
            if (TooLongIndex is { } i)
            {
                throw new RoostChatException(new RoostError(RoostErrorCode.PromptTooLong, "Input too long.", $"input[{i}]"));
            }

            lock (Calls)
            {
                Calls.Add(([.. inputs], [.. labels], threshold));
            }

            return [.. inputs.Select(t => new RoostExtraction(
                t.StartsWith("Ada", StringComparison.Ordinal) ? [new RoostEntity("person", "Ada", 0, 3, 0.9f)] : [],
                8))];
        }
    }
}
