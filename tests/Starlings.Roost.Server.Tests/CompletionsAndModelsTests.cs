using System.Net;
using System.Text.Json;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary><c>POST /v1/completions</c> (a raw prompt, no chat template) and <c>GET /v1/models</c>.</summary>
public sealed class CompletionsAndModelsTests
{
    [Fact]
    public async Task ACompletionTokenisesThePromptAsGiven()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(" four") });
        const string prompt = "two plus two is";

        var (response, body) = await PostAsync(rig.Http, "/v1/completions", $$"""{"model":"roost-tiny-hybrid-test","prompt":["{{prompt}}"],"echo":false,"best_of":1,"suffix":""}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text_completion", body.GetProperty("object").GetString());
        Assert.StartsWith("cmpl-", body.GetProperty("id").GetString(), StringComparison.Ordinal);
        var choice = Assert.Single(body.GetProperty("choices").EnumerateArray());
        Assert.Equal(" four", choice.GetProperty("text").GetString());
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
        Assert.Equal(ServerRig.Encode(prompt).Length, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(ServerRig.Encode(prompt).Length, rig.Executor.Steps.First().ChunkTokens[0]);
    }

    [Fact]
    public async Task MaxTokensDefaultsTo16()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("z"), NeverStop = true });

        var (_, body) = await PostAsync(rig.Http, "/v1/completions", """{"model":"roost-tiny-hybrid-test","prompt":"x"}""");

        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal(16, body.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
        Assert.Equal(new string('z', 16), body.GetProperty("choices")[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task AStreamedCompletionSendsTextChunksThenUsageThenDone()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("abcdef") });

        var response = await rig.Http.PostAsync("/v1/completions",
            Json("""{"model":"roost-tiny-hybrid-test","prompt":"x","stream":true,"stream_options":{"include_usage":true},"max_tokens":4}"""), Ct);
        var events = Events(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        Assert.Equal("[DONE]", events[^1]);
        var chunks = events[..^1].Select(e => JsonDocument.Parse(e).RootElement).ToList();
        Assert.All(chunks, c => Assert.Equal("text_completion", c.GetProperty("object").GetString()));
        Assert.Equal(4, chunks[^1].GetProperty("usage").GetProperty("completion_tokens").GetInt32());
        Assert.Empty(chunks[^1].GetProperty("choices").EnumerateArray());
        Assert.Equal("length", chunks[^2].GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal("abcd", string.Concat(chunks[..^1].Select(c => c.GetProperty("choices")[0].GetProperty("text").GetString())));
    }

    [Fact]
    public async Task ModelsListsTheServedModel()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var response = await rig.Http.GetAsync("/v1/models", Ct);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("list", doc.RootElement.GetProperty("object").GetString());
        var model = Assert.Single(doc.RootElement.GetProperty("data").EnumerateArray());
        Assert.Equal(ServerRig.ModelId, model.GetProperty("id").GetString());
        Assert.Equal("model", model.GetProperty("object").GetString());
        Assert.Equal("roost", model.GetProperty("owned_by").GetString());
        Assert.True(model.GetProperty("created").GetInt64() > 0);
    }
}
